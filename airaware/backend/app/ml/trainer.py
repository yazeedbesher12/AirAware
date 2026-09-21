from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import MODEL_NAMES
from .dataset import HORIZONS, filter_sensor, load_phase4_dataset
from .evaluation import evaluate, prediction_frame, sensor_metrics
from .gru_model import GRUModel
from .lightgbm_model import LightGBMModel
from .lstm_model import LSTMModel
from .persistence_model import PersistenceModel
from .prophet_model import ProphetModel
from .random_forest_model import RandomForestModel
from .results_manager import ResultsManager
from .reporting import write_phase4_reports
from .tft_model import TFTModel
from .xgboost_model import XGBoostModel


MODEL_FACTORIES = {
    "persistence": PersistenceModel, "random_forest": RandomForestModel, "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel, "prophet": ProphetModel, "lstm": LSTMModel, "gru": GRUModel, "tft": TFTModel,
}
EXTENSIONS = {"persistence": ".json", "prophet": ".json", "lstm": ".pt", "gru": ".pt", "tft": ".ckpt"}


class Phase4Trainer:
    def __init__(self, backend_dir: Path):
        self.backend_dir = backend_dir
        self.manager = ResultsManager(backend_dir)
        self.dataset = load_phase4_dataset(self.manager.outputs)

    def run(self, models: list[str], horizons: list[str], sensor_id: int | None = None, target: str = "pm25") -> dict[str, Any]:
        if target != "pm25": raise ValueError("Phase 3 contains valid forecasting targets only for pm25")
        metadata = self.manager.training_metadata(self.manager.outputs / "AirAware_ML_Features.csv", models)
        self.manager.write_json("training_run.json", metadata)
        summary = self.manager.load_json("model_results.json", {"project": "AirAware", "phase": "4A", "generated_at": None, "models": {}})
        detailed = self.manager.load_json("model_results_detailed.json", [])
        detailed = [row for row in detailed if row.get("model") not in models or row.get("horizon") not in horizons or row.get("sensor_filter") != sensor_id]
        for model_name in models:
            logger = self._logger(model_name)
            previous = summary.get("models", {}).get(model_name, {})
            model_summary = {"status": "completed", "horizons": dict(previous.get("horizons", {})), "warnings": []}
            for horizon in horizons:
                try:
                    record = self._run_one(model_name, horizon, sensor_id, logger)
                    detailed.extend(record)
                    overall = next(row for row in record if row["sensor_scope"] == "all")
                    model_summary["horizons"][horizon] = overall["metrics"]
                except Exception as exc:
                    logger.exception("Training failed for %s %s", model_name, horizon)
                    model_summary["status"] = "partial" if model_summary["horizons"] else "failed"
                    model_summary["horizons"][horizon] = {"status": "failed", "error": str(exc)}
                    model_summary["warnings"].append(f"{horizon}: {exc}")
                    detailed.append({"model": model_name, "horizon": horizon, "sensor_scope": "all", "sensor_filter": sensor_id, "status": "failed", "error": str(exc), "traceback": traceback.format_exc(limit=8)})
            summary["models"][model_name] = model_summary
            summary["generated_at"] = datetime.now(timezone.utc).isoformat()
            self.manager.write_json("model_results.json", summary); self.manager.write_json("model_results_detailed.json", detailed)
        summary = self.manager.rebuild_summary(detailed)
        self.manager.write_json("model_results.json", summary)
        leaderboard = self.manager.build_leaderboard(detailed)
        self.manager.write_json("model_leaderboard.json", leaderboard)
        oof = self.manager.combine_predictions("oof", "stacking_oof_predictions.csv")
        test = self.manager.combine_predictions("test", "test_predictions_all_models.csv")
        self.manager.diversity(test)
        self._reports(summary, detailed, leaderboard, oof, test)
        return summary

    def _run_one(self, model_name: str, horizon: str, sensor_id: int | None, logger: logging.Logger) -> list[dict[str, Any]]:
        start = time.perf_counter(); logger.info("START model=%s horizon=%s sensor=%s", model_name, horizon, sensor_id)
        train, validation, test = self.dataset.splits(horizon)
        train = filter_sensor(train, sensor_id); validation = filter_sensor(validation, sensor_id); test = filter_sensor(test, sensor_id)
        if min(len(train), len(validation), len(test)) == 0: raise ValueError("One chronological split is empty")
        target_columns = (f"target_pm25_{horizon}", f"target_pm25_who_24h_avg_{horizon}")
        prediction_dir = self.manager.predictions / model_name; prediction_dir.mkdir(parents=True, exist_ok=True)
        oof_frames = []
        for fold_id, fold_train, fold_valid in self.dataset.oof_folds(horizon):
            fold_train = filter_sensor(fold_train, sensor_id); fold_valid = filter_sensor(fold_valid, sensor_id)
            if not len(fold_train) or not len(fold_valid): continue
            logger.info("OOF %s train=%d validation=%d", fold_id, len(fold_train), len(fold_valid))
            output = MODEL_FACTORIES[model_name]().fit_predict(fold_train, fold_valid, self.dataset.features, target_columns)
            oof_frames.append(prediction_frame(fold_valid, output, model_name, horizon, "oof", fold_id))
        oof = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
        oof_path = prediction_dir / f"oof_pm25_{horizon}.csv"; oof.to_csv(oof_path, index=False)
        final_train = pd.concat([train, validation], ignore_index=True).sort_values(["timestamp", "sensor_id"])
        extension = EXTENSIONS.get(model_name, ".joblib")
        artifact_path = self.manager.models / model_name / f"pm25_{horizon}{extension}"
        model = MODEL_FACTORIES[model_name]()
        output = model.fit_predict(final_train, test, self.dataset.features, target_columns, artifact_path)
        predictions = prediction_frame(test, output, model_name, horizon, "test", "final_test")
        prediction_path = prediction_dir / f"test_pm25_{horizon}.csv"; predictions.to_csv(prediction_path, index=False)
        metrics = evaluate(predictions.actual.to_numpy(float), predictions.predicted.to_numpy(float), predictions.actual_health_average.to_numpy(float), predictions.predicted_health_average.to_numpy(float))
        baseline_path = self.manager.predictions / "persistence" / f"test_pm25_{horizon}.csv"
        baseline_rmse = None
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)
            baseline_rmse = evaluate(baseline.actual.to_numpy(float), baseline.predicted.to_numpy(float), baseline.actual_health_average.to_numpy(float), baseline.predicted_health_average.to_numpy(float))["rmse"]
        duration = time.perf_counter() - start
        config_path = artifact_path.parent / f"pm25_{horizon}_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({"model": model_name, "horizon": horizon, "target_columns": target_columns, "features": self.dataset.features, "hyperparameters": model.hyperparameters, "chronological_split": True, "shuffle": False}, indent=2), encoding="utf-8")
        if output.history: (artifact_path.parent / f"pm25_{horizon}_history.json").write_text(json.dumps(output.history, indent=2), encoding="utf-8")
        if output.feature_importance: pd.DataFrame(output.feature_importance).to_csv(artifact_path.parent / f"pm25_{horizon}_feature_importance.csv", index=False)
        common = {
            "model": model_name, "horizon": horizon, "pollutant": "pm25", "sensor_filter": sensor_id,
            "hyperparameters": model.hyperparameters, "features_used": self.dataset.features if model_name != "prophet" else ["ds", "y"],
            "training_sample_count": len(train), "validation_sample_count": len(validation), "test_sample_count": len(test),
            "oof_prediction_count": int(oof.predicted.notna().sum()) if len(oof) else 0, "training_duration_seconds": duration,
            "best_epoch": output.best_epoch, "baseline_rmse": baseline_rmse,
            "beats_persistence": bool(baseline_rmse is not None and metrics["rmse"] is not None and metrics["rmse"] < baseline_rmse),
            "prediction_file": str(prediction_path.relative_to(self.backend_dir)), "oof_prediction_file": str(oof_path.relative_to(self.backend_dir)),
            "saved_model_path": str(artifact_path.relative_to(self.backend_dir)), "preprocessing": "Fitted on training data only",
            "warnings": output.warnings, "limitations": ["17-day dataset", "Test set is chronologically newest and was never used for selection"], "status": "completed",
        }
        records = [{**common, "sensor_scope": "all", "metrics": metrics}]
        for row in sensor_metrics(predictions): records.append({**common, "sensor_scope": f"sensor_{row['sensor_id']}", "metrics": row})
        logger.info("END model=%s horizon=%s duration=%.2fs metrics=%s", model_name, horizon, duration, metrics)
        return records

    def _logger(self, model_name: str) -> logging.Logger:
        logger = logging.getLogger(f"airaware.{model_name}"); logger.setLevel(logging.INFO); logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(self.manager.logs / f"{model_name}.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")); logger.addHandler(handler)
        return logger

    def _reports(self, summary: dict[str, Any], detailed: list[dict[str, Any]], leaderboard: dict[str, Any], oof: pd.DataFrame, test: pd.DataFrame) -> None:
        write_phase4_reports(self.manager.reports, self.manager.outputs, summary, detailed, leaderboard, oof, test)

    def _legacy_reports(self, summary: dict[str, Any], detailed: list[dict[str, Any]], leaderboard: dict[str, Any], oof: pd.DataFrame, test: pd.DataFrame) -> None:
        statuses = "\n".join(f"- {name}: {payload.get('status')}" for name, payload in summary.get("models", {}).items())
        training = f"# 11 Model Training Report\n\nGenerated: {summary.get('generated_at')}\n\n## Models\n{statuses}\n\n## Method\nChronological train/validation/test periods from Phase 3; two expanding-origin validation folds; no shuffle. Missing-value preprocessing and scaling are fitted on training data only. PM2.5 concentration and its future valid 24-hour health average are regressed separately or jointly. No ensemble or stacking model was trained.\n"
        self.manager.reports.joinpath("11_model_training_report.md").write_text(training, encoding="utf-8")
        lines = ["# 12 Model Evaluation Report", "", "## Leaderboard"]
        for horizon, rankings in leaderboard.items():
            lines += [f"### {horizon}", "", "| Rank | Model | RMSE |", "|---:|---|---:|"] + [f"| {row['rank']} | {row['model']} | {row['rmse']:.4f} |" for row in rankings.get("rmse", [])]
        lines += ["", "## Baseline", "Every advanced model is evaluated against the same persistence forecast on the newest test period.", "", "## Event evaluation", "Event predictions are derived from predicted future PM2.5 24-hour averages using the Phase 3 threshold; no classifier and no source AQI were used."]
        self.manager.reports.joinpath("12_model_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
        residual_path = self.manager.outputs / "model_residual_correlations.csv"
        residual = pd.read_csv(residual_path) if residual_path.exists() and residual_path.stat().st_size else pd.DataFrame()
        candidates = []
        for horizon in ("1h", "3h", "6h"):
            candidates.extend([row["model"] for row in leaderboard.get(horizon, {}).get("rmse", [])[:3]])
        candidates = list(dict.fromkeys(candidates))[:5]
        readiness = f"# 13 Ensemble Readiness Report\n\n## Accuracy candidates\n{', '.join(candidates) if candidates else 'No completed candidates yet.'}\n\n## Diversity\nPrediction and residual correlations are saved separately. Prefer accurate models with lower residual correlation, not merely duplicate predictions.\n\n## Stacking readiness\nTime-safe OOF rows: {len(oof):,}. Test archive rows: {len(test):,}. These archives are inputs only; no weighted ensemble, stacking meta-model, or combined prediction was trained.\n\n## Limitations\nThe dataset spans 17 days. Neural/TFT stability must be judged conservatively, and incomplete prediction intersections must remain explicit.\n"
        self.manager.reports.joinpath("13_ensemble_readiness_report.md").write_text(readiness, encoding="utf-8")
