from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_json(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp): return value.isoformat()
    return value


class ResultsManager:
    def __init__(self, backend_dir: Path):
        self.backend_dir = backend_dir
        self.outputs = backend_dir / "outputs"
        self.models = backend_dir / "models"
        self.logs = backend_dir / "logs"
        self.predictions = self.outputs / "predictions"
        self.reports = backend_dir / "reports"
        for path in (self.outputs, self.models, self.logs, self.predictions, self.reports): path.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> None:
        (self.outputs / name).write_text(json.dumps(safe_json(payload), indent=2, allow_nan=False), encoding="utf-8")

    def load_json(self, name: str, default: Any) -> Any:
        path = self.outputs / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def build_leaderboard(self, detailed: list[dict[str, Any]]) -> dict[str, Any]:
        leaderboard: dict[str, Any] = {}
        for horizon in ("1h", "3h", "6h"):
            rows = [row for row in detailed if row.get("horizon") == horizon and row.get("sensor_scope") == "all" and row.get("status") == "completed"]
            rankings = {}
            for metric, reverse in (("rmse", False), ("mae", False), ("r2", True), ("event_recall", True), ("event_f1", True)):
                valid = [row for row in rows if row.get("metrics", {}).get(metric) is not None]
                valid.sort(key=lambda row: row["metrics"][metric], reverse=reverse)
                rankings[metric] = [{"rank": index + 1, "model": row["model"], metric: row["metrics"][metric]} for index, row in enumerate(valid)]
            leaderboard[horizon] = rankings
        return leaderboard

    def rebuild_summary(self, detailed: list[dict[str, Any]]) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for row in detailed:
            if row.get("sensor_scope") != "all" or row.get("sensor_filter") is not None: continue
            model = row.get("model"); horizon = row.get("horizon")
            payload = models.setdefault(model, {"status": "failed", "horizons": {}, "warnings": []})
            if row.get("status") == "completed":
                payload["horizons"][horizon] = row.get("metrics", {})
            else:
                payload["horizons"][horizon] = {"status": "failed", "error": row.get("error")}
                payload["warnings"].append(f"{horizon}: {row.get('error')}")
        for payload in models.values():
            completed = sum(1 for value in payload["horizons"].values() if value.get("rmse") is not None)
            payload["status"] = "completed" if completed == 3 else "partial" if completed else "failed"
        return {"project": "AirAware", "phase": "4A", "generated_at": datetime.now(timezone.utc).isoformat(), "models": models}

    def combine_predictions(self, split: str, filename: str) -> pd.DataFrame:
        files = list(self.predictions.glob(f"*/{split}_*_*.csv"))
        frames = []
        grouped_files: dict[str, list[Path]] = {}
        for file in files: grouped_files.setdefault(file.parent.name, []).append(file)
        for model, model_files in grouped_files.items():
            parts = [pd.read_csv(file) for file in sorted(model_files)]
            frame = pd.concat([part for part in parts if not part.empty], ignore_index=True) if any(not part.empty for part in parts) else pd.DataFrame()
            if frame.empty: continue
            keep = ["timestamp", "sensor_id", "city", "target", "horizon", "actual", "actual_health_average", "pollution_event_actual", "fold_id"]
            frame = frame[keep + ["predicted", "predicted_health_average", "residual"]].rename(columns={"predicted": f"prediction_{model}", "predicted_health_average": f"health_prediction_{model}", "residual": f"residual_{model}"})
            frames.append(frame)
        if not frames:
            result = pd.DataFrame(); result.to_csv(self.outputs / filename, index=False); return result
        keys = ["timestamp", "sensor_id", "city", "target", "horizon", "actual", "actual_health_average", "pollution_event_actual", "fold_id"]
        result = frames[0]
        for frame in frames[1:]: result = result.merge(frame, on=keys, how="outer")
        result.sort_values(["horizon", "timestamp", "sensor_id"]).to_csv(self.outputs / filename, index=False)
        return result

    def diversity(self, test_archive: pd.DataFrame) -> None:
        prediction_columns = [column for column in test_archive if column.startswith("prediction_")]
        residual_columns = [column for column in test_archive if column.startswith("residual_")]
        prediction_rows = []; residual_rows = []; overlap_rows = []
        for horizon, group in test_archive.groupby("horizon") if len(test_archive) else []:
            for index, left in enumerate(prediction_columns):
                for right in prediction_columns[index + 1:]:
                    pair = group[[left, right]].dropna()
                    prediction_rows.append({"horizon": horizon, "model_a": left.removeprefix("prediction_"), "model_b": right.removeprefix("prediction_"), "correlation": pair[left].corr(pair[right]) if len(pair) > 2 else np.nan, "overlap": len(pair)})
            for index, left in enumerate(residual_columns):
                for right in residual_columns[index + 1:]:
                    pair = group[[left, right]].dropna()
                    residual_rows.append({"horizon": horizon, "model_a": left.removeprefix("residual_"), "model_b": right.removeprefix("residual_"), "correlation": pair[left].corr(pair[right]) if len(pair) > 2 else np.nan, "overlap": len(pair)})
                    if len(pair):
                        a = pair[left].abs(); b = pair[right].abs(); qa = a.quantile(.75); qb = b.quantile(.75)
                        overlap_rows.append({"horizon": horizon, "model_a": left.removeprefix("residual_"), "model_b": right.removeprefix("residual_"), "both_large_error_count": int(((a >= qa) & (b >= qb)).sum()), "overlap": len(pair)})
        pd.DataFrame(prediction_rows).to_csv(self.outputs / "model_prediction_correlations.csv", index=False)
        pd.DataFrame(residual_rows).to_csv(self.outputs / "model_residual_correlations.csv", index=False)
        pd.DataFrame(overlap_rows).to_csv(self.outputs / "model_error_overlap.csv", index=False)

    def training_metadata(self, dataset_path: Path, selected_models: list[str]) -> dict[str, Any]:
        packages = {}
        for name in ("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "prophet", "torch", "lightning", "pytorch-forecasting"):
            try: packages[name] = version(name)
            except PackageNotFoundError: packages[name] = None
        try:
            import torch
            device = {"cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
        except Exception:
            device = {"cuda_available": False, "device": "CPU"}
        return {
            "project": "AirAware", "phase": "4A", "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(dataset_path.relative_to(self.backend_dir)), "feature_dataset_version": "Phase 3 / 3.0.0",
            "who_config_version": "WHO 2021 / 2021.1", "train_period": "2026-08-06 through 2026-08-15",
            "validation_period": "2026-08-16 through 2026-08-19", "test_period": "2026-08-20 through 2026-08-23",
            "split_strategy": "Chronological; two expanding-origin validation folds; no shuffle",
            "random_seeds": {"global": 42}, "selected_models": selected_models, "python": sys.version,
            "platform": platform.platform(), "processor": platform.processor(), "device": device, "library_versions": packages,
        }
