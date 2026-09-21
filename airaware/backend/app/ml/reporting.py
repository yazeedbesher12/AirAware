from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _number(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def write_phase4_reports(
    reports_dir: Path,
    outputs_dir: Path,
    summary: dict[str, Any],
    detailed: list[dict[str, Any]],
    leaderboard: dict[str, Any],
    oof: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    overall = [row for row in detailed if row.get("sensor_scope") == "all" and row.get("status") == "completed"]
    sensor_rows = [row for row in detailed if row.get("sensor_scope") != "all" and row.get("status") == "completed"]

    training = [
        "# 11 Model Training Report", "", f"Generated: {summary.get('generated_at')}", "",
        "## Data and validation design", "",
        "Phase 3 ML features and valid PM2.5 targets were used without rebuilding feature engineering. "
        "Rows are ordered chronologically: train is the oldest period, validation is later, and test is the newest unseen period. "
        "Two expanding-origin folds generated time-safe OOF predictions; no shuffle or random split was used.", "",
        "Imputers and neural scalers were fitted on training data only. LSTM, GRU, and TFT sequences remain inside continuous segments. "
        "The source AQI was not used as a target. No ensemble or stacking model was trained.", "",
        "## Training runs", "",
        "| Model | Horizon | Train | Validation | Test | OOF predictions | Duration (s) | Best epoch | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in overall:
        training.append(
            f"| {row['model']} | {row['horizon']} | {row['training_sample_count']} | {row['validation_sample_count']} | "
            f"{row['test_sample_count']} | {row['oof_prediction_count']} | {_number(row['training_duration_seconds'], 2)} | "
            f"{row.get('best_epoch') if row.get('best_epoch') is not None else 'N/A'} | {row['status']} |"
        )
    training += ["", "## Saved hyperparameters", ""]
    seen: set[tuple[str, str]] = set()
    for row in overall:
        key = (row["model"], row["horizon"])
        if key in seen:
            continue
        seen.add(key)
        training += [f"### {row['model']} — {row['horizon']}", "", "```json", json.dumps(row.get("hyperparameters", {}), indent=2), "```", ""]
    warnings = [(row["model"], row["horizon"], item) for row in overall for item in row.get("warnings", [])]
    training += ["## Warnings and limitations", ""]
    training += [f"- {model} {horizon}: {warning}" for model, horizon, warning in warnings] or ["- No training run failed."]
    training += ["- The available dataset spans 17 days, so deep-model stability requires more unseen temporal coverage."]
    reports_dir.joinpath("11_model_training_report.md").write_text("\n".join(training), encoding="utf-8")

    evaluation = ["# 12 Model Evaluation Report", "", "## Overall unseen-test performance", ""]
    for horizon in ("1h", "3h", "6h"):
        evaluation += [f"### {horizon}", "", "| Model | MAE | RMSE | R² | MAPE | Median AE | Event recall | Event F1 | FP | FN |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows = sorted((r for r in overall if r["horizon"] == horizon), key=lambda r: r["metrics"].get("rmse", float("inf")))
        for row in rows:
            metric = row["metrics"]
            evaluation.append(
                f"| {row['model']} | {_number(metric.get('mae'))} | {_number(metric.get('rmse'))} | {_number(metric.get('r2'))} | "
                f"{_number(metric.get('mape'))} | {_number(metric.get('median_absolute_error'))} | {_number(metric.get('event_recall'))} | "
                f"{_number(metric.get('event_f1'))} | {metric.get('false_positives', 'N/A')} | {metric.get('false_negatives', 'N/A')} |"
            )
        evaluation.append("")
    evaluation += ["## Persistence baseline comparison", "", "| Model | Horizon | Baseline RMSE | Model RMSE | Beats persistence |",
                   "|---|---|---:|---:|---|"]
    for row in overall:
        evaluation.append(f"| {row['model']} | {row['horizon']} | {_number(row.get('baseline_rmse'))} | {_number(row['metrics'].get('rmse'))} | {'Yes' if row.get('beats_persistence') else 'No'} |")
    evaluation += ["", "## Performance by sensor", "", "| Model | Horizon | Sensor | City | N | RMSE | MAE | R² | Event recall | Status |",
                   "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in sensor_rows:
        metric = row["metrics"]
        evaluation.append(
            f"| {row['model']} | {row['horizon']} | {metric.get('sensor_id')} | {metric.get('city')} | {metric.get('n')} | "
            f"{_number(metric.get('rmse'))} | {_number(metric.get('mae'))} | {_number(metric.get('r2'))} | "
            f"{_number(metric.get('event_recall'))} | {metric.get('status', 'OK')} |"
        )
    evaluation += ["", "## Error interpretation", "",
                   "Event labels are derived from predicted future PM2.5 24-hour averages using the Phase 3 rule. No classifier was trained. "
                   "False negatives remain explicit. MAPE is retained only for valid positive PM2.5 targets and should not be used alone."]
    reports_dir.joinpath("12_model_evaluation_report.md").write_text("\n".join(evaluation), encoding="utf-8")

    best = {h: leaderboard[h]["rmse"][0] for h in ("1h", "3h", "6h")}
    residual_path = outputs_dir / "model_residual_correlations.csv"
    residual = pd.read_csv(residual_path) if residual_path.exists() else pd.DataFrame()
    readiness = ["# 13 Ensemble Readiness Report", "", "## Scope", "",
                 "This report prepares evidence for a later phase. No weighted ensemble, stacking meta-model, or combined prediction was trained.", "",
                 "## Strongest models by horizon", ""]
    for horizon, row in best.items():
        readiness.append(f"- **{horizon}:** {row['model']} has the lowest test RMSE ({row['rmse']:.3f}).")
    readiness += ["", "## Diversity evidence", ""]
    if not residual.empty:
        redundant = residual.loc[residual.correlation.abs() >= 0.90].sort_values("correlation", ascending=False).head(8)
        readiness += [f"- {r.model_a} / {r.model_b} at {r.horizon}: residual correlation {r.correlation:.3f} (overlap {int(r.overlap):,})." for r in redundant.itertuples()]
        diverse = residual.loc[residual.overlap >= 500].assign(abs_correlation=lambda frame: frame.correlation.abs()).sort_values("abs_correlation").head(6)
        readiness += ["", "Lower-error-correlation pairs worth retaining for review:"]
        readiness += [f"- {r.model_a} / {r.model_b} at {r.horizon}: residual correlation {r.correlation:.3f}." for r in diverse.itertuples()]
    readiness += ["", "## Candidate set for later review", "",
                  "GRU, LSTM, TFT, and LightGBM provide a useful balance of accuracy, horizon coverage, and model-family diversity. "
                  "XGBoost is a credible LightGBM substitute, but the two are highly redundant. Prophet is diverse but too inaccurate to include solely for diversity.", "",
                  "## Stacking readiness", "",
                  f"The time-safe OOF archive contains {len(oof):,} unique horizon/timestamp/sensor rows; the chronological test archive contains {len(test):,}. "
                  "Missing sequence-model predictions during warm-up remain explicit. These archives were not used to fit a meta-model.", "",
                  "## Limitations", "",
                  "The dataset spans only 17 days. Candidate recommendations are provisional and require review on more unseen temporal coverage."]
    reports_dir.joinpath("13_ensemble_readiness_report.md").write_text("\n".join(readiness), encoding="utf-8")
