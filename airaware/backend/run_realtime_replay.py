from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.evaluation import evaluate
from app.ml.realtime_features import MultiSensorHistoryBuffer, generate_realtime_features, history_requirement
from app.ml.realtime_inference import SavedModelPredictor, artifact_size_mb


BACKEND = Path(__file__).resolve().parent
OUTPUTS = BACKEND / "outputs"
MODELS = ("persistence", "random_forest", "xgboost", "lightgbm", "prophet", "lstm", "gru")
HORIZONS = ("1h", "3h", "6h")
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used")


def main() -> None:
    cleaned = pd.read_csv(BACKEND / "data/cleaned/AirAware_Tulkarem_Nablus_cleaned.csv", low_memory=False)
    cleaned["timestamp"] = pd.to_datetime(cleaned.timestamp, utc=True)
    targets = pd.read_csv(OUTPUTS / "AirAware_Forecast_Targets.csv", low_memory=False)
    targets["timestamp"] = pd.to_datetime(targets.timestamp, utc=True)
    target_lookup = targets.set_index(["sensor_id", "timestamp"])
    test_start = pd.Timestamp("2026-08-20T00:00:00Z"); warm_start = test_start - pd.Timedelta(hours=30)
    replay_source = cleaned[(cleaned.timestamp >= warm_start) & (cleaned.timestamp < pd.Timestamp("2026-08-24T00:00:00Z"))].sort_values(["timestamp", "sensor_id"])
    reference = pd.read_csv(OUTPUTS / "AirAware_ML_Features.csv", low_memory=False); reference["timestamp"] = pd.to_datetime(reference.timestamp, utc=True)
    feature_names = json.loads((BACKEND / "models/random_forest/pm25_1h_config.json").read_text(encoding="utf-8"))["features"]
    predictors = {(model, horizon): SavedModelPredictor(BACKEND, model, horizon) for model in MODELS for horizon in HORIZONS}
    buffer = MultiSensorHistoryBuffer(max_hours=30); rows = []; feature_checks = []
    for reading in replay_source.to_dict("records"):
        history, reset = buffer.update(reading); sensor = int(reading["sensor_id"]); timestamp = pd.to_datetime(reading["timestamp"], utc=True)
        if reset:
            for predictor in predictors.values(): predictor.reset_sensor(sensor)
        generated = generate_realtime_features(history, feature_names).tail(1)
        generated["continuous_segment_id"] = reading.get("continuous_segment_id", generated.continuous_segment_id.iloc[0])
        if timestamp < test_start:
            # Prime recurrent state from the historical buffer exactly as a live
            # service would before its first evaluated forecast. Predictions are
            # discarded and no fitting or target access occurs here.
            for model in ("lstm", "gru"):
                for horizon in HORIZONS:
                    predictors[(model, horizon)].predict(generated)
            continue
        reference_row = reference[(reference.sensor_id == sensor) & (reference.timestamp == timestamp)]
        if len(reference_row):
            comparable = [name for name in feature_names if name in generated and name in reference_row and pd.api.types.is_numeric_dtype(reference_row[name])]
            differences = []
            for name in comparable:
                left, right = generated[name].iloc[0], reference_row[name].iloc[0]
                if pd.notna(left) and pd.notna(right): differences.append(abs(float(left) - float(right)))
            feature_checks.append(max(differences) if differences else 0.0)
        key = (sensor, timestamp)
        if key not in target_lookup.index: continue
        actual_row = target_lookup.loc[key]
        for model in MODELS:
            for horizon in HORIZONS:
                if not bool(actual_row[f"forecast_sample_valid_{horizon}"]): continue
                requirement = history_requirement(model)
                if buffer.reading_count(sensor) < requirement["minimum_samples"]:
                    predicted, predicted_health, latency = np.nan, np.nan, 0.0
                else:
                    predicted, predicted_health, latency = predictors[(model, horizon)].predict(generated)
                rows.append({"timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "sensor_id": sensor, "city": reading["city_name"], "pollutant": "pm25", "horizon": horizon, "model": model, "current_value": generated.pm25.iloc[0], "actual": actual_row[f"target_pm25_{horizon}"], "predicted": predicted, "actual_health_average": actual_row[f"target_pm25_who_24h_avg_{horizon}"], "predicted_health_average": predicted_health, "pollution_event_actual": actual_row[f"pollution_event_{horizon}"], "pollution_event_predicted": bool(predicted_health > 15) if np.isfinite(predicted_health) else pd.NA, "sensor_data_quality": generated.overall_quality_flag.iloc[0], "prediction_available": bool(np.isfinite(predicted)), "inference_ms": latency, "model_version": "Phase4A-1.0"})
    predictions = pd.DataFrame(rows); predictions.to_csv(OUTPUTS / "realtime_replay_predictions.csv", index=False)
    results = {"project": "AirAware", "phase": "4A", "generated_at": datetime.now(timezone.utc).isoformat(), "mode": "chronological reading-by-reading replay; no retraining", "feature_consistency_max_absolute_difference": max(feature_checks) if feature_checks else None, "models": {}}
    for model in MODELS:
        model_rows = predictions[predictions.model == model]
        latencies = model_rows.loc[model_rows.prediction_available, "inference_ms"].to_numpy(float)
        horizon_metrics = {}
        for horizon, group in model_rows[model_rows.prediction_available].groupby("horizon"):
            horizon_metrics[horizon] = evaluate(group.actual.to_numpy(float), group.predicted.to_numpy(float), group.actual_health_average.to_numpy(float), group.predicted_health_average.to_numpy(float))
        requirement = history_requirement(model)
        config_path = BACKEND / "models" / model / "pm25_1h_config.json"
        required_features = json.loads(config_path.read_text(encoding="utf-8")).get("features", []) if config_path.exists() else ["pm25"]
        if model == "prophet": required_features = ["timestamp", "sensor_id"]
        results["models"][model] = {**requirement, "required_features": required_features, "average_inference_ms": float(np.mean(latencies)) if len(latencies) else None, "p95_inference_ms": float(np.percentile(latencies, 95)) if len(latencies) else None, "model_size_mb": artifact_size_mb(BACKEND / "models" / model), "real_time_ready": bool(len(latencies)), "cpu_ready": True, "gpu_required": False, "reload_smoke_test": "PASS" if len(latencies) else "FAIL", "horizons": horizon_metrics, "prediction_count": int(len(latencies)), "missing_reading_behavior": "Gap >90 minutes resets the sensor-specific buffer; predictions remain unavailable until minimum history returns."}
    tft_config = json.loads((BACKEND / "models/tft/pm25_1h_config.json").read_text(encoding="utf-8"))
    results["models"]["tft"] = {**history_requirement("tft"), "required_features": tft_config.get("features", []), "real_time_ready": False, "cpu_ready": True, "gpu_required": False, "reload_smoke_test": "NOT_RUN", "warning": "TFT checkpoints are saved, but an incremental per-reading adapter has not passed replay; deployment adapter remains required.", "model_size_mb": artifact_size_mb(BACKEND / "models/tft")}
    (OUTPUTS / "realtime_replay_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    enrich_detailed(results)
    write_report(results)


def enrich_detailed(replay: dict) -> None:
    path = OUTPUTS / "model_results_detailed.json"; detailed = json.loads(path.read_text(encoding="utf-8"))
    for row in detailed:
        operational = replay["models"].get(row.get("model"), {})
        row.update({key: operational.get(key) for key in ("average_inference_ms", "p95_inference_ms", "model_size_mb", "minimum_history_required", "sampling_interval", "real_time_ready", "cpu_ready", "gpu_required", "reload_smoke_test")})
    path.write_text(json.dumps(detailed, indent=2, allow_nan=False), encoding="utf-8")


def write_report(results: dict) -> None:
    lines = ["# 14 Real-Time Model Readiness Report", "", "Historical test rows were replayed chronologically, one reading at a time, with an independent gap-aware buffer per sensor. No model retrained during inference.", "", "| Model | Accuracy (1h RMSE) | Avg inference ms | P95 ms | Model size MB | Minimum history | CPU ready | GPU required | Real-time ready |", "|---|---:|---:|---:|---:|---|---|---|---|"]
    for model, data in results["models"].items():
        rmse = data.get("horizons", {}).get("1h", {}).get("rmse")
        lines.append(f"| {model} | {rmse if rmse is not None else 'N/A'} | {data.get('average_inference_ms', 'N/A')} | {data.get('p95_inference_ms', 'N/A')} | {data.get('model_size_mb', 'N/A')} | {data.get('minimum_history_required')} | {data.get('cpu_ready')} | {data.get('gpu_required')} | {data.get('real_time_ready')} |")
    lines += ["", "## Operational interpretation", "Tree models and persistence are the easiest CPU deployment candidates. Prophet is reloadable but is sensor-specific and does not share the multivariate feature set. LSTM/GRU require sequence state and approximately nine hours of warm history. TFT is computationally heavier and is not marked deployment-ready until a dedicated incremental adapter passes the same replay.", "", "## Missing readings", "A delayed interval remains missing. A gap above 90 minutes resets continuity; the service must return prediction_available=false until the model-specific history requirement is restored.", "", "No final model is selected here. Accuracy and later ensemble/stacking evidence must be reviewed together."]
    (BACKEND / "reports/14_realtime_model_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()
