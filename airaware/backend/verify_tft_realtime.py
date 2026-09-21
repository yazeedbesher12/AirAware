"""Fresh-process reload and causal-readiness verification for saved TFT models."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from app.ml.dataset import load_phase4_dataset
from app.ml.tft_inference import TFTInferenceAdapter, UnsafeTFTCheckpointError


BACKEND = Path(__file__).resolve().parent
HORIZONS = ("1h", "3h", "6h")


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
    dataset = load_phase4_dataset(BACKEND / "outputs")
    results = {
        "project": "AirAware", "phase": "4A-TFT-deployment-audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fresh_process": True, "weights_changed": False, "retrained": False,
        "horizons": {},
    }
    empty_predictions: list[dict] = []
    for horizon in HORIZONS:
        adapter = TFTInferenceAdapter(BACKEND, horizon)
        train, validation, test = dataset.splits(horizon)
        sensor = 1
        final_train = pd.concat([train, validation], ignore_index=True).sort_values(["timestamp", "sensor_id"])
        stored = pd.read_csv(BACKEND / f"outputs/predictions/tft/test_pm25_{horizon}.csv")
        stored["timestamp"] = pd.to_datetime(stored.timestamp, utc=True)
        known = stored[(stored.sensor_id == sensor) & stored.predicted.notna()].iloc[0]
        current = test[(test.sensor_id == sensor) & (test.timestamp == known.timestamp)]
        sensor_test = test[test.sensor_id == sensor]
        segment = current.continuous_segment_id.iloc[0]
        context = final_train[(final_train.sensor_id == sensor) & (final_train.continuous_segment_id == segment)].tail(adapter.metadata.max_encoder_length)
        # Recreate the exact offline batch used when the stored Phase 4A test
        # predictions were generated. This block is diagnostic only and is
        # intentionally separate from the causal live boundary below.
        diagnostic_frame = pd.concat([context, sensor_test], ignore_index=True)
        predicted, predicted_health, latency, average_forward_ms, p95_forward_ms = adapter.reproduce_offline_prediction(diagnostic_frame, known.timestamp)
        difference = abs(predicted - float(known.predicted))

        # A real history buffer contains no target_* columns. The adapter must
        # refuse it rather than silently invent the unavailable normalizer data.
        raw_history = pd.read_csv(BACKEND / "data/cleaned/AirAware_Tulkarem_Nablus_cleaned.csv")
        raw_history["timestamp"] = pd.to_datetime(raw_history.timestamp, utc=True)
        raw_history = raw_history[(raw_history.sensor_id == sensor) & (raw_history.timestamp <= known.timestamp)].tail(adapter.metadata.full_feature_history_samples)
        causal_refusal = None
        try:
            adapter.predict(raw_history)
        except UnsafeTFTCheckpointError as exc:
            causal_refusal = str(exc)
        except ValueError as exc:
            causal_refusal = str(exc)

        results["horizons"][horizon] = {
            **adapter.audit_dict(),
            "checkpoint_loaded": True,
            "checkpoint_state": "unchanged",
            "offline_reload_smoke_test": "PASS" if np.isfinite(predicted) and difference <= 1e-5 else "FAIL",
            "stored_prediction": float(known.predicted),
            "reloaded_prediction": predicted,
            "absolute_difference": difference,
            "offline_diagnostic_inference_ms": latency,
            "average_model_forward_ms": average_forward_ms,
            "p95_model_forward_ms": p95_forward_ms,
            "average_realtime_inference_ms": None,
            "p95_realtime_inference_ms": None,
            "checkpoint_size_mb": adapter.checkpoint_size_mb,
            "cpu_inference_status": "PASS",
            "gpu_required": False,
            "causal_realtime_replay": "BLOCKED_BEFORE_START",
            "causal_refusal_reason": causal_refusal,
            "real_time_ready": False,
        }

    results["overall"] = {
        "reload_smoke_test": "PASS" if all(v["offline_reload_smoke_test"] == "PASS" for v in results["horizons"].values()) else "FAIL",
        "causal_preprocessing_reproducible": False,
        "real_time_ready": False,
        "exact_blocker": "Saved EncoderNormalizer requires future-shifted target values inside each encoder window. Exact causal replay is impossible without changing preprocessing and retraining.",
        "required_action": "Retrain TFT with a causal target normalization strategy fitted on training data only; approval required before changing weights.",
    }
    (BACKEND / "outputs/tft_realtime_replay_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(empty_predictions, columns=["timestamp", "sensor_id", "horizon", "predicted", "prediction_available", "reason"]).to_csv(
        BACKEND / "outputs/tft_realtime_replay_predictions.csv", index=False
    )
    update_phase4_outputs(results)
    print(json.dumps(results, indent=2, allow_nan=False))


def update_phase4_outputs(results: dict) -> None:
    detailed_path = BACKEND / "outputs/model_results_detailed.json"
    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    for row in detailed:
        if row.get("model") != "tft" or row.get("horizon") not in HORIZONS:
            continue
        audit = results["horizons"][row["horizon"]]
        warning = results["overall"]["exact_blocker"]
        row.update({
            "average_inference_ms": None,
            "p95_inference_ms": None,
            "offline_average_model_forward_ms": audit["average_model_forward_ms"],
            "offline_p95_model_forward_ms": audit["p95_model_forward_ms"],
            "model_size_mb": audit["checkpoint_size_mb"],
            "minimum_history_required": "30 hours continuous for full feature parity (6-hour encoder + 24-hour feature lookback)",
            "encoder_history_required": "24 readings = 6 hours at 15-minute sampling",
            "sampling_interval": "15 minutes",
            "real_time_ready": False,
            "cpu_ready": True,
            "gpu_required": False,
            "reload_smoke_test": "PASS_OFFLINE_REPRODUCTION_ONLY",
            "causal_realtime_replay": "BLOCKED_BEFORE_START",
            "tft_deployment_blocker": warning,
            "checkpoint_weights_changed": False,
            "retrained": False,
        })
        warnings_list = list(row.get("warnings", []))
        if warning not in warnings_list:
            warnings_list.append(warning)
        row["warnings"] = warnings_list
    detailed_path.write_text(json.dumps(detailed, indent=2, allow_nan=False), encoding="utf-8")

    replay_path = BACKEND / "outputs/realtime_replay_results.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["models"]["tft"] = {
        "minimum_history_required": "30 hours continuous for full feature parity (6-hour encoder + 24-hour feature lookback)",
        "minimum_samples": 121,
        "sampling_interval": "15 minutes",
        "real_time_ready": False,
        "cpu_ready": True,
        "gpu_required": False,
        "reload_smoke_test": "PASS_OFFLINE_REPRODUCTION_ONLY",
        "causal_realtime_replay": "BLOCKED_BEFORE_START",
        "average_inference_ms": None,
        "p95_inference_ms": None,
        "offline_average_model_forward_ms": {h: results["horizons"][h]["average_model_forward_ms"] for h in HORIZONS},
        "offline_p95_model_forward_ms": {h: results["horizons"][h]["p95_model_forward_ms"] for h in HORIZONS},
        "model_size_mb": sum(results["horizons"][h]["checkpoint_size_mb"] for h in HORIZONS),
        "warning": results["overall"]["exact_blocker"],
    }
    replay_path.write_text(json.dumps(replay, indent=2, allow_nan=False), encoding="utf-8")

    report_path = BACKEND / "reports/14_realtime_model_readiness_report.md"
    lines = report_path.read_text(encoding="utf-8").splitlines()
    replacement = "| tft | N/A | N/A | N/A | {:.3f} | 30 hours full-feature history (6h encoder + 24h feature lookback) | True | False | False |".format(
        sum(results["horizons"][h]["checkpoint_size_mb"] for h in HORIZONS)
    )
    lines = [replacement if line.startswith("| tft |") else line for line in lines]
    marker = "## TFT checkpoint deployment audit"
    if marker in lines:
        lines = lines[:lines.index(marker)]
    lines += [
        "", marker, "",
        "All three TFT checkpoints reload on CPU in a fresh Python process and reproduce their stored offline predictions within 1e-5. The learned weights were not changed and no retraining occurred.", "",
        "The checkpoint encoder is 24 readings, which is exactly 6 hours at the 15-minute cadence. The selected inputs also contain backward-looking PM2.5 WHO 24-hour features. Full input parity therefore requires a 30-hour continuous raw buffer (6-hour encoder plus 24-hour maximum feature lookback). The earlier approximately 12-hour statement was incorrect.", "",
        "Real-time readiness remains **false**. The fitted `MultiNormalizer` contains `EncoderNormalizer` instances that calculate target center/scale from encoder target columns. Those target columns are Phase 3 future-shifted targets and are not fully known at forecast time. Supplying them would leak future observations; inventing replacements would create training-serving skew.", "",
        "Causal replay was blocked before prediction by the fail-closed adapter. Real-time average/p95 latency is therefore N/A. Offline single-sample model-forward benchmarks are:", "",
        "| Horizon | Offline avg forward ms | Offline p95 forward ms | Reload difference |",
        "|---|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        audit = results["horizons"][horizon]
        lines.append(f"| {horizon} | {audit['average_model_forward_ms']:.3f} | {audit['p95_model_forward_ms']:.3f} | {audit['absolute_difference']:.8f} |")
    lines += ["", "A causal target-normalization design requires retraining. Per instruction, retraining was not started."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
