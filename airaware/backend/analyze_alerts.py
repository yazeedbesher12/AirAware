"""Replay AlertEngine from existing forecast artifacts and sensor-health history."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import OUTPUT_DIR, REPORT_DIR, PROJECT_DIR
from app.ml.alert_engine import AlertEngine, empty_alert_state
from app.ml.forecast_interpreter import interpret_who_pm25
from app.ml.forecast_safety import ForecastSafety
from app.ml.high_pm_risk import HighPMRisk


ALERT_OUTPUT_DIR = OUTPUT_DIR / "alerts"
EVENTS_PATH = ALERT_OUTPUT_DIR / "alert_events.csv"
CURRENT_PATH = ALERT_OUTPUT_DIR / "current_alerts.json"
SUMMARY_PATH = ALERT_OUTPUT_DIR / "alert_summary.json"
REPORT_PATH = REPORT_DIR / "alerts_report.md"
PREDICTIONS_PATH = PROJECT_DIR / "test_predictions_all_models.csv"
HEALTH_DIR = OUTPUT_DIR / "sensor_health"


def _json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def load_forecasts() -> dict[tuple[int, pd.Timestamp], dict[str, dict[str, Any]]]:
    """Normalize existing LightGBM test predictions; no model inference occurs."""
    source = pd.read_csv(PREDICTIONS_PATH)
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
    source = source.loc[
        source.target.eq("pm25")
        & source.prediction_lightgbm.notna()
        & source.health_prediction_lightgbm.notna()
    ].copy()
    safety_layer = ForecastSafety()
    risk_layer = HighPMRisk()
    normalized: dict[tuple[int, pd.Timestamp], dict[str, dict[str, Any]]] = {}
    for row in source.sort_values(["timestamp", "sensor_id", "horizon"]).itertuples(index=False):
        horizon = str(row.horizon)
        point = float(row.prediction_lightgbm)
        future_average = float(row.health_prediction_lightgbm)
        safety = safety_layer.apply(horizon, point)
        risk = risk_layer.assess(horizon, point, safety["safety_upper_estimate"])
        who = interpret_who_pm25(future_average)
        normalized.setdefault((int(row.sensor_id), row.timestamp), {})[horizon] = {
            "pm25": {"forecast": point, "safety_upper_estimate": safety["safety_upper_estimate"], "safety_margin": safety["safety_margin"]},
            "future_24h_average": {"forecast": future_average},
            "high_pm_risk": {"level": risk["level"]},
            "who": who,
        }
    return normalized


def load_health() -> dict[tuple[int, pd.Timestamp], dict[str, Any]]:
    normalized: dict[tuple[int, pd.Timestamp], dict[str, Any]] = {}
    for path in sorted(HEALTH_DIR.glob("history_sensor_*.csv")):
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        for row in frame.itertuples(index=False):
            timestamp = row.timestamp
            last_reading = timestamp - pd.Timedelta(minutes=float(row.minutes_since_last_reading))
            normalized[(int(row.sensor_id), timestamp)] = {
                "status": row.status,
                "issues": json.loads(row.issues_json),
                "last_reading_at": last_reading.isoformat(),
                "minutes_since_last_reading": float(row.minutes_since_last_reading),
            }
    return normalized


def replay() -> dict[str, Any]:
    forecasts = load_forecasts()
    health = load_health()
    keys = sorted(set(forecasts) | set(health), key=lambda value: (value[1], value[0]))
    engine = AlertEngine()
    state = empty_alert_state()
    transition_count = 0
    for sensor_id, timestamp in keys:
        result = engine.evaluate(
            sensor_id=sensor_id,
            timestamp=timestamp,
            forecasts=forecasts.get((sensor_id, timestamp)),
            sensor_health=health.get((sensor_id, timestamp)),
            previous_alert_state=state,
        )
        state = result["state"]
        transition_count += len(result["transitions"])
    return {"state": state, "evaluations": len(keys), "lifecycle_transitions": transition_count, "forecast_observations": len(forecasts), "health_observations": len(health)}


def write_outputs(result: dict[str, Any]) -> dict[str, Any]:
    ALERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alerts = result["state"]["alerts"]
    columns = [
        "alert_id", "sensor_id", "timestamp", "type", "category", "severity", "priority", "status",
        "horizon", "channel", "dedup_key", "title", "message", "created_at", "updated_at", "resolved_at",
        "notification_recommended", "alert_recommended", "source", "evidence",
    ]
    rows = [{
        "alert_id": alert["id"], "sensor_id": alert["sensor_id"], "timestamp": alert["created_at"],
        "type": alert["type"], "category": alert["category"], "severity": alert["severity"],
        "priority": alert["priority"], "status": alert["status"], "horizon": alert.get("horizon"),
        "channel": alert.get("channel"), "dedup_key": alert["dedup_key"], "title": alert["title"],
        "message": alert["message"], "created_at": alert["created_at"], "updated_at": alert["updated_at"],
        "resolved_at": alert["resolved_at"], "notification_recommended": alert["notification_recommended"],
        "alert_recommended": alert["alert_recommended"], "source": alert["source"],
        "evidence": json.dumps(alert["evidence"], separators=(",", ":"), default=_json_value),
    } for alert in alerts]
    pd.DataFrame(rows, columns=columns).to_csv(EVENTS_PATH, index=False)
    active = [alert for alert in alerts if alert["status"] == "active"]
    CURRENT_PATH.write_text(json.dumps({"generated_from": "historical replay; not live", "active_alerts": active}, indent=2, default=_json_value), encoding="utf-8")
    summary = {
        "data_scope": {
            "forecast_source": str(PREDICTIONS_PATH),
            "forecast_note": "Existing LightGBM TEST predictions only; gaps or unavailable horizons were not fabricated.",
            "forecast_observations": result["forecast_observations"],
            "sensor_health_observations": result["health_observations"],
            "evaluations": result["evaluations"],
        },
        "total_events": len(alerts),
        "active_alerts": len(active),
        "resolved_alerts": len(alerts) - len(active),
        "counts_by_severity": dict(Counter(alert["severity"] for alert in alerts)),
        "counts_by_type": dict(Counter(alert["type"] for alert in alerts)),
        "counts_by_sensor": {str(key): value for key, value in Counter(alert["sensor_id"] for alert in alerts).items()},
        "deduplicated_repeats_suppressed": result["state"]["deduplicated_repeats_suppressed"],
        "escalations": result["state"]["escalations"],
        "lifecycle_transitions": result["lifecycle_transitions"],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# AirAware Alerts & Early Warning Replay", "",
        "> Historical development replay only. Current frontend alerts remain clearly labeled demo data until a real API is integrated.", "",
        "## Architecture", "",
        "Normalized ForecastEngine/SensorHealthEngine-shaped objects -> AlertEngine rules -> deterministic deduplication -> lifecycle state -> active/resolved outputs.", "",
        "## Results", "",
        f"- Total lifecycle events: {summary['total_events']}",
        f"- Active at final available evaluation: {summary['active_alerts']}",
        f"- Resolved: {summary['resolved_alerts']}",
        f"- Deduplicated update notifications suppressed: {summary['deduplicated_repeats_suppressed']}",
        f"- Escalations: {summary['escalations']}",
        f"- Counts by severity: {summary['counts_by_severity']}",
        f"- Counts by type: {summary['counts_by_type']}",
        f"- Counts by sensor: {summary['counts_by_sensor']}", "",
        "## Limitations", "",
        "Forecast replay uses only existing LightGBM TEST point and health-average predictions. Existing AirAware safety, High-PM risk, and WHO utilities normalize those artifacts. Missing forecast timestamps or horizons are not synthesized. Sensor-health replay uses the generated causal histories.", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    result = replay()
    summary = write_outputs(result)
    write_report(summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
