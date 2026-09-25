import numpy as np
import pandas as pd

from app.ml.alert_engine import AlertEngine, empty_alert_state
from app.ml.sensor_health import SensorHealthEngine


NOW = pd.Timestamp("2026-01-01T00:00:00Z")


def forecast(level="Normal", who="Meets WHO Guideline", horizon="3h"):
    return {horizon: {
        "pm25": {"forecast": 23.717, "safety_upper_estimate": 31.538, "safety_margin": 7.821},
        "future_24h_average": {"forecast": 17.59},
        "high_pm_risk": {"level": level},
        "who": {"status": who, "guideline": 15.0, "ratio_to_guideline": 1.17},
    }}


def health(status="Healthy", issues=None):
    return {"status": status, "issues": issues or [], "last_reading_at": NOW.isoformat(), "minutes_since_last_reading": 0}


def evaluate(forecasts=None, sensor_health=None, state=None, timestamp=NOW, sensor_id=1):
    return AlertEngine().evaluate(sensor_id, timestamp, forecasts, sensor_health, state)


def test_no_condition_no_alert():
    result = evaluate(forecast(), health())
    assert result["active_alerts"] == []


def test_elevated_high_pm_is_warning():
    alert = evaluate(forecast("Elevated"))["active_alerts"][0]
    assert alert["type"] == "high_pm_risk"
    assert alert["severity"] == "warning"


def test_high_high_pm_is_prioritized_by_horizon():
    one = evaluate(forecast("High", horizon="1h"))["active_alerts"][0]
    six = evaluate(forecast("High", horizon="6h"))["active_alerts"][0]
    assert (one["severity"], one["priority"]) == ("critical", "urgent")
    assert (six["severity"], six["priority"]) == ("warning", "high")


def test_who_above_guideline_is_warning_and_uses_24h_average():
    alerts = evaluate(forecast(who="Above WHO Guideline"))["active_alerts"]
    who = next(alert for alert in alerts if alert["type"] == "who_exceedance")
    assert who["severity"] == "warning"
    assert who["evidence"]["predicted_24h_average"] == 17.59
    assert "24h" in who["title"]


def test_who_below_resolves_existing_alert():
    first = evaluate(forecast(who="Above WHO Guideline"))
    second = evaluate(forecast(who="Meets WHO Guideline"), state=first["state"], timestamp=NOW + pd.Timedelta(minutes=15))
    assert not any(alert["type"] == "who_exceedance" for alert in second["active_alerts"])
    assert any(alert["type"] == "who_exceedance" for alert in second["resolved_alerts"])


def test_sensor_offline_is_critical():
    alert = evaluate(sensor_health={**health("Offline"), "minutes_since_last_reading": 200})["active_alerts"][0]
    assert alert["type"] == "sensor_offline"
    assert alert["severity"] == "critical"


def test_repeated_offline_is_one_alert():
    first = evaluate(sensor_health=health("Offline"))
    second = evaluate(sensor_health=health("Offline"), state=first["state"], timestamp=NOW + pd.Timedelta(minutes=15))
    assert len(second["state"]["alerts"]) == 1
    assert second["state"]["deduplicated_repeats_suppressed"] == 1


def test_sensor_return_resolves_offline():
    first = evaluate(sensor_health=health("Offline"))
    second = evaluate(sensor_health=health("Healthy"), state=first["state"], timestamp=NOW + pd.Timedelta(minutes=15))
    assert second["active_alerts"] == []
    assert second["resolved_alerts"][0]["resolved_at"] is not None


def test_possible_drift_is_warning():
    issue = {"type": "possible_drift", "severity": "warning", "channel": "pm25", "message": "Possible Drift detected.", "evidence": {"median_shift": 12}}
    alert = evaluate(sensor_health=health(issues=[issue]))["active_alerts"][0]
    assert alert["severity"] == "warning"
    assert alert["title"] == "Possible Sensor Drift"


def test_suspicious_reading_inherits_critical_severity():
    issue = {"type": "suspicious_value", "severity": "critical", "channel": "o3", "message": "Negative value.", "evidence": {"value": -2.3}}
    alert = evaluate(sensor_health=health(issues=[issue]))["active_alerts"][0]
    assert alert["type"] == "suspicious_reading"
    assert alert["severity"] == "critical"


def test_same_condition_inside_cooldown_does_not_notify_again():
    first = evaluate(forecast("Elevated"))
    second = evaluate(forecast("Elevated"), state=first["state"], timestamp=NOW + pd.Timedelta(minutes=15))
    assert second["active_alerts"][0]["notification_recommended"] is False


def test_severity_escalation_bypasses_cooldown():
    first = evaluate(forecast("Elevated", horizon="1h"))
    second = evaluate(forecast("High", horizon="1h"), state=first["state"], timestamp=NOW + pd.Timedelta(minutes=15))
    assert second["active_alerts"][0]["notification_recommended"] is True
    assert second["active_alerts"][0]["lifecycle_transition"] == "escalated"
    assert second["state"]["escalations"] == 1


def test_alert_resolves_when_condition_clears():
    first = evaluate(forecast("Elevated"))
    second = evaluate(forecast("Normal"), state=first["state"], timestamp=NOW + pd.Timedelta(minutes=15))
    assert second["resolved_alerts"][0]["status"] == "resolved"


def test_condition_return_after_resolution_creates_new_event():
    first = evaluate(forecast("Elevated"))
    second = evaluate(forecast("Normal"), state=first["state"], timestamp=NOW + pd.Timedelta(minutes=15))
    third = evaluate(forecast("Elevated"), state=second["state"], timestamp=NOW + pd.Timedelta(minutes=30))
    alerts = [alert for alert in third["state"]["alerts"] if alert["type"] == "high_pm_risk"]
    assert len(alerts) == 2
    assert alerts[0]["id"] != alerts[1]["id"]


def test_forecast_dedup_is_horizon_aware():
    forecasts = forecast("High", horizon="1h") | forecast("High", horizon="6h")
    alerts = evaluate(forecasts)["active_alerts"]
    assert len(alerts) == 2
    assert len({alert["dedup_key"] for alert in alerts}) == 2


def test_ids_and_keys_are_deterministic():
    one = evaluate(forecast("Elevated"))["active_alerts"][0]
    two = evaluate(forecast("Elevated"))["active_alerts"][0]
    assert one["id"] == two["id"]
    assert one["dedup_key"] == "1:high_pm_risk:3h:pm25"


def test_unsupported_nablus_gases_create_no_alert():
    timestamps = pd.date_range(NOW, periods=20, freq="15min")
    data = pd.DataFrame({"sensor_id": 2, "timestamp": timestamps, "pm25": 12.0, "temperature": 20.0, "humidity": 50.0, "no2": np.nan, "o3": np.nan})
    sensor_health = SensorHealthEngine().evaluate(data, timestamps[-1])
    result = evaluate(sensor_health=sensor_health, sensor_id=2, timestamp=timestamps[-1])
    assert result["active_alerts"] == []


def test_history_after_timestamp_cannot_change_prior_alerts():
    first = evaluate(forecast("Elevated"), timestamp=NOW)
    later = evaluate(forecast("High"), state=first["state"], timestamp=NOW + pd.Timedelta(hours=1))
    prior_again = evaluate(forecast("Elevated"), timestamp=NOW)
    assert prior_again["active_alerts"] == first["active_alerts"]
    assert later["active_alerts"] != first["active_alerts"]
