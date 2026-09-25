import numpy as np
import pandas as pd

from app.ml.sensor_health import SensorHealthEngine


def stream(sensor_id=1, values=None, periods=40, start="2026-01-01"):
    timestamps = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    pm25 = np.linspace(10, 20, periods) if values is None else np.asarray(values, dtype=float)
    return pd.DataFrame({
        "sensor_id": sensor_id,
        "timestamp": timestamps,
        "pm25": pm25,
        "temperature": 20 + np.sin(np.arange(periods) / 5),
        "humidity": 50 + np.cos(np.arange(periods) / 5),
        "no2": 12 + np.sin(np.arange(periods) / 4) if sensor_id == 1 else np.nan,
        "o3": 30 + np.cos(np.arange(periods) / 4) if sensor_id == 1 else np.nan,
    })


def types(result):
    return {issue["type"] for issue in result["issues"]}


def test_healthy_normal_stream():
    data = stream()
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert result["status"] == "Healthy"
    assert result["health_score"] == 100


def test_missing_supported_pm25_reading():
    data = stream(sensor_id=2)
    data.loc[data.index[-1], "pm25"] = np.nan
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert "missing_reading" in types(result)
    assert next(i for i in result["issues"] if i["type"] == "missing_reading")["channel"] == "pm25"


def test_unsupported_no2_on_nablus_does_not_cause_failure():
    data = stream(sensor_id=2)
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert not any(issue["channel"] in {"no2", "o3"} for issue in result["issues"])
    assert result["status"] == "Healthy"


def test_sampling_gap_on_resumed_arrival():
    data = stream(periods=10)
    data.loc[data.index[-1], "timestamp"] += pd.Timedelta(minutes=30)
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert "sampling_gap" in types(result)


def test_offline_condition():
    data = stream(periods=10)
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1] + pd.Timedelta(minutes=180))
    assert result["status"] == "Offline"
    assert result["health_score"] == 0


def test_flatline():
    values = np.r_[np.arange(20.0), np.repeat(19.0, 8)]
    data = stream(values=values, periods=len(values))
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert "flatline" in types(result)


def test_sudden_jump_uses_prior_reference():
    values = np.r_[10 + np.arange(30) * .2, 80]
    data = stream(values=values, periods=len(values))
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert "jump" in types(result)
    jump = next(i for i in result["issues"] if i["type"] == "jump")
    assert jump["evidence"]["reference_differences"] == 29


def test_suspicious_impossible_value():
    data = stream()
    data.loc[data.index[-1], "humidity"] = 120
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert "suspicious_value" in types(result)
    assert result["status"] == "Critical"


def test_possible_drift_case():
    values = np.r_[10 + np.sin(np.arange(192)) * .2, 30 + np.sin(np.arange(96)) * .2]
    data = stream(values=values, periods=len(values))
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert "possible_drift" in types(result)


def test_conservative_abnormal_noise():
    baseline = 10 + np.arange(57) * .1
    noisy = baseline[-1] + np.array([10, -10, 11, -11, 12, -12, 13, -13])
    values = np.r_[baseline, noisy]
    data = stream(values=values, periods=len(values))
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert "abnormal_noise" in types(result)


def test_multiple_simultaneous_issues():
    values = np.r_[np.arange(30.0), 100]
    data = stream(values=values, periods=len(values))
    data.loc[data.index[-1], "temperature"] = np.nan
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert {"missing_reading", "jump"}.issubset(types(result))


def test_score_is_bounded():
    data = stream()
    final = data.index[-1]
    data.loc[final, ["pm25", "temperature", "humidity", "no2", "o3"]] = [np.nan, np.inf, 200, -1, -1]
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert 0 <= result["health_score"] <= 100


def test_critical_condition_overrides_score_range():
    data = stream()
    data.loc[data.index[-1], "humidity"] = -1
    result = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert result["health_score"] >= 65
    assert result["status"] == "Critical"


def test_evaluation_ignores_future_rows():
    data = stream(periods=50)
    as_of = data.timestamp.iloc[30]
    expected = SensorHealthEngine().evaluate(data.iloc[:31], as_of)
    changed = data.copy()
    changed.loc[changed.index[31]:, "pm25"] = 1_000_000
    actual = SensorHealthEngine().evaluate(changed, as_of)
    assert actual == expected


def test_nablus_structural_gas_missingness_has_zero_penalty():
    data = stream(sensor_id=5)
    with_absent_gases = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    data[["no2", "o3"]] = 999999
    with_irrelevant_gases = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])
    assert with_absent_gases["health_score"] == with_irrelevant_gases["health_score"] == 100
    assert with_absent_gases["dimensions"] == with_irrelevant_gases["dimensions"]


def test_issue_contract_contains_explanation_and_alert_readiness():
    data = stream(sensor_id=2)
    data.loc[data.index[-1], "pm25"] = np.nan
    issue = SensorHealthEngine().evaluate(data, data.timestamp.iloc[-1])["issues"][0]
    assert {"type", "severity", "channel", "timestamp", "evidence", "message", "alert_recommended"} <= issue.keys()
