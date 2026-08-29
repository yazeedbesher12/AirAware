import json

import numpy as np
import pandas as pd

from app.analysis.phase3_pipeline import (
    add_exact_lag, add_future_value, backward_rolling, build_segments,
    prepare_targets, who_analysis,
)


def sample_frame(with_gap: bool = False) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC").tolist()
    if with_gap:
        timestamps += pd.date_range("2026-01-03", periods=4, freq="15min", tz="UTC").tolist()
    size = len(timestamps)
    frame = pd.DataFrame({
        "timestamp": timestamps, "sensor_id": 1, "city_name": "Tulkarem",
        "pm25": np.arange(1, size + 1, dtype=float), "temperature": 20.0,
        "humidity": 50.0, "no2": 10.0, "o3": 60.0,
        "missing_flag": False,
    })
    return build_segments(frame)


def who_config() -> dict:
    return {
        "source_version": "WHO 2021", "source_url": "official", "minimum_window_coverage_percent": 75,
        "pollutants": [
            {"pollutant": "pm25", "guideline": 15, "unit": "ug/m3", "averaging_period": "24h", "dataset_presence": "AVAILABLE", "dataset_unit": "ug/m3", "unit_compatible": True, "interim_targets": {"IT-4": 25}},
            {"pollutant": "no2", "guideline": 25, "unit": "ug/m3", "averaging_period": "24h", "dataset_presence": "TULKAREM_ONLY", "dataset_unit": "UNVERIFIED", "unit_compatible": False, "interim_targets": {}},
        ],
    }


def test_segments_reset_at_major_gap():
    frame = sample_frame(with_gap=True)
    assert frame.continuous_segment_id.nunique() == 2


def test_exact_lag_never_crosses_gap():
    frame = sample_frame(with_gap=True)
    lag = add_exact_lag(frame, "pm25", 15)
    second_start = frame.index[frame.continuous_segment_number.eq(2)][0]
    assert pd.isna(lag.loc[second_start])
    assert lag.loc[1] == frame.loc[0, "pm25"]


def test_backward_rolling_does_not_use_future_or_prior_segment():
    frame = sample_frame(with_gap=True)
    rolling = backward_rolling(frame, "pm25", 60)
    assert rolling["mean"].loc[1] == 1.5
    second_start = frame.index[frame.continuous_segment_number.eq(2)][0]
    assert rolling["count"].loc[second_start] == 1


def test_future_target_uses_exact_elapsed_timestamp():
    frame = sample_frame()
    target = add_future_value(frame, frame.pm25, 60)
    assert target.loc[0] == frame.loc[4, "pm25"]
    assert pd.isna(target.iloc[-1])


def test_who_average_requires_full_coverage_threshold():
    frame = sample_frame()
    results, _, series = who_analysis(frame, who_config())
    assert series["pm25"]["coverage"].iloc[0] < 75
    assert results.loc[results.pollutant.eq("pm25"), "status"].iloc[0] == "Insufficient Data"


def test_unverified_who_unit_is_not_compared():
    frame = sample_frame()
    frame = pd.concat([frame] * 10, ignore_index=True)
    frame["timestamp"] = pd.date_range("2026-01-01", periods=len(frame), freq="15min", tz="UTC")
    frame = build_segments(frame)
    results, _, _ = who_analysis(frame, who_config())
    no2 = results.loc[results.pollutant.eq("no2")]
    assert "Unit Not Compatible" in set(no2.status)
    assert no2.ratio.isna().all()


def test_structurally_unavailable_nablus_gas_is_not_fabricated():
    frame = sample_frame()
    frame["city_name"] = "Nablus"
    frame["no2"] = np.nan
    results, _, _ = who_analysis(frame, who_config())
    no2 = results.loc[results.pollutant.eq("no2")]
    assert set(no2.status) == {"Not Available"}
    assert no2.average.isna().all()


def test_event_target_uses_future_who_average_not_instantaneous_threshold():
    frame = sample_frame()
    frame["pm25_lag_6h"] = 1.0
    frame["pm25_mean_3h"] = 1.0
    who_series = {"pm25": {"average": pd.Series(20.0, index=frame.index), "coverage": pd.Series(100.0, index=frame.index)}}
    targets, _, _ = prepare_targets(frame, who_series)
    assert targets.loc[0, "pollution_event_1h"] == 1
    assert targets.loc[0, "target_pm25_1h"] == frame.loc[4, "pm25"]


def test_output_order_is_chronological_within_sensor():
    frame = sample_frame(with_gap=True)
    assert frame.groupby("sensor_id").timestamp.apply(lambda value: value.is_monotonic_increasing).all()
