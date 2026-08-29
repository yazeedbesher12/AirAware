"""AirAware Phase 1 audit and conservative cleaning pipeline.

The pipeline never edits the source CSV. It creates a cleaned copy with compact
row-level metadata flags and separate event/report artifacts for detailed QA.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.config import (
    ANALYSIS_JSON,
    CHANGE_LOG_CSV,
    CLEANED_CSV,
    ENVIRONMENTAL_FEATURES,
    FEATURE_UNITS,
    GAPS_CSV,
    MISSING_PERIODS_CSV,
    RAW_CSV,
    REPORT_DIR,
    SUSPICIOUS_CSV,
    ensure_directories,
)

PIPELINE_VERSION = "1.0.0"


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mode_seconds(seconds: pd.Series) -> float | None:
    values = seconds.dropna().round(3)
    if values.empty:
        return None
    modes = values.mode()
    return float(modes.iloc[0]) if not modes.empty else None


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return None
    value = left.corr(right, method=method)
    return _json_value(value)


def _longest_true_runs(mask: pd.Series, timestamps: pd.Series) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start: int | None = None
    values = mask.fillna(False).to_numpy(dtype=bool)
    for index, active in enumerate(values):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(values) - 1):
            end = index if active and index == len(values) - 1 else index - 1
            runs.append(
                {
                    "start_time": timestamps.iloc[start],
                    "end_time": timestamps.iloc[end],
                    "readings": end - start + 1,
                }
            )
            start = None
    return runs


def _gap_severity(missing_expected: int, duration_seconds: float, expected_seconds: float) -> str:
    if missing_expected <= 0:
        return "NORMAL"
    if missing_expected == 1:
        return "MINOR"
    if missing_expected <= 4:
        return "MODERATE"
    if duration_seconds < 24 * 3600 and missing_expected < max(96, int(24 * 3600 / expected_seconds)):
        return "MAJOR"
    return "CRITICAL"


def _availability(raw: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sensor_id, group in raw.groupby("sensor_id", sort=True):
        row: dict[str, Any] = {
            "sensor_id": int(sensor_id),
            "city": group["city_name"].iloc[0],
            "location": group["location_name"].iloc[0],
            "readings": int(len(group)),
        }
        for feature in ENVIRONMENTAL_FEATURES:
            present = int(group[feature].notna().sum())
            if present == 0:
                status = "STRUCTURALLY_UNAVAILABLE"
            elif present < len(group):
                status = "PARTIALLY_MISSING"
            else:
                status = "AVAILABLE"
            row[feature] = status
            row[f"{feature}_present"] = present
        output.append(row)
    return output


def _sampling_and_gaps(raw: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for sensor_id, group in raw.groupby("sensor_id", sort=True):
        ordered = group.sort_values("timestamp")
        unique_times = ordered["timestamp"].drop_duplicates().sort_values()
        intervals = unique_times.diff().dt.total_seconds().dropna()
        median_seconds = float(intervals.median()) if not intervals.empty else 0.0
        mode_seconds = _mode_seconds(intervals) or median_seconds
        expected_seconds = mode_seconds if mode_seconds > 0 else median_seconds
        sensor_gaps: list[dict[str, Any]] = []
        if expected_seconds > 0:
            previous = unique_times.shift(1)
            for gap_end, gap_start, duration in zip(unique_times, previous, unique_times.diff().dt.total_seconds()):
                if pd.isna(duration) or duration <= expected_seconds * 1.5:
                    continue
                missing_expected = max(0, int(round(duration / expected_seconds)) - 1)
                item = {
                    "sensor_id": int(sensor_id),
                    "city": ordered["city_name"].iloc[0],
                    "location": ordered["location_name"].iloc[0],
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "duration_seconds": float(duration),
                    "duration_hours": float(duration / 3600),
                    "missing_expected_readings": missing_expected,
                    "severity": _gap_severity(missing_expected, duration, expected_seconds),
                }
                gaps.append(item)
                sensor_gaps.append(item)
        summaries.append(
            {
                "sensor_id": int(sensor_id),
                "city": ordered["city_name"].iloc[0],
                "location": ordered["location_name"].iloc[0],
                "median_interval_seconds": median_seconds,
                "mode_interval_seconds": mode_seconds,
                "min_interval_seconds": float(intervals.min()) if not intervals.empty else None,
                "max_interval_seconds": float(intervals.max()) if not intervals.empty else None,
                "expected_frequency_minutes": expected_seconds / 60 if expected_seconds else None,
                "missing_expected_readings": int(sum(item["missing_expected_readings"] for item in sensor_gaps)),
                "gap_count": len(sensor_gaps),
                "major_gap_count": sum(item["severity"] in {"MAJOR", "CRITICAL"} for item in sensor_gaps),
                "longest_gap_hours": max((item["duration_hours"] for item in sensor_gaps), default=0.0),
                "interval_distribution": [
                    {"minutes": float(seconds / 60), "count": int(count)}
                    for seconds, count in sorted(Counter(intervals.round(3).tolist()).items())
                ],
            }
        )
    return summaries, pd.DataFrame(gaps)


def _missing_analysis(raw: pd.DataFrame, availability: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    structural = {
        (row["sensor_id"], feature)
        for row in availability
        for feature in ENVIRONMENTAL_FEATURES
        if row[feature] == "STRUCTURALLY_UNAVAILABLE"
    }
    per_column = []
    for column in raw.columns:
        count = int(raw[column].isna().sum())
        per_column.append({"column": column, "missing_count": count, "missing_percentage": 100 * count / len(raw)})
    per_city = []
    for city, group in raw.groupby("city_name"):
        for feature in ENVIRONMENTAL_FEATURES:
            count = int(group[feature].isna().sum())
            per_city.append({"city": city, "feature": feature, "missing_count": count, "missing_percentage": 100 * count / len(group)})
    per_sensor = []
    periods: list[dict[str, Any]] = []
    for sensor_id, group in raw.sort_values("timestamp").groupby("sensor_id", sort=True):
        city = group["city_name"].iloc[0]
        for feature in ENVIRONMENTAL_FEATURES:
            missing = group[feature].isna()
            count = int(missing.sum())
            runs = _longest_true_runs(missing, group["timestamp"])
            if (int(sensor_id), feature) in structural:
                classification = "STRUCTURALLY_UNAVAILABLE"
            elif count and runs and max(run["readings"] for run in runs) == 1:
                classification = "RANDOM_MISSING"
            elif count:
                classification = "UNKNOWN_MISSING"
            else:
                classification = "NONE"
            per_sensor.append(
                {
                    "sensor_id": int(sensor_id),
                    "city": city,
                    "feature": feature,
                    "missing_count": count,
                    "missing_percentage": 100 * count / len(group),
                    "longest_missing_sequence": max((run["readings"] for run in runs), default=0),
                    "classification": classification,
                }
            )
            for run in runs:
                periods.append(
                    {
                        "sensor_id": int(sensor_id),
                        "city": city,
                        "feature": feature,
                        **run,
                        "classification": classification,
                    }
                )
    return {"per_column": per_column, "per_city": per_city, "per_sensor": per_sensor}, pd.DataFrame(periods)


def _duplicate_analysis(raw: pd.DataFrame) -> dict[str, Any]:
    full_mask = raw.duplicated(keep=False)
    sensor_time_mask = raw.duplicated(["sensor_id", "timestamp"], keep=False)
    repeated_measurements = raw.duplicated(
        ["sensor_id", "timestamp", *ENVIRONMENTAL_FEATURES], keep=False
    )
    timestamp_counts = raw.groupby("timestamp").size()
    return {
        "full_duplicate_rows": int(raw.duplicated().sum()),
        "full_duplicate_rows_in_groups": int(full_mask.sum()),
        "duplicate_timestamps_global": int((timestamp_counts > 1).sum()),
        "duplicate_timestamp_rows_per_sensor": int(sensor_time_mask.sum()),
        "duplicate_sensor_timestamp_groups": int(raw.loc[sensor_time_mask].groupby(["sensor_id", "timestamp"]).ngroups),
        "repeated_record_rows": int(repeated_measurements.sum()),
        "logic_note": "Only sensor_id + timestamp collisions are treated as timestamp duplicates; simultaneous readings from different sensors are legitimate.",
    }


def _numeric_audit(raw: pd.DataFrame) -> list[dict[str, Any]]:
    output = []
    for feature in ENVIRONMENTAL_FEATURES:
        values = raw[feature]
        finite = values[np.isfinite(values)]
        output.append(
            {
                "feature": feature,
                "unit": FEATURE_UNITS[feature],
                "count": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "infinite": int(np.isinf(values).sum()),
                "zero_count": int((values == 0).sum()),
                "negative_count": int((values < 0).sum()),
                "min": finite.min() if not finite.empty else None,
                "max": finite.max() if not finite.empty else None,
                "mean": finite.mean() if not finite.empty else None,
                "median": finite.median() if not finite.empty else None,
                "std": finite.std() if len(finite) > 1 else None,
                "p95": finite.quantile(0.95) if not finite.empty else None,
                "p99": finite.quantile(0.99) if not finite.empty else None,
            }
        )
    return output


def _value_events(raw: pd.DataFrame, availability: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    structural = {
        (row["sensor_id"], feature)
        for row in availability
        for feature in ENVIRONMENTAL_FEATURES
        if row[feature] == "STRUCTURALLY_UNAVAILABLE"
    }
    broad_bounds = {
        "pm25": (0, None), "temperature": (-50, 70), "humidity": (0, 100),
        "no2": (0, None), "o3": (0, None), "aqi": (0, None),
    }
    for sensor_id, group in raw.sort_values("timestamp").groupby("sensor_id", sort=True):
        for feature in ENVIRONMENTAL_FEATURES:
            values = group[feature]
            if (int(sensor_id), feature) in structural:
                continue
            finite = values.replace([np.inf, -np.inf], np.nan).dropna()
            q1, q3 = (finite.quantile(0.25), finite.quantile(0.75)) if len(finite) else (np.nan, np.nan)
            iqr = q3 - q1
            data_extreme = q3 + 4 * iqr if pd.notna(iqr) and iqr > 0 else np.inf
            low, high = broad_bounds[feature]
            previous = values.shift(1)
            delta = values - previous
            relative = delta / previous.abs().replace(0, np.nan)
            for idx in group.index:
                value = values.loc[idx]
                flag = None
                reason = None
                if pd.isna(value):
                    flag, reason = "MISSING", "Value is absent within a feature the sensor otherwise provides."
                elif not np.isfinite(value):
                    flag, reason = "POSSIBLE_SENSOR_ERROR", "Non-finite numeric value."
                elif value < low or (high is not None and value > high):
                    flag, reason = "POSSIBLE_SENSOR_ERROR", "Value is outside broad physical plausibility bounds."
                elif value > data_extreme:
                    flag, reason = "EXTREME", "Value exceeds the sensor-feature Q3 + 4×IQR fence; retained for review."
                elif value == 0 and feature in {"pm25", "no2", "o3"}:
                    flag, reason = "SUSPICIOUS", "Zero reading may be valid but warrants sensor-context review."
                if flag:
                    events.append(
                        {
                            "row_index": int(idx), "timestamp": group.at[idx, "timestamp"],
                            "city": group.at[idx, "city_name"], "sensor_id": int(sensor_id),
                            "feature": feature, "value": value, "previous_value": previous.loc[idx],
                            "absolute_change": abs(delta.loc[idx]) if pd.notna(delta.loc[idx]) else None,
                            "relative_change": abs(relative.loc[idx]) if pd.notna(relative.loc[idx]) else None,
                            "flag": flag, "reason": reason,
                        }
                    )
    return events


def _jump_events(raw: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for sensor_id, group in raw.sort_values("timestamp").groupby("sensor_id", sort=True):
        for feature in ENVIRONMENTAL_FEATURES:
            values = group[feature]
            diff = values.diff()
            absolute = diff.abs()
            finite = absolute.dropna()
            if len(finite) < 12:
                continue
            median = float(finite.median())
            mad = float(stats.median_abs_deviation(finite, nan_policy="omit", scale="normal"))
            q1, q3 = finite.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            threshold = max(median + 8 * mad, float(q3 + 3 * iqr), 1e-12)
            elapsed_hours = group["timestamp"].diff().dt.total_seconds() / 3600
            candidates = absolute >= threshold
            for idx in group.index[candidates.fillna(False)]:
                previous = values.shift(1).loc[idx]
                delta = diff.loc[idx]
                events.append(
                    {
                        "row_index": int(idx), "timestamp": group.at[idx, "timestamp"],
                        "city": group.at[idx, "city_name"], "sensor_id": int(sensor_id),
                        "feature": feature, "value": values.loc[idx], "previous_value": previous,
                        "absolute_change": abs(delta),
                        "relative_change": abs(delta / previous) if pd.notna(previous) and previous != 0 else None,
                        "rate_per_hour": abs(delta) / elapsed_hours.loc[idx] if elapsed_hours.loc[idx] > 0 else None,
                        "robust_threshold": threshold,
                        "robust_strength": abs(delta) / threshold if threshold else None,
                        "flag": "SUSPICIOUS",
                        "reason": "Sensor-specific sudden change exceeds both robust MAD and IQR criteria.",
                    }
                )
    return events


def _flatline_events(raw: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for sensor_id, group in raw.sort_values("timestamp").groupby("sensor_id", sort=True):
        expected = group["timestamp"].drop_duplicates().sort_values().diff().dt.total_seconds().mode()
        expected_seconds = float(expected.iloc[0]) if not expected.empty else 900.0
        minimum_readings = max(8, int(round(2 * 3600 / expected_seconds)))
        for feature in ENVIRONMENTAL_FEATURES:
            series = group[feature]
            if series.notna().sum() == 0:
                continue
            run_id = (series.ne(series.shift()) | series.isna()).cumsum()
            for _, run in group.assign(_value=series, _run=run_id).dropna(subset=["_value"]).groupby("_run"):
                if len(run) < minimum_readings:
                    continue
                duration = (run["timestamp"].iloc[-1] - run["timestamp"].iloc[0]).total_seconds()
                events.append(
                    {
                        "sensor_id": int(sensor_id), "city": group["city_name"].iloc[0],
                        "feature": feature, "start_time": run["timestamp"].iloc[0],
                        "end_time": run["timestamp"].iloc[-1], "duration_hours": duration / 3600,
                        "readings": int(len(run)), "value": run["_value"].iloc[0],
                        "flag": "POSSIBLE_STUCK_PATTERN",
                        "reason": "Long identical-value run; pattern only, not confirmed hardware failure.",
                    }
                )
            rolling_std = series.rolling(minimum_readings, min_periods=minimum_readings).std()
            tolerance = max(float(series.dropna().quantile(0.75) - series.dropna().quantile(0.25)) * 0.001, 1e-6)
            near_mask = rolling_std <= tolerance
            for run in _longest_true_runs(near_mask, group["timestamp"]):
                if run["readings"] >= minimum_readings * 2:
                    events.append(
                        {
                            "sensor_id": int(sensor_id), "city": group["city_name"].iloc[0],
                            "feature": feature, **run,
                            "duration_hours": (run["end_time"] - run["start_time"]).total_seconds() / 3600,
                            "value": None, "flag": "POSSIBLE_STUCK_PATTERN",
                            "reason": "Near-zero rolling variance period; overlaps may be consolidated during review.",
                        }
                    )
    return events


def _drift_events(raw: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for sensor_id, group in raw.sort_values("timestamp").groupby("sensor_id", sort=True):
        indexed = group.set_index("timestamp")
        for feature in ENVIRONMENTAL_FEATURES:
            values = indexed[feature].dropna()
            if len(values) < 192:
                continue
            short = values.rolling("24h", min_periods=48).median()
            baseline = values.rolling("7d", min_periods=192).median()
            delta = short - baseline
            finite = delta.dropna()
            if len(finite) < 48:
                continue
            scale = float(stats.median_abs_deviation(finite, nan_policy="omit", scale="normal"))
            if not np.isfinite(scale) or scale <= 0:
                continue
            mask = delta.abs() > 3 * scale
            timestamps = pd.Series(mask.index, index=range(len(mask)))
            for run in _longest_true_runs(pd.Series(mask.to_numpy()), timestamps):
                if run["readings"] < 24:
                    continue
                section = delta.loc[run["start_time"]:run["end_time"]]
                events.append(
                    {
                        "sensor_id": int(sensor_id), "city": group["city_name"].iloc[0],
                        "feature": feature, "start_time": run["start_time"], "end_time": run["end_time"],
                        "magnitude": float(section.median()), "readings": int(run["readings"]),
                        "label": "Possible Baseline Shift",
                        "evidence": "24-hour rolling median differs from the trailing 7-day median by more than 3 robust baseline scales.",
                    }
                )
    return events


def _rolling_summary(raw: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    windows = ["1h", "3h", "6h", "12h", "24h"]
    for sensor_id, group in raw.sort_values("timestamp").groupby("sensor_id", sort=True):
        indexed = group.set_index("timestamp")
        for feature in ENVIRONMENTAL_FEATURES:
            values = indexed[feature].dropna()
            if values.empty:
                continue
            item: dict[str, Any] = {
                "sensor_id": int(sensor_id), "city": group["city_name"].iloc[0], "feature": feature,
                "overall_std": values.std(), "overall_variance": values.var(),
            }
            for window in windows:
                rolled = values.rolling(window, min_periods=2)
                item[f"{window}_mean_latest"] = rolled.mean().iloc[-1]
                item[f"{window}_median_latest"] = rolled.median().iloc[-1]
                item[f"{window}_std_median"] = rolled.std().median()
                item[f"{window}_variance_median"] = rolled.var().median()
                item[f"{window}_min"] = rolled.min().min()
                item[f"{window}_max"] = rolled.max().max()
            output.append(item)
    return output


def _aqi_analysis(raw: pd.DataFrame) -> dict[str, Any]:
    values = raw["aqi"]
    stats_record = {
        "count": int(values.notna().sum()), "min": values.min(), "max": values.max(),
        "mean": values.mean(), "median": values.median(), "std": values.std(),
        "above_500_count": int((values > 500).sum()),
    }
    relationships = []
    for sensor_id, group in raw.groupby("sensor_id", sort=True):
        for feature in ["pm25", "temperature", "humidity", "no2", "o3"]:
            pair = group[["aqi", feature]].dropna()
            relationships.append(
                {
                    "sensor_id": int(sensor_id), "city": group["city_name"].iloc[0], "feature": feature,
                    "overlap_count": int(len(pair)),
                    "pearson": _safe_corr(pair["aqi"], pair[feature], "pearson"),
                    "spearman": _safe_corr(pair["aqi"], pair[feature], "spearman"),
                }
            )
    top_columns = ["timestamp", "city_name", "sensor_id", "pm25", "temperature", "humidity", "no2", "o3", "aqi"]
    return {
        "warning": "Existing AQI methodology not yet verified.",
        "statistics": stats_record,
        "relationships": relationships,
        "top_observations": _records(raw.nlargest(50, "aqi")[top_columns]),
        "above_500": _records(raw.loc[raw["aqi"] > 500, top_columns].sort_values("aqi", ascending=False)),
    }


def _nablus_comparison(raw: pd.DataFrame) -> list[dict[str, Any]]:
    nablus = raw.loc[raw["city_name"].str.casefold() == "nablus"]
    output: list[dict[str, Any]] = []
    sensors = sorted(nablus["sensor_id"].unique())
    for left_id, right_id in itertools.combinations(sensors, 2):
        left = nablus.loc[nablus["sensor_id"] == left_id].set_index("timestamp")
        right = nablus.loc[nablus["sensor_id"] == right_id].set_index("timestamp")
        for feature in ["pm25", "temperature", "humidity", "aqi"]:
            pair = pd.concat([left[feature].rename("left"), right[feature].rename("right")], axis=1, join="inner").dropna()
            absolute = (pair["left"] - pair["right"]).abs()
            denominator = ((pair["left"].abs() + pair["right"].abs()) / 2).replace(0, np.nan)
            output.append(
                {
                    "feature": feature, "sensor_a": int(left_id), "sensor_b": int(right_id),
                    "sensor_pair": f"{left_id} ↔ {right_id}", "overlap_count": int(len(pair)),
                    "pearson": _safe_corr(pair["left"], pair["right"], "pearson"),
                    "spearman": _safe_corr(pair["left"], pair["right"], "spearman"),
                    "mean_absolute_difference": absolute.mean() if len(pair) else None,
                    "median_absolute_difference": absolute.median() if len(pair) else None,
                    "mean_relative_difference": (absolute / denominator).mean() if len(pair) else None,
                }
            )
    return output


def _quality_summaries(
    raw: pd.DataFrame,
    availability: list[dict[str, Any]],
    sampling: list[dict[str, Any]],
    flatlines: list[dict[str, Any]],
    jumps: list[dict[str, Any]],
    drifts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    availability_map = {row["sensor_id"]: row for row in availability}
    sampling_map = {row["sensor_id"]: row for row in sampling}
    for sensor_id, group in raw.groupby("sensor_id", sort=True):
        provided = [feature for feature in ENVIRONMENTAL_FEATURES if availability_map[int(sensor_id)][feature] != "STRUCTURALLY_UNAVAILABLE"]
        missing_rate = 100 * group[provided].isna().sum().sum() / (len(group) * len(provided)) if provided else 0
        sensor_flatlines = [item for item in flatlines if item["sensor_id"] == int(sensor_id)]
        sensor_jumps = [item for item in jumps if item["sensor_id"] == int(sensor_id)]
        sensor_drifts = [item for item in drifts if item["sensor_id"] == int(sensor_id)]
        sampling_row = sampling_map[int(sensor_id)]
        reasons = []
        statuses = []
        if sampling_row["longest_gap_hours"] >= 24:
            statuses.append("Poor Availability")
            reasons.append(f"Longest outage is {sampling_row['longest_gap_hours']:.1f} hours.")
        if sensor_drifts:
            statuses.append("Possible Drift")
            reasons.append(f"{len(sensor_drifts)} exploratory baseline-shift period(s).")
        if sensor_flatlines:
            statuses.append("Possible Stuck Pattern")
            reasons.append(f"{len(sensor_flatlines)} long low-variance or identical-value pattern(s).")
        if sensor_jumps:
            reasons.append(f"{len(sensor_jumps)} robust sudden-change flag(s) require review.")
        if not statuses:
            statuses.append("Needs Review" if sensor_jumps or missing_rate > 1 else "Healthy-looking")
        output.append(
            {
                "sensor_id": int(sensor_id), "city": group["city_name"].iloc[0],
                "location": group["location_name"].iloc[0], "readings": int(len(group)),
                "missing_rate": missing_rate, "longest_gap_hours": sampling_row["longest_gap_hours"],
                "gap_count": sampling_row["gap_count"], "flatline_events": len(sensor_flatlines),
                "sudden_jump_count": len(sensor_jumps), "possible_drift_periods": len(sensor_drifts),
                "variability": {feature: _json_value(group[feature].std()) for feature in provided},
                "available_measurements": provided, "statuses": statuses,
                "explanation": " ".join(reasons) or "No major availability or pattern flags were found by the documented Phase 1 checks.",
            }
        )
    return output


def _write_reports(analysis: dict[str, Any]) -> None:
    overview = analysis["overview"]
    sampling = analysis["sampling"]
    quality = analysis["sensor_quality"]
    gaps = sorted(analysis["gaps"], key=lambda item: item["duration_hours"], reverse=True)
    comparisons = [item for item in analysis["nablus_comparison"] if item["feature"] == "pm25" and item["pearson"] is not None]
    strongest = max(comparisons, key=lambda item: item["pearson"], default=None)
    divergent = max(comparisons, key=lambda item: item["mean_absolute_difference"] or -1, default=None)
    best = min(quality, key=lambda item: item["missing_rate"])
    worst = max(quality, key=lambda item: item["missing_rate"])
    schema_table = "\n".join(f"| {column} | {dtype} |" for column, dtype in overview["dtypes"].items())
    missing_table = "\n".join(
        f"| {row['column']} | {row['missing_count']:,} | {row['missing_percentage']:.3f}% |"
        for row in analysis["missing"]["per_column"]
    )
    numeric_table = "\n".join(
        f"| {row['feature']} | {row['count']:,} | {row['min']} | {row['max']} | {row['mean']:.3f} | {row['median']:.3f} | {row['negative_count']} | {row['infinite']} |"
        for row in analysis["numeric_audit"]
    )
    sampling_table = "\n".join(
        f"| {row['sensor_id']} | {row['city']} | {row['expected_frequency_minutes']:.1f} | {row['min_interval_seconds']/60:.1f} | {row['max_interval_seconds']/60:.1f} | {row['gap_count']} | {row['missing_expected_readings']} |"
        for row in sampling
    )
    availability_table = "\n".join(
        f"| {row['sensor_id']} | {row['city']} | {row['location']} | " + " | ".join(row[feature] for feature in ENVIRONMENTAL_FEATURES) + " |"
        for row in analysis["availability"]
    )
    reports = {
        "01_data_audit_report.md": f"""# AirAware Phase 1 — Data Audit

Generated: {analysis['generated_at']}  
Source SHA-256: `{overview['source_sha256']}`

## Structure

- Rows: {overview['row_count']:,}
- Columns: {overview['column_count']}
- Cities: {', '.join(overview['cities'])}
- Sensors: {overview['sensor_count']}
- Time range: {overview['earliest_timestamp']} to {overview['latest_timestamp']}
- Memory use: {overview['memory_usage_bytes']:,} bytes
- Timestamp parse failures: {overview['timestamp_parse_failures']}
- Coordinate fields found: {', '.join(overview['coordinate_fields']) if overview['coordinate_fields'] else 'none'}

## Schema

| Column | Parsed type |
|---|---|
{schema_table}

## Missingness by column

| Column | Missing | Missing % |
|---|---:|---:|
{missing_table}

## Numeric ranges

| Feature | Count | Min | Max | Mean | Median | Negative | Infinite |
|---|---:|---:|---:|---:|---:|---:|---:|
{numeric_table}

## Sensor measurement availability

| Sensor | City | Location | PM2.5 | Temperature | Humidity | NO2 | O3 | AQI |
|---:|---|---|---|---|---|---|---|---|
{availability_table}

## Sensor-specific sampling

| Sensor | City | Expected min | Min interval min | Max interval min | Gaps | Missing expected |
|---:|---|---:|---:|---:|---:|---:|
{sampling_table}

## Duplicate interpretation

{analysis['duplicates']['logic_note']}

Full duplicate rows: {analysis['duplicates']['full_duplicate_rows']}. Sensor/timestamp duplicate rows: {analysis['duplicates']['duplicate_timestamp_rows_per_sensor']}.
""",
        "02_cleaning_report.md": f"""# AirAware Phase 1 — Conservative Cleaning Report

The source file was never overwritten. Parsing normalized timestamps and numeric representation in the derived file only. No AQI values were replaced, no structurally unavailable readings were fabricated, and no statistical outliers were automatically removed.

- Original rows: {overview['row_count']:,}
- Cleaned rows: {overview['row_count']:,}
- Rows removed: 0
- Observation values modified: 0
- Suspicious event records retained: {overview['suspicious_observations']:,}
- Metadata columns added: `missing_flag`, `sampling_gap_flag`, `suspicious_value_flag`, `flatline_flag`, `jump_flag`, `possible_drift_flag`, `overall_quality_flag`

The machine-readable modification ledger is `outputs/cleaning_change_log.csv`; it contains only its header because no source field was changed.
""",
        "03_sensor_quality_report.md": "# AirAware Phase 1 — Sensor Quality\n\n" + "\n\n".join(
            f"## Sensor {item['sensor_id']} — {item['location']}\n\nStatus: {', '.join(item['statuses'])}. Missing among provided features: {item['missing_rate']:.2f}%. Longest gap: {item['longest_gap_hours']:.2f} h. {item['explanation']}"
            for item in quality
        ),
        "04_phase1_findings.md": f"""# AirAware Phase 1 — Findings Before Advanced EDA

## Evidence summary

1. Best completeness: sensor {best['sensor_id']} ({best['location']}) at {best['missing_rate']:.2f}% missing among measurements it provides.
2. Worst completeness: sensor {worst['sensor_id']} ({worst['location']}) at {worst['missing_rate']:.2f}% missing among measurements it provides.
3. Largest outage: {f"sensor {gaps[0]['sensor_id']}, {gaps[0]['duration_hours']:.2f} hours ({gaps[0]['gap_start']} to {gaps[0]['gap_end']})" if gaps else "none detected"}.
4. Major/critical gap count: {sum(item['severity'] in {'MAJOR', 'CRITICAL'} for item in gaps)}.
5. Suspicious value/jump event records retained: {overview['suspicious_observations']:,}.
6. Possible stuck-pattern periods: {overview['flatline_events']:,}; these are patterns, not confirmed hardware failures.
7. Exploratory behavior-change periods: {overview['drift_events']:,}; these do not confirm calibration drift.
8. Strongest Nablus PM2.5 agreement: {f"sensors {strongest['sensor_pair']} (Pearson {strongest['pearson']:.3f}, n={strongest['overlap_count']})" if strongest else "insufficient overlapping data"}.
9. Largest Nablus PM2.5 divergence: {f"sensors {divergent['sensor_pair']} (MAE {divergent['mean_absolute_difference']:.2f} µg/m³, n={divergent['overlap_count']})" if divergent else "insufficient overlapping data"}.
10. Existing AQI range: {analysis['aqi_analysis']['statistics']['min']}–{analysis['aqi_analysis']['statistics']['max']}; values above 500: {analysis['aqi_analysis']['statistics']['above_500_count']}. Existing AQI methodology is not verified.

## Major concerns before Advanced EDA

Structural NO2/O3 unavailability is kept distinct from random missingness. Multi-day outages can bias comparisons. Robust jump, flat-line, and baseline-shift flags require contextual review; no pollution cause or hardware failure is inferred here. The existing AQI must remain exploratory until its source methodology is established.
""",
    }
    for filename, content in reports.items():
        (REPORT_DIR / filename).write_text(content.strip() + "\n", encoding="utf-8")


def run_pipeline(source_path: Path = RAW_CSV) -> dict[str, Any]:
    ensure_directories()
    if not source_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {source_path}")

    source_hash = _sha256(source_path)
    raw_text = pd.read_csv(source_path, dtype=str)
    raw = raw_text.copy()
    numeric_columns = ["sensor_id", "city_id", *ENVIRONMENTAL_FEATURES]
    parse_problems: dict[str, int] = {}
    for column in numeric_columns:
        parsed = pd.to_numeric(raw[column], errors="coerce")
        parse_problems[column] = int((raw[column].notna() & parsed.isna()).sum())
        raw[column] = parsed
    raw["sensor_id"] = raw["sensor_id"].astype("Int64")
    raw["city_id"] = raw["city_id"].astype("Int64")
    parsed_timestamp = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    parse_problems["timestamp"] = int((raw["timestamp"].notna() & parsed_timestamp.isna()).sum())
    raw["timestamp"] = parsed_timestamp
    raw = raw.reset_index(drop=True)

    availability = _availability(raw)
    sampling, gaps_frame = _sampling_and_gaps(raw)
    missing, missing_periods = _missing_analysis(raw, availability)
    if not gaps_frame.empty:
        time_gap_periods = []
        availability_map = {row["sensor_id"]: row for row in availability}
        for gap in gaps_frame.to_dict("records"):
            for feature in ENVIRONMENTAL_FEATURES:
                if availability_map[int(gap["sensor_id"])][feature] == "STRUCTURALLY_UNAVAILABLE":
                    continue
                time_gap_periods.append({
                    "sensor_id": int(gap["sensor_id"]), "city": gap["city"], "feature": feature,
                    "start_time": gap["gap_start"], "end_time": gap["gap_end"],
                    "readings": int(gap["missing_expected_readings"]), "classification": "TIME_GAP",
                })
        missing_periods = pd.concat([missing_periods, pd.DataFrame(time_gap_periods)], ignore_index=True)
    duplicates = _duplicate_analysis(raw)
    numeric = _numeric_audit(raw)
    value_events = _value_events(raw, availability)
    jump_events = _jump_events(raw)
    flatline_events = _flatline_events(raw)
    drift_events = _drift_events(raw)
    rolling = _rolling_summary(raw)
    nablus = _nablus_comparison(raw)
    aqi = _aqi_analysis(raw)

    suspicious_frame = pd.DataFrame(value_events + jump_events)
    if not suspicious_frame.empty:
        suspicious_frame = suspicious_frame.sort_values(["timestamp", "sensor_id", "feature"]).reset_index(drop=True)
    sensor_quality = _quality_summaries(raw, availability, sampling, flatline_events, jump_events, drift_events)

    # Start from the original text frame so all source field representations are
    # byte-for-byte equivalent after CSV parsing; only metadata columns are added.
    cleaned = raw_text.copy()
    row_missing = pd.Series(False, index=raw.index)
    for sensor_id, group in raw.groupby("sensor_id"):
        provided = [feature for feature in ENVIRONMENTAL_FEATURES if next(row for row in availability if row["sensor_id"] == int(sensor_id))[feature] != "STRUCTURALLY_UNAVAILABLE"]
        row_missing.loc[group.index] = group[provided].isna().any(axis=1)
    cleaned["missing_flag"] = row_missing.to_numpy()
    cleaned["sampling_gap_flag"] = False
    if not gaps_frame.empty:
        gap_ends = set(zip(gaps_frame["sensor_id"], gaps_frame["gap_end"]))
        cleaned["sampling_gap_flag"] = [(int(sensor), timestamp) in gap_ends for sensor, timestamp in zip(raw["sensor_id"], raw["timestamp"])]
    value_rows = {item["row_index"] for item in value_events if item["flag"] not in {"MISSING"}}
    jump_rows = {item["row_index"] for item in jump_events}
    cleaned["suspicious_value_flag"] = cleaned.index.isin(value_rows)
    cleaned["jump_flag"] = cleaned.index.isin(jump_rows)
    cleaned["flatline_flag"] = False
    cleaned["possible_drift_flag"] = False
    for event in flatline_events:
        mask = (raw["sensor_id"] == event["sensor_id"]) & raw["timestamp"].between(event["start_time"], event["end_time"])
        cleaned.loc[mask, "flatline_flag"] = True
    for event in drift_events:
        mask = (raw["sensor_id"] == event["sensor_id"]) & raw["timestamp"].between(event["start_time"], event["end_time"])
        cleaned.loc[mask, "possible_drift_flag"] = True
    conditions = [
        cleaned["suspicious_value_flag"] | cleaned["jump_flag"], cleaned["flatline_flag"],
        cleaned["possible_drift_flag"], cleaned["sampling_gap_flag"], cleaned["missing_flag"],
    ]
    labels = ["SUSPICIOUS", "POSSIBLE_STUCK_PATTERN", "POSSIBLE_DRIFT", "GAP", "MISSING_OR_UNAVAILABLE"]
    cleaned["overall_quality_flag"] = np.select(conditions, labels, default="NORMAL")
    cleaned.to_csv(CLEANED_CSV, index=False)

    gaps_frame.to_csv(GAPS_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    missing_periods.to_csv(MISSING_PERIODS_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    suspicious_frame.to_csv(SUSPICIOUS_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    pd.DataFrame(columns=["row_index", "column", "original_value", "new_value", "reason"]).to_csv(CHANGE_LOG_CSV, index=False)

    sensor_timelines = []
    for sensor_id, group in raw.groupby("sensor_id", sort=True):
        sensor_timelines.append(
            {
                "sensor_id": int(sensor_id), "city": group["city_name"].iloc[0],
                "location": group["location_name"].iloc[0], "start": group["timestamp"].min(),
                "end": group["timestamp"].max(), "readings": int(len(group)),
            }
        )
    overall_missing = int(raw.isna().sum().sum())
    overview = {
        "row_count": int(len(raw)), "column_count": int(len(raw.columns)), "columns": list(raw.columns),
        "dtypes": {column: str(dtype) for column, dtype in raw.dtypes.items()},
        "memory_usage_bytes": int(raw.memory_usage(deep=True).sum()),
        "cities": sorted(raw["city_name"].dropna().unique().tolist()),
        "city_count": int(raw["city_name"].nunique()), "sensor_count": int(raw["sensor_id"].nunique()),
        "sensors_per_city": {city: [int(value) for value in sorted(group["sensor_id"].unique())] for city, group in raw.groupby("city_name")},
        "locations": _records(raw[["sensor_id", "city_name", "location_name"]].drop_duplicates()),
        "coordinate_fields": [], "timestamp_source_format": "ISO-like timestamp with UTC offset",
        "timestamp_parse_failures": parse_problems["timestamp"], "numeric_parse_problems": parse_problems,
        "earliest_timestamp": raw["timestamp"].min(), "latest_timestamp": raw["timestamp"].max(),
        "observation_duration_hours": (raw["timestamp"].max() - raw["timestamp"].min()).total_seconds() / 3600,
        "overall_cell_missing_count": overall_missing,
        "overall_cell_missing_percentage": 100 * overall_missing / raw.size,
        "environmental_missing_percentage": 100 * raw[ENVIRONMENTAL_FEATURES].isna().sum().sum() / (len(raw) * len(ENVIRONMENTAL_FEATURES)),
        "expected_sampling_minutes": float(np.median([item["expected_frequency_minutes"] for item in sampling if item["expected_frequency_minutes"]])),
        "major_gaps": int(sum(item["severity"] in {"MAJOR", "CRITICAL"} for item in _records(gaps_frame))),
        "suspicious_observations": int(len(suspicious_frame)), "flatline_events": len(flatline_events),
        "drift_events": len(drift_events), "source_sha256": source_hash,
        "cleaned_sha256": _sha256(CLEANED_CSV), "sensor_timelines": sensor_timelines,
    }
    analysis = {
        "pipeline_version": PIPELINE_VERSION, "generated_at": datetime.now(timezone.utc),
        "overview": overview, "availability": availability, "sampling": sampling,
        "gaps": _records(gaps_frame), "missing": missing, "missing_periods": _records(missing_periods),
        "duplicates": duplicates, "numeric_audit": numeric, "flatlines": flatline_events,
        "jumps": sorted(jump_events, key=lambda item: item["robust_strength"] or 0, reverse=True),
        "drifts": drift_events, "rolling_summary": rolling, "sensor_quality": sensor_quality,
        "nablus_comparison": nablus, "aqi_analysis": aqi,
        "methodology": {
            "gap_classification": "Intervals >1.5× the sensor mode interval are gaps. MINOR=1 missing reading; MODERATE=2–4; MAJOR=5+ but under 24 hours; CRITICAL=24 hours or longer.",
            "jumps": "Per sensor-feature absolute differences are flagged only when they exceed both median + 8 scaled MAD and Q3 + 3 IQR.",
            "flatlines": "Identical runs of at least eight readings (a nominal two-hour sampling block) and extended near-zero-variance windows are possible stuck patterns.",
            "drift": "Exploratory only: sustained 24-hour rolling-median displacement from a trailing 7-day median beyond three robust scales.",
            "cleaning": "No rows removed and no observation values changed. Flags are metadata; structural absence is never interpolated.",
        },
    }
    ANALYSIS_JSON.write_text(json.dumps(analysis, default=_json_value, allow_nan=False, indent=2), encoding="utf-8")
    _write_reports(json.loads(ANALYSIS_JSON.read_text(encoding="utf-8")))

    if _sha256(source_path) != source_hash:
        raise RuntimeError("Source CSV changed during pipeline execution.")
    return json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps({"overview": result["overview"], "reports": sorted(path.name for path in REPORT_DIR.glob("*.md"))}, indent=2))
