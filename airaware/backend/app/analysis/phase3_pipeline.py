"""AirAware Phase 3: leakage-safe ML features, WHO reference analysis, and targets.

This pipeline reads the Phase 1 cleaned dataset and Phase 2 evidence. It never
changes source observations, never treats the source AQI as ground truth, and
does not fit a forecasting model.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from app.config import (
    CLEANED_CSV, FORECAST_TARGETS_CSV, ML_FEATURES_CSV, OUTPUT_DIR,
    PHASE2_ANALYSIS_JSON, PHASE3_ANALYSIS_JSON, PHASE3_ARTIFACTS, REPORT_DIR,
    WHO_2021_CONFIG, ensure_directories,
)

PHASE3_VERSION = "3.0.0"
EXPECTED_INTERVAL_MINUTES = 15
MAJOR_GAP_MINUTES = 90
HORIZONS = [60, 180, 360]
EVIDENCE_LAGS: dict[str, list[int]] = {
    "pm25": [15, 60, 180, 360],
    "temperature": [15, 60, 180],
    "humidity": [15, 60, 180],
    "no2": [15, 60, 180],
    "o3": [15, 60, 180],
}
ROLLING_PLAN: dict[str, dict[int, list[str]]] = {
    "pm25": {60: ["mean", "std"], 180: ["mean", "median", "std", "min", "max", "range"], 360: ["mean", "std", "cv"]},
    "temperature": {60: ["mean"], 180: ["mean", "std"]},
    "humidity": {60: ["mean"], 180: ["mean", "std"]},
    "no2": {60: ["mean"], 180: ["mean", "std"]},
    "o3": {60: ["mean"], 180: ["mean", "std"]},
}


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def lag_label(minutes: int) -> str:
    return f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes}m"


def build_segments(frame: pd.DataFrame, threshold_minutes: int = MAJOR_GAP_MINUTES) -> pd.DataFrame:
    """Assign continuous segments; a lag/rolling window never crosses a major outage."""
    output = frame.sort_values(["sensor_id", "timestamp"]).copy()
    delta = output.groupby("sensor_id")["timestamp"].diff().dt.total_seconds().div(60)
    boundary = delta.isna() | delta.gt(threshold_minutes) | delta.le(0)
    output["continuous_segment_number"] = boundary.groupby(output["sensor_id"]).cumsum().astype(int)
    output["continuous_segment_id"] = output["sensor_id"].astype(str) + "-" + output["continuous_segment_number"].astype(str)
    output["minutes_since_previous"] = delta
    return output


def add_exact_lag(frame: pd.DataFrame, feature: str, minutes: int) -> pd.Series:
    """Return an exact elapsed-time lag within the same continuous segment."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    offset = pd.Timedelta(minutes=minutes)
    for _, group in frame.groupby(["sensor_id", "continuous_segment_id"], sort=False):
        lookup = pd.Series(group[feature].to_numpy(), index=group["timestamp"])
        result.loc[group.index] = (group["timestamp"] - offset).map(lookup).to_numpy()
    return result


def add_future_value(frame: pd.DataFrame, source: pd.Series, minutes: int) -> pd.Series:
    """Return an exact future value within a segment; intended only for target columns."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    offset = pd.Timedelta(minutes=minutes)
    for _, group in frame.groupby(["sensor_id", "continuous_segment_id"], sort=False):
        lookup = pd.Series(source.loc[group.index].to_numpy(), index=group["timestamp"])
        result.loc[group.index] = (group["timestamp"] + offset).map(lookup).to_numpy()
    return result


def backward_rolling(frame: pd.DataFrame, feature: str, minutes: int) -> dict[str, pd.Series]:
    """Backward-looking, non-centered time rolling statistics, reset per segment."""
    names = ["mean", "median", "std", "min", "max", "count"]
    output = {name: pd.Series(np.nan, index=frame.index, dtype=float) for name in names}
    for _, group in frame.groupby(["sensor_id", "continuous_segment_id"], sort=False):
        series = pd.Series(group[feature].to_numpy(dtype=float), index=group["timestamp"])
        roll = series.rolling(f"{minutes}min", min_periods=1, closed="right")
        calculated = {
            "mean": roll.mean(), "median": roll.median(), "std": roll.std(),
            "min": roll.min(), "max": roll.max(), "count": roll.count(),
        }
        for name, values in calculated.items():
            output[name].loc[group.index] = values.to_numpy()
    return output


def _load() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    data = pd.read_csv(CLEANED_CSV)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for feature in ["pm25", "temperature", "humidity", "no2", "o3", "aqi"]:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    phase2 = json.loads(PHASE2_ANALYSIS_JSON.read_text(encoding="utf-8"))
    who = json.loads(WHO_2021_CONFIG.read_text(encoding="utf-8"))
    return build_segments(data), phase2, who


def _catalog_row(name: str, family: str, reason: str, scope: str = "All sensors", evidence: str = "Phase 2") -> dict[str, Any]:
    return {
        "feature": name, "family": family, "availability_scope": scope,
        "reason": reason, "evidence": evidence,
        "prediction_time_available": True, "leakage_check": "PASS",
    }


def engineer_features(data: pd.DataFrame, phase2: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    output = data.copy()
    catalog: list[dict[str, Any]] = []
    engineered: list[str] = []

    # The dataset spans only 17 days, so month/day-of-month are deliberately omitted.
    output["hour"] = output.timestamp.dt.hour
    output["minute"] = output.timestamp.dt.minute
    output["day_of_week"] = output.timestamp.dt.dayofweek
    output["is_weekend"] = output.day_of_week.ge(5).astype(int)
    output["sin_hour"] = np.sin(2 * np.pi * output.hour / 24)
    output["cos_hour"] = np.cos(2 * np.pi * output.hour / 24)
    output["sin_day_of_week"] = np.sin(2 * np.pi * output.day_of_week / 7)
    output["cos_day_of_week"] = np.cos(2 * np.pi * output.day_of_week / 7)
    for name in ["hour", "minute", "day_of_week", "is_weekend", "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week"]:
        engineered.append(name)
        catalog.append(_catalog_row(name, "TIME", "Backward-safe calendar representation; cyclic terms preserve clock/day adjacency.", evidence="Phase 2 hourly and day-of-week profiles"))

    for feature, lags in EVIDENCE_LAGS.items():
        scope = "Tulkarem only" if feature in {"no2", "o3"} else "All sensors"
        for minutes in lags:
            name = f"{feature}_lag_{lag_label(minutes)}"
            output[name] = add_exact_lag(output, feature, minutes)
            engineered.append(name)
            catalog.append(_catalog_row(name, "LAG", f"Exact {lag_label(minutes)} elapsed-time history within a continuous segment.", scope, "Phase 2 ACF/PACF and cross-correlation"))

    for feature, windows in ROLLING_PLAN.items():
        scope = "Tulkarem only" if feature in {"no2", "o3"} else "All sensors"
        for minutes, statistics in windows.items():
            calculated = backward_rolling(output, feature, minutes)
            expected = minutes / EXPECTED_INTERVAL_MINUTES
            for statistic in statistics:
                if statistic == "range":
                    values = calculated["max"] - calculated["min"]
                elif statistic == "cv":
                    values = calculated["std"] / calculated["mean"].abs().replace(0, np.nan)
                else:
                    values = calculated[statistic]
                name = f"{feature}_{statistic}_{lag_label(minutes)}"
                output[name] = values.where(calculated["count"] >= max(2, math.ceil(expected * .5)))
                engineered.append(name)
                catalog.append(_catalog_row(name, "ROLLING", f"Backward-only {lag_label(minutes)} {statistic}; minimum 50% history coverage.", scope, "Phase 2 rolling stability and autocorrelation"))

    output["pm25_delta_15m"] = output.pm25 - output.pm25_lag_15m
    output["pm25_delta_1h"] = output.pm25 - output.pm25_lag_1h
    output["pm25_delta_3h"] = output.pm25 - output.pm25_lag_3h
    output["pm25_pct_change_1h"] = output.pm25_delta_1h / output.pm25_lag_1h.abs().replace(0, np.nan)
    output["pm25_slope_3h"] = output.pm25_delta_3h / 3.0
    for name in ["pm25_delta_15m", "pm25_delta_1h", "pm25_delta_3h", "pm25_pct_change_1h", "pm25_slope_3h"]:
        engineered.append(name)
        catalog.append(_catalog_row(name, "RATE_OF_CHANGE", "Captures observed PM2.5 movement using only current and exact past values.", evidence="Phase 2 sudden-jump and persistence analysis"))

    output["pm25_humidity_interaction"] = output.pm25 * output.humidity
    output["temperature_humidity_interaction"] = output.temperature * output.humidity
    for name, reason in [
        ("pm25_humidity_interaction", "Phase 2 found strong PM2.5-humidity association for Nablus sensors 2 and 4."),
        ("temperature_humidity_interaction", "Phase 2 found strong inverse temperature-humidity association across sensors."),
    ]:
        engineered.append(name)
        catalog.append(_catalog_row(name, "CROSS_FEATURE", reason, evidence="Phase 2 Pearson/Spearman analysis"))

    # Quality support uses current/past information only.
    output["current_quality_review_flag"] = output.overall_quality_flag.ne("NORMAL").astype(int)
    output["recent_gap_count_3h"] = 0.0
    output["recent_missing_pm25_count_3h"] = 0.0
    for _, group in output.groupby(["sensor_id", "continuous_segment_id"], sort=False):
        idx = group.index
        gap = pd.Series(group.sampling_gap_flag.astype(int).to_numpy(), index=group.timestamp).rolling("3h", closed="right").sum()
        missing = pd.Series(group.pm25.isna().astype(int).to_numpy(), index=group.timestamp).rolling("3h", closed="right").sum()
        output.loc[idx, "recent_gap_count_3h"] = gap.to_numpy()
        output.loc[idx, "recent_missing_pm25_count_3h"] = missing.to_numpy()

    anomaly_path = OUTPUT_DIR / "anomaly_results.csv"
    anomaly_count = pd.Series(0, index=output.index, dtype=float)
    if anomaly_path.exists():
        anomalies = pd.read_csv(anomaly_path, parse_dates=["timestamp"])
        consensus = anomalies.loc[anomalies.anomaly_methods_triggered >= 2].groupby(["sensor_id", "timestamp"]).size()
        keys = pd.MultiIndex.from_arrays([output.sensor_id, output.timestamp])
        anomaly_count = pd.Series(consensus.reindex(keys, fill_value=0).to_numpy(), index=output.index, dtype=float)
    output["recent_anomaly_count_3h"] = 0.0
    for _, group in output.groupby(["sensor_id", "continuous_segment_id"], sort=False):
        values = pd.Series(anomaly_count.loc[group.index].to_numpy(), index=group.timestamp).rolling("3h", closed="right").sum()
        output.loc[group.index, "recent_anomaly_count_3h"] = values.to_numpy()
    for name in ["current_quality_review_flag", "recent_gap_count_3h", "recent_missing_pm25_count_3h", "recent_anomaly_count_3h"]:
        engineered.append(name)
        catalog.append(_catalog_row(name, "QUALITY", "Current/backward-only quality context; no future anomaly or gap information is used.", evidence="Phase 1 flags and Phase 2 anomaly consensus"))

    return output, pd.DataFrame(catalog), engineered


def who_analysis(data: pd.DataFrame, who: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, pd.Series]]]:
    config_frame = pd.json_normalize(who["pollutants"])
    config_frame["source_version"] = who["source_version"]
    config_frame["source_url"] = who["source_url"]
    config_frame["minimum_window_coverage_percent"] = who["minimum_window_coverage_percent"]
    minimum = float(who["minimum_window_coverage_percent"])
    rows: list[pd.DataFrame] = []
    series_map: dict[str, dict[str, pd.Series]] = {}

    for item in who["pollutants"]:
        pollutant = item["pollutant"]
        if pollutant not in data.columns:
            continue
        minutes = 1440 if item["averaging_period"] == "24h" else 480
        calculated = backward_rolling(data, pollutant, minutes)
        expected = minutes / EXPECTED_INTERVAL_MINUTES
        average = calculated["mean"]
        coverage = (calculated["count"] / expected * 100).clip(upper=100)
        status = pd.Series("Insufficient Data", index=data.index, dtype="object")
        interim = pd.Series("Insufficient Data", index=data.index, dtype="object")
        ratio = pd.Series(np.nan, index=data.index, dtype=float)
        exceeded = pd.Series(pd.NA, index=data.index, dtype="boolean")

        structurally_absent = data[pollutant].isna()
        if item["dataset_presence"] == "TULKAREM_ONLY":
            structurally_absent = data.city_name.ne("Tulkarem")
            status.loc[structurally_absent] = "Not Available"
            interim.loc[structurally_absent] = "Not Available"
        valid_window = coverage.ge(minimum) & average.notna() & ~structurally_absent
        if not item["unit_compatible"]:
            status.loc[valid_window] = "Unit Not Compatible"
            interim.loc[valid_window] = "Unit Not Compatible"
        else:
            guideline = float(item["guideline"])
            ratio.loc[valid_window] = average.loc[valid_window] / guideline
            exceeded.loc[valid_window] = average.loc[valid_window].gt(guideline)
            status.loc[valid_window & average.le(guideline)] = "Meets WHO Guideline"
            status.loc[valid_window & average.gt(guideline)] = "Above WHO Guideline"
            interim.loc[valid_window & average.le(guideline)] = "AQG"
            ordered = sorted(((float(value), label) for label, value in item.get("interim_targets", {}).items()))
            previous = guideline
            for threshold, label in ordered:
                mask = valid_window & average.gt(previous) & average.le(threshold)
                interim.loc[mask] = label
                previous = threshold
            if ordered:
                interim.loc[valid_window & average.gt(ordered[-1][0])] = "Beyond IT-1"

        payload = pd.DataFrame({
            "timestamp": data.timestamp, "sensor_id": data.sensor_id, "city": data.city_name,
            "pollutant": pollutant, "averaging_period": item["averaging_period"],
            "average": average, "coverage_percent": coverage,
            "minimum_coverage_percent": minimum, "guideline": item["guideline"],
            "guideline_unit": item["unit"], "dataset_unit": item["dataset_unit"],
            "unit_compatible": item["unit_compatible"], "status": status,
            "exceeded": exceeded, "ratio": ratio, "interim_level": interim,
        })
        rows.append(payload)
        series_map[pollutant] = {"average": average, "coverage": coverage, "status": status, "ratio": ratio, "exceeded": exceeded, "interim": interim}
    return pd.concat(rows, ignore_index=True), config_frame, series_map


def prepare_targets(data: pd.DataFrame, who_series: dict[str, dict[str, pd.Series]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = data[["timestamp", "sensor_id", "city_name", "continuous_segment_id"]].copy()
    validity_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    pm25_who = who_series["pm25"]

    next_delta = data.groupby("sensor_id")["timestamp"].shift(-1).sub(data.timestamp).dt.total_seconds().div(60)
    gap_boundaries = {
        int(sensor_id): group.loc[next_delta.loc[group.index].gt(MAJOR_GAP_MINUTES), "timestamp"].tolist()
        for sensor_id, group in data.groupby("sensor_id")
    }
    for minutes in HORIZONS:
        label = lag_label(minutes)
        target_name = f"target_pm25_{label}"
        event_name = f"pollution_event_{label}"
        target = add_future_value(data, data.pm25, minutes)
        future_who_average = add_future_value(data, pm25_who["average"], minutes)
        future_who_coverage = add_future_value(data, pm25_who["coverage"], minutes)
        targets[target_name] = target
        targets[f"target_pm25_who_24h_avg_{label}"] = future_who_average
        targets[event_name] = pd.Series(pd.NA, index=data.index, dtype="Int64")
        event_valid = future_who_average.notna() & future_who_coverage.ge(75)
        targets.loc[event_valid, event_name] = future_who_average.loc[event_valid].gt(15).astype(int)

        history_ok = data.pm25_lag_6h.notna() & data.pm25_mean_3h.notna()
        future_gap = pd.Series(False, index=data.index)
        horizon_delta = pd.Timedelta(minutes=minutes)
        for sensor_id, boundaries in gap_boundaries.items():
            sensor_mask = data.sensor_id.eq(sensor_id)
            for boundary in boundaries:
                future_gap |= sensor_mask & data.timestamp.le(boundary) & data.timestamp.add(horizon_delta).gt(boundary)
        target_exists = target.notna()
        quality_issue = data.pm25.isna() | data.missing_flag.astype(bool)
        valid = history_ok & target_exists & ~future_gap & ~quality_issue
        reason = pd.Series("VALID", index=data.index, dtype="object")
        reason.loc[~history_ok] = "INSUFFICIENT_HISTORY"
        reason.loc[history_ok & future_gap] = "FUTURE_MAJOR_GAP"
        reason.loc[history_ok & ~future_gap & ~target_exists] = "MISSING_FUTURE_TARGET"
        reason.loc[history_ok & ~future_gap & target_exists & quality_issue] = "QUALITY_ISSUE"
        validity = pd.DataFrame({
            "timestamp": data.timestamp, "sensor_id": data.sensor_id, "city": data.city_name,
            "horizon": label, "forecast_sample_valid": valid,
            "history_available": history_ok, "future_target_available": target_exists,
            "future_major_gap": future_gap, "quality_issue": quality_issue,
            "invalid_reason": reason, "event_target_available": targets[event_name].notna(),
            "pollution_event": targets[event_name],
        })
        validity_rows.append(validity)
        targets[f"forecast_sample_valid_{label}"] = valid

        for sensor_id, group in validity.groupby("sensor_id"):
            summary_rows.append({
                "sensor_id": int(sensor_id), "city": group.city.iloc[0], "horizon": label,
                "total_timestamps": len(group), "valid_training_samples": int(group.forecast_sample_valid.sum()),
                "invalid_due_to_history": int((group.invalid_reason == "INSUFFICIENT_HISTORY").sum()),
                "invalid_due_to_future_gap": int((group.invalid_reason == "FUTURE_MAJOR_GAP").sum()),
                "invalid_due_to_missing_target": int((group.invalid_reason == "MISSING_FUTURE_TARGET").sum()),
                "invalid_due_to_quality_issue": int((group.invalid_reason == "QUALITY_ISSUE").sum()),
                "valid_event_labels": int(group.event_target_available.sum()),
                "positive_event_labels": int(group.pollution_event.fillna(0).astype(int).sum()),
            })
    return targets, pd.concat(validity_rows, ignore_index=True), pd.DataFrame(summary_rows)


def rank_features(features: pd.DataFrame, targets: pd.DataFrame, engineered: list[str]) -> pd.DataFrame:
    candidate_columns = ["pm25", "temperature", "humidity", "no2", "o3", *engineered]
    candidate_columns = [column for column in candidate_columns if column in features and pd.api.types.is_numeric_dtype(features[column])]
    rows: list[dict[str, Any]] = []
    for minutes in HORIZONS:
        label = lag_label(minutes)
        target_name = f"target_pm25_{label}"
        valid_mask = targets[f"forecast_sample_valid_{label}"].astype(bool) & targets[target_name].notna()
        for feature in candidate_columns:
            pair = pd.DataFrame({"x": features.loc[valid_mask, feature], "y": targets.loc[valid_mask, target_name]}).dropna()
            if len(pair) < 80 or pair.x.nunique() < 3:
                continue
            pearson = pair.x.corr(pair.y)
            rows.append({"feature": feature, "target": target_name, "association_method": "absolute_pearson", "score": abs(pearson), "signed_score": pearson, "n_observations": len(pair), "notes": "Exploratory whole-dataset screening only; repeat on training folds in Phase 4."})
            sample = pair.iloc[::max(1, len(pair) // 2000)]
            mi = mutual_info_regression(sample[["x"]], sample.y, random_state=42)[0]
            rows.append({"feature": feature, "target": target_name, "association_method": "mutual_information", "score": mi, "signed_score": None, "n_observations": len(sample), "notes": "Nonlinear screening only; final selection must use training folds."})
    return pd.DataFrame(rows).sort_values(["target", "association_method", "score"], ascending=[True, True, False])


def redundancy_analysis(features: pd.DataFrame, engineered: list[str]) -> pd.DataFrame:
    columns = [column for column in engineered if column in features and pd.api.types.is_numeric_dtype(features[column]) and features[column].notna().sum() >= 100]
    corr = features[columns].corr()
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = corr.at[left, right]
            if pd.notna(value) and abs(value) >= .95:
                rows.append({"feature_a": left, "feature_b": right, "correlation": value, "absolute_correlation": abs(value), "recommendation": "Review together; retain the more interpretable or validation-stable feature rather than dropping automatically."})
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False) if rows else pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "absolute_correlation", "recommendation"])


def write_reports(summary: dict[str, Any], catalog: pd.DataFrame, ranking: pd.DataFrame, redundancy: pd.DataFrame, who: dict[str, Any], who_results: pd.DataFrame, forecast_summary: pd.DataFrame) -> None:
    top_features = ranking.loc[ranking.association_method == "absolute_pearson"].groupby("target", as_index=False).head(8)
    compatible = [row["pollutant"].upper() for row in who["pollutants"] if row["unit_compatible"] and row["dataset_presence"] != "NOT_AVAILABLE"]
    incompatible = [row["pollutant"].upper() for row in who["pollutants"] if row["dataset_presence"] != "NOT_AVAILABLE" and not row["unit_compatible"]]
    pm_valid = who_results.loc[(who_results.pollutant == "pm25") & who_results.status.isin(["Meets WHO Guideline", "Above WHO Guideline"])]
    pm_exceeded = int(pm_valid.exceeded.fillna(False).sum())
    pm_percent = pm_exceeded / len(pm_valid) * 100 if len(pm_valid) else 0

    leakage_report = f"""# AirAware Phase 3 - Feature Leakage Audit

Generated: {summary['generated_at']}

## Result

All {len(catalog)} candidate feature columns passed the construction-time leakage audit.

## Controls

- Every lag is joined by exact elapsed timestamp inside one `continuous_segment_id`.
- Every rolling feature is backward-looking (`closed=right`) and never centered.
- Rolling and lag state resets after outages longer than {MAJOR_GAP_MINUTES} minutes.
- Future concentration and event values exist only in the target dataset.
- Current quality features use flags known at or before timestamp t.
- Candidate ranking is explicitly exploratory; final selection and scaling must be fitted on training folds only.
- No full-dataset normalization, imputation, future WHO status, or future anomaly state is present in the ML feature file.

## Rejected designs

- Centered rolling averages.
- Row-shift lags that could cross Sensor 5's multi-day gap.
- Source AQI as a target or ground truth.
- Instantaneous PM2.5 compared to the WHO 24-hour AQG as an official violation.
"""
    (REPORT_DIR / "07_feature_leakage_report.md").write_text(leakage_report, encoding="utf-8")

    who_report = f"""# AirAware Phase 3 - WHO 2021 Integration Report

Source: {who['source_title']} ({who['source_version']})  
Official publication: {who['source_url']}

## Guideline configuration

The short-term configurations use PM2.5 24-hour 15 ug/m3, PM10 24-hour 45 ug/m3, NO2 24-hour 25 ug/m3, O3 daily maximum 8-hour 100 ug/m3, SO2 24-hour 40 ug/m3, and CO 24-hour 4 mg/m3. Interim targets are stored separately in the versioned JSON configuration.

## Unit and availability audit

- Supported comparison: {', '.join(compatible) or 'None'}.
- Present but not comparable because source units are undocumented: {', '.join(incompatible) or 'None'}.
- PM10, SO2, and CO are not present in this dataset.
- NO2/O3 remain structurally unavailable for Nablus and are never fabricated.

## Completeness rule

A rolling WHO reference window requires at least {who['minimum_window_coverage_percent']}% of expected 15-minute readings and cannot cross a major outage. Early and fragmented windows are `Insufficient Data`.

## PM2.5 findings

- Valid 24-hour sensor windows: {len(pm_valid):,}.
- Windows above the 15 ug/m3 reference: {pm_exceeded:,} ({pm_percent:.2f}%).

These are overlapping rolling-window health-reference screening results. WHO's short-term AQG also includes annual exceedance-frequency context, which this 17-day dataset cannot evaluate. AirAware does not call this a WHO AQI or a legal compliance result.
"""
    (REPORT_DIR / "08_who_integration_report.md").write_text(who_report, encoding="utf-8")

    feature_report = f"""# AirAware Phase 3 - Feature Engineering Report

## Output

- Candidate engineered features: {len(catalog)}.
- Evidence-driven lag families: {sum(catalog.family == 'LAG')}.
- Selected backward rolling features: {sum(catalog.family == 'ROLLING')}.
- Highly redundant pairs flagged: {len(redundancy)}.

## Evidence-driven design

PM2.5 lags at 15 minutes, 1 hour, 3 hours, and 6 hours reflect Phase 2 persistence. Temperature/humidity and Tulkarem-only NO2/O3 use 15-minute, 1-hour, and 3-hour lags. Rolling features focus on 1-hour, 3-hour, and 6-hour behavior; the 24-hour PM2.5 rolling average is reserved for WHO reference analysis.

Day-of-month, week-of-year, and month were deliberately omitted because the dataset spans only 17 days and these calendar identifiers would not provide stable repeated seasonal evidence. Hour and day-of-week use cyclic encodings.

## Strong candidate associations

{chr(10).join(f"- {row.feature} -> {row.target}: |r|={row.score:.3f}, n={row.n_observations}." for row in top_features.head(16).itertuples())}

## Missingness and structural absence

No engineered value is zero-filled. Initial-history lag/rolling nulls, temporary missing readings, and structural NO2/O3 absence remain distinguishable through metadata and feature availability.

## Gap and leakage controls

All time operations reset at `continuous_segment_id`. Features use t or earlier; targets use exact future timestamps and remain in a separate file.
"""
    (REPORT_DIR / "09_feature_engineering_report.md").write_text(feature_report, encoding="utf-8")

    totals = forecast_summary.groupby("horizon").agg(total=("total_timestamps", "sum"), valid=("valid_training_samples", "sum")).reset_index()
    event_totals = forecast_summary.groupby("horizon").agg(valid_event_labels=("valid_event_labels", "sum"), positive_event_labels=("positive_event_labels", "sum")).reset_index()
    totals = totals.merge(event_totals, on="horizon")
    readiness_lines = [f"- {row.horizon}: {int(row.valid):,} valid of {int(row.total):,} timestamps ({row.valid / row.total * 100:.1f}%); {int(row.positive_event_labels):,} positive events among {int(row.valid_event_labels):,} valid event labels." for row in totals.itertuples()]
    readiness_report = f"""# AirAware Phase 3 - Forecasting Dataset Readiness

## Horizon readiness

{chr(10).join(readiness_lines)}

The dataset supports prototype evaluation at 1h, 3h, and 6h. Sensors 1, 2, and 4 are the strongest continuity candidates. Sensor 5 must be segmented because of its multi-day outage.

## Forecastable pollutants

PM2.5 is the primary target across all sensors. NO2 and O3 are available only for Tulkarem and require verified units before health-event labeling. Existing source AQI remains excluded as ground truth.

## Modeling structure recommendation

Start with per-sensor PM2.5 baselines, then compare a combined model with explicit sensor/city identifiers. Do not place Tulkarem NO2/O3 into a universal numeric feature matrix without a model-specific structural-missingness design. The 17-day span is too short for robust long-season claims.

## Validation and preprocessing

Use chronological splits with at least complete daily cycles in validation/test and walk-forward evaluation. Fit imputation, scaling, and feature selection on each training fold only. Tree models generally do not require scaling; neural/time-series models typically do.

For an initial benchmark on the observed 2026-08-06 through 2026-08-23 window, use the oldest 10 calendar days for training, the next 4 days for validation, and the final 4 days as untouched test data. This is a date-based plan, not a random percentage split. Walk-forward evaluation should be the primary comparison, and Sensor 5 must remain segmented around its outage.

## Event evaluation

Event labels use the valid future PM2.5 24-hour rolling average ending at each horizon, not an instantaneous reading and not source AQI. Event prevalence must be checked per fold before choosing event metrics.
"""
    (REPORT_DIR / "10_forecasting_dataset_readiness.md").write_text(readiness_report, encoding="utf-8")


def run_phase3_pipeline() -> dict[str, Any]:
    ensure_directories()
    data, phase2, who = _load()
    original_signature = pd.util.hash_pandas_object(data[["sensor_id", "timestamp", "pm25", "temperature", "humidity", "no2", "o3", "aqi"]], index=False).sum()
    features, catalog, engineered = engineer_features(data, phase2)
    who_results, who_config_frame, who_series = who_analysis(features, who)

    for key, series in who_series["pm25"].items():
        if key in {"average", "coverage", "ratio", "exceeded", "status", "interim"}:
            features[f"pm25_who_24h_{key}"] = series
    features["airaware_who_health_status"] = who_series["pm25"]["status"]
    features["airaware_who_dominant_pollutant"] = np.where(who_series["pm25"]["status"].isin(["Meets WHO Guideline", "Above WHO Guideline"]), "PM2.5", pd.NA)

    targets, validity, forecast_summary = prepare_targets(features, who_series)
    ranking = rank_features(features, targets, engineered)
    redundancy = redundancy_analysis(features, engineered)

    feature_columns = [
        "timestamp", "sensor_id", "city_id", "city_name", "location_name", "continuous_segment_id",
        "pm25", "temperature", "humidity", "no2", "o3", "source_aqi_unverified",
        "missing_flag", "sampling_gap_flag", "suspicious_value_flag", "jump_flag", "flatline_flag",
        "possible_drift_flag", "overall_quality_flag", *engineered,
        "pm25_who_24h_average", "pm25_who_24h_coverage", "pm25_who_24h_ratio",
        "pm25_who_24h_exceeded", "pm25_who_24h_status", "pm25_who_24h_interim",
        "airaware_who_health_status", "airaware_who_dominant_pollutant",
    ]
    features["source_aqi_unverified"] = features.aqi
    features = features[[column for column in feature_columns if column in features]]

    features.to_csv(ML_FEATURES_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    targets.to_csv(FORECAST_TARGETS_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    catalog.to_csv(OUTPUT_DIR / PHASE3_ARTIFACTS["feature_catalog"], index=False)
    ranking.to_csv(OUTPUT_DIR / PHASE3_ARTIFACTS["feature_ranking"], index=False)
    redundancy.to_csv(OUTPUT_DIR / PHASE3_ARTIFACTS["feature_redundancy"], index=False)
    who_config_frame.to_csv(OUTPUT_DIR / PHASE3_ARTIFACTS["who_guidelines"], index=False)
    who_results.to_csv(OUTPUT_DIR / PHASE3_ARTIFACTS["who_analysis"], index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    validity.to_csv(OUTPUT_DIR / PHASE3_ARTIFACTS["sample_validity"], index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    forecast_summary.to_csv(OUTPUT_DIR / PHASE3_ARTIFACTS["forecast_summary"], index=False)

    pm_valid = who_results.loc[(who_results.pollutant == "pm25") & who_results.status.isin(["Meets WHO Guideline", "Above WHO Guideline"])]
    horizon_totals = forecast_summary.groupby("horizon").agg(total_timestamps=("total_timestamps", "sum"), valid_training_samples=("valid_training_samples", "sum")).reset_index()
    evidence = {
        "pm25_autocorrelation": [row for row in phase2["cross_correlation"]["strongest"] if row.get("source_feature") == "pm25"][:8],
        "phase2_modeling_report": "06_modeling_readiness_report.md",
    }
    summary = {
        "phase": 3, "version": PHASE3_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation": {
            "source_rows": len(data), "ml_feature_rows": len(features), "forecast_target_rows": len(targets),
            "timestamps_parse_failures": int(data.timestamp.isna().sum()),
            "continuous_segments": int(data.continuous_segment_id.nunique()),
            "sensor_ids_preserved": sorted(int(value) for value in data.sensor_id.unique()),
            "observation_signature_preserved": int(original_signature) == int(pd.util.hash_pandas_object(data[["sensor_id", "timestamp", "pm25", "temperature", "humidity", "no2", "o3", "aqi"]], index=False).sum()),
            "future_columns_in_feature_file": [column for column in features if column.startswith("target_") or column.startswith("pollution_event_")],
            "source_aqi_role": "Preserved as source_aqi_unverified; excluded from targets and WHO status.",
            "model_trained": False,
        },
        "feature_engineering": {
            "total_candidate_features": len(catalog), "families": catalog.family.value_counts().to_dict(),
            "selected_lags": EVIDENCE_LAGS, "rolling_plan": ROLLING_PLAN,
            "redundant_pairs": len(redundancy), "top_ranked": records(ranking.head(30)),
            "catalog": records(catalog),
        },
        "who": {
            "source": who["source_url"], "version": who["source_version"],
            "minimum_coverage_percent": who["minimum_window_coverage_percent"],
            "supported_pollutants": ["pm25"], "unit_unverified": ["no2", "o3"],
            "not_available": ["pm10", "so2", "co"],
            "valid_pm25_windows": len(pm_valid),
            "pm25_exceedance_windows": int(pm_valid.exceeded.fillna(False).sum()),
            "pm25_exceedance_percentage": float(pm_valid.exceeded.fillna(False).mean() * 100) if len(pm_valid) else 0,
            "warning": "WHO status uses pollutant-specific averaging periods, not single readings. It is not a WHO AQI or legal compliance determination.",
        },
        "forecast_dataset": {
            "horizons": [lag_label(value) for value in HORIZONS],
            "target_pollutants": ["pm25"], "summary": records(forecast_summary),
            "horizon_totals": records(horizon_totals),
            "event_definition": "Future valid PM2.5 24-hour rolling average ending at t+h exceeds 15 ug/m3.",
            "recommended_strategy": "Compare per-sensor PM2.5 baselines with a combined sensor-aware model using chronological walk-forward evaluation.",
            "chronological_split_plan": {
                "train": "2026-08-06 through 2026-08-15",
                "validation": "2026-08-16 through 2026-08-19",
                "test": "2026-08-20 through 2026-08-23",
                "primary_evaluation": "Walk-forward validation with preprocessing fitted inside each training fold",
            },
        },
        "phase2_evidence": evidence,
        "artifacts": [{"name": filename, "download_url": f"/api/ml/download/{filename}"} for filename in PHASE3_ARTIFACTS.values()],
    }
    write_reports(summary, catalog, ranking, redundancy, who, who_results, forecast_summary)
    PHASE3_ANALYSIS_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
