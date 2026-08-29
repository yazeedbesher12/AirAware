from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.pipeline import run_pipeline
from app.config import ANALYSIS_JSON, CLEANED_CSV, ENVIRONMENTAL_FEATURES, FEATURE_UNITS


def _clean(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _clean(value) for key, value in row.items()} for row in frame.to_dict("records")]


class DataStore:
    def __init__(self) -> None:
        if not ANALYSIS_JSON.exists() or not CLEANED_CSV.exists():
            run_pipeline()
        self.analysis = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
        self.data = pd.read_csv(CLEANED_CSV, parse_dates=["timestamp"])
        self.suspicious = pd.read_csv(ANALYSIS_JSON.parent / "suspicious_observations.csv", parse_dates=["timestamp"])

    def filtered(
        self,
        city: str | None = None,
        sensor_id: int | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        quality_flag: str | None = None,
    ) -> pd.DataFrame:
        frame = self.data
        if city:
            frame = frame.loc[frame["city_name"].str.casefold() == city.casefold()]
        if sensor_id is not None:
            frame = frame.loc[frame["sensor_id"] == sensor_id]
        if start_time:
            frame = frame.loc[frame["timestamp"] >= pd.to_datetime(start_time, utc=True)]
        if end_time:
            frame = frame.loc[frame["timestamp"] <= pd.to_datetime(end_time, utc=True)]
        if quality_flag:
            frame = frame.loc[frame["overall_quality_flag"].str.casefold() == quality_flag.casefold()]
        return frame

    def feature_available(self, frame: pd.DataFrame, feature: str) -> bool:
        return feature in ENVIRONMENTAL_FEATURES and feature in frame and bool(frame[feature].notna().any())

    def timeseries(
        self,
        feature: str,
        city: str | None,
        sensor_id: int | None,
        start_time: str | None,
        end_time: str | None,
        rolling_window: str | None,
        max_points: int,
    ) -> dict[str, Any]:
        frame = self.filtered(city, sensor_id, start_time, end_time).sort_values(["sensor_id", "timestamp"])
        if feature not in ENVIRONMENTAL_FEATURES:
            raise ValueError(f"Unknown feature: {feature}")
        if not self.feature_available(frame, feature):
            return {"feature": feature, "unit": FEATURE_UNITS[feature], "available": False, "message": f"{feature.upper()} is not available for the selected sensors.", "series": []}
        output = []
        for sid, group in frame.groupby("sensor_id", sort=True):
            selected = group[["timestamp", feature, "overall_quality_flag", "jump_flag", "sampling_gap_flag"]].copy()
            if rolling_window and rolling_window != "none":
                indexed = selected.set_index("timestamp")
                indexed["rolling_mean"] = indexed[feature].rolling(rolling_window, min_periods=1).mean()
                selected = indexed.reset_index()
            else:
                selected["rolling_mean"] = np.nan
            stride = max(1, int(np.ceil(len(selected) / max_points)))
            selected = selected.iloc[::stride]
            output.append(
                {
                    "sensor_id": int(sid), "city": group["city_name"].iloc[0], "location": group["location_name"].iloc[0],
                    "points": records(selected.rename(columns={feature: "value"})),
                }
            )
        return {"feature": feature, "unit": FEATURE_UNITS[feature], "available": True, "series": output}

    def distribution(self, feature: str, city: str | None, sensor_id: int | None) -> dict[str, Any]:
        frame = self.filtered(city, sensor_id)
        if feature not in ENVIRONMENTAL_FEATURES or not self.feature_available(frame, feature):
            return {"feature": feature, "unit": FEATURE_UNITS.get(feature, ""), "available": False, "series": []}
        series = []
        for sid, group in frame.groupby("sensor_id", sort=True):
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
            counts, edges = np.histogram(values, bins=min(40, max(10, int(np.sqrt(len(values))))))
            sample = values.iloc[:: max(1, int(np.ceil(len(values) / 1000)))].tolist()
            series.append(
                {
                    "sensor_id": int(sid), "city": group["city_name"].iloc[0], "location": group["location_name"].iloc[0],
                    "statistics": {
                        "count": int(len(values)), "mean": _clean(values.mean()), "median": _clean(values.median()),
                        "std": _clean(values.std()), "min": _clean(values.min()), "max": _clean(values.max()),
                        "p95": _clean(values.quantile(.95)), "p99": _clean(values.quantile(.99)),
                    },
                    "histogram": {"counts": counts.tolist(), "bin_edges": edges.tolist()}, "sample": sample,
                }
            )
        return {"feature": feature, "unit": FEATURE_UNITS[feature], "available": True, "series": series}

    def scatter(self, x_feature: str, y_feature: str, city: str | None, sensor_id: int | None) -> dict[str, Any]:
        if x_feature not in ENVIRONMENTAL_FEATURES or y_feature not in ENVIRONMENTAL_FEATURES:
            raise ValueError("Unknown scatter feature.")
        frame = self.filtered(city, sensor_id)
        output = []
        for sid, group in frame.groupby("sensor_id", sort=True):
            pair = group[["timestamp", x_feature, y_feature]].dropna()
            if pair.empty:
                continue
            pair = pair.iloc[:: max(1, int(np.ceil(len(pair) / 1200)))]
            output.append({
                "sensor_id": int(sid), "city": group["city_name"].iloc[0], "location": group["location_name"].iloc[0],
                "points": records(pair.rename(columns={x_feature: "x", y_feature: "y"})),
            })
        return {"x_feature": x_feature, "y_feature": y_feature, "available": bool(output), "series": output}

    def suspicious_rows(
        self, city: str | None, sensor_id: int | None, feature: str | None,
        flag: str | None, start_time: str | None, end_time: str | None, limit: int, offset: int,
    ) -> dict[str, Any]:
        frame = self.suspicious
        if city:
            frame = frame.loc[frame["city"].str.casefold() == city.casefold()]
        if sensor_id is not None:
            frame = frame.loc[frame["sensor_id"] == sensor_id]
        if feature:
            frame = frame.loc[frame["feature"] == feature]
        if flag:
            frame = frame.loc[frame["flag"].str.casefold() == flag.casefold()]
        if start_time:
            frame = frame.loc[frame["timestamp"] >= pd.to_datetime(start_time, utc=True)]
        if end_time:
            frame = frame.loc[frame["timestamp"] <= pd.to_datetime(end_time, utc=True)]
        total = len(frame)
        columns = ["timestamp", "city", "sensor_id", "feature", "value", "previous_value", "absolute_change", "relative_change", "flag", "reason"]
        return {"total": total, "offset": offset, "limit": limit, "items": records(frame.iloc[offset:offset + limit][columns])}


@lru_cache(maxsize=1)
def get_store() -> DataStore:
    return DataStore()
