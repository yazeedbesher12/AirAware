from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from app.config import (
    ENVIRONMENTAL_FEATURES, FEATURE_UNITS, OUTPUT_DIR, PHASE2_ANALYSIS_JSON,
    PHASE2_ARTIFACTS,
)
from app.services.data_store import get_store, records


class Phase2Store:
    def __init__(self) -> None:
        if not PHASE2_ANALYSIS_JSON.exists():
            from app.analysis.phase2_pipeline import run_phase2_pipeline
            run_phase2_pipeline()
        self.analysis = json.loads(PHASE2_ANALYSIS_JSON.read_text(encoding="utf-8"))
        self.frames: dict[str, pd.DataFrame] = {}
        for key, filename in PHASE2_ARTIFACTS.items():
            path = OUTPUT_DIR / filename
            parse_dates = [column for column in ["timestamp", "change_timestamp"] if column in pd.read_csv(path, nrows=0).columns]
            self.frames[key] = pd.read_csv(path, parse_dates=parse_dates)
        self.base = get_store()

    def _sensor_city_filter(self, frame: pd.DataFrame, city: str | None, sensor_id: int | None, feature: str | None) -> pd.DataFrame:
        output = frame
        if city and "city" in output:
            output = output.loc[output.city.str.casefold() == city.casefold()]
        if sensor_id is not None and "sensor_id" in output:
            output = output.loc[output.sensor_id == sensor_id]
        if feature and "feature" in output:
            output = output.loc[output.feature == feature]
        return output

    def statistics(self, city: str | None, sensor_id: int | None, feature: str | None) -> dict[str, Any]:
        frame = self.frames["statistics"]
        if sensor_id is not None:
            frame = frame.loc[(frame.scope_type == "sensor") & (frame.scope_value.astype(str) == str(sensor_id))]
        elif city:
            frame = frame.loc[(frame.scope_type == "city") & (frame.scope_value.str.casefold() == city.casefold())]
        else:
            frame = frame.loc[frame.scope_type == "global"]
        if feature:
            frame = frame.loc[frame.feature == feature]
        interpretations = [item for item in self.analysis["statistics"]["interpretations"] if (not feature or item["feature"] == feature) and (sensor_id is None or (item["scope_type"] == "sensor" and item["scope_value"] == str(sensor_id))) and (not city or sensor_id is not None or (item["scope_type"] == "city" and item["scope_value"].casefold() == city.casefold()))]
        return {"items": records(frame), "interpretations": interpretations[:30]}

    def distributions(self, city: str | None, sensor_id: int | None, feature: str) -> dict[str, Any]:
        if feature not in ENVIRONMENTAL_FEATURES:
            raise ValueError("Unknown feature.")
        frame = self.base.filtered(city, sensor_id)
        if not self.base.feature_available(frame, feature):
            return {"feature": feature, "unit": FEATURE_UNITS[feature], "available": False, "message": f"{feature.upper()} is not available for the selected sensors.", "series": []}
        output = []
        for sid, group in frame.groupby("sensor_id", sort=True):
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna().sort_values()
            if values.empty:
                continue
            stride = max(1, len(values) // 1200)
            sample = values.iloc[::stride].to_numpy()
            ecdf_stride = max(1, len(values) // 800)
            ecdf_x = values.iloc[::ecdf_stride].to_numpy()
            ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)
            kde_payload = {"x": [], "density": []}
            if values.nunique() > 2 and len(values) >= 20:
                grid = np.linspace(values.quantile(.005), values.quantile(.995), 120)
                density = gaussian_kde(values.to_numpy())(grid)
                kde_payload = {"x": grid.tolist(), "density": density.tolist()}
            output.append({"sensor_id": int(sid), "city": group.city_name.iloc[0], "location": group.location_name.iloc[0], "sample": sample.tolist(), "ecdf": {"x": ecdf_x.tolist(), "y": ecdf_y.tolist()}, "kde": kde_payload})
        return {"feature": feature, "unit": FEATURE_UNITS[feature], "available": bool(output), "series": output}

    def temporal(self, city: str | None, sensor_id: int | None, feature: str | None, pattern_type: str | None) -> dict[str, Any]:
        frame = self._sensor_city_filter(self.frames["temporal"], city, sensor_id, feature)
        if pattern_type:
            frame = frame.loc[frame.pattern_type == pattern_type]
        findings = [item for item in self.analysis["temporal"]["findings"] if (sensor_id is None or item["sensor_id"] == sensor_id) and (not feature or item["feature"] == feature)]
        return {"items": records(frame), "findings": findings[:30]}

    def rolling(self, city: str | None, sensor_id: int | None, feature: str, window: str, start_time: str | None, end_time: str | None) -> dict[str, Any]:
        if window not in {"1h", "3h", "6h", "12h", "24h"}:
            raise ValueError("Rolling window must be 1h, 3h, 6h, 12h, or 24h.")
        frame = self.base.filtered(city, sensor_id, start_time, end_time).sort_values(["sensor_id", "timestamp"])
        if not self.base.feature_available(frame, feature):
            return {"available": False, "feature": feature, "window": window, "series": []}
        output = []
        for sid, group in frame.groupby("sensor_id", sort=True):
            indexed = group.set_index("timestamp")[feature]
            roll = indexed.rolling(window, min_periods=2)
            result = pd.DataFrame({"timestamp": indexed.index, "raw": indexed.values, "mean": roll.mean(), "median": roll.median(), "std": roll.std(), "variance": roll.var(), "min": roll.min(), "max": roll.max()}).reset_index(drop=True)
            result["timestamp"] = indexed.index
            result = result.iloc[::max(1, len(result) // 1200)]
            output.append({"sensor_id": int(sid), "city": group.city_name.iloc[0], "points": records(result)})
        return {"available": bool(output), "feature": feature, "unit": FEATURE_UNITS[feature], "window": window, "series": output}

    def correlations(self, method: str, city: str | None, sensor_id: int | None, feature: str | None) -> dict[str, Any]:
        key = "spearman" if method.casefold() == "spearman" else "pearson"
        frame = self.frames[key]
        if sensor_id is not None:
            frame = frame.loc[(frame.scope_type == "sensor") & (frame.scope_value.astype(str) == str(sensor_id))]
        elif city:
            frame = frame.loc[(frame.scope_type == "city") & (frame.scope_value.str.casefold() == city.casefold())]
        else:
            frame = frame.loc[frame.scope_type == "global"]
        if feature:
            frame = frame.loc[(frame.feature_a == feature) | (frame.feature_b == feature)]
        features = sorted(set(frame.feature_a) | set(frame.feature_b)) if len(frame) else []
        lookup = {(row.feature_a, row.feature_b): row.correlation for row in frame.itertuples()}
        matrix = [[1.0 if left == right else lookup.get((left, right), lookup.get((right, left))) for right in features] for left in features]
        return {"method": key, "features": features, "matrix": matrix, "relationships": records(frame.sort_values("absolute_correlation", ascending=False)), "nonlinear_candidates": self.analysis["correlation"]["nonlinear_candidates"] if key == "spearman" else []}

    def filtered_frame(self, key: str, city: str | None, sensor_id: int | None, feature: str | None, limit: int = 500) -> list[dict[str, Any]]:
        frame = self._sensor_city_filter(self.frames[key], city, sensor_id, feature)
        return records(frame.head(limit))


@lru_cache(maxsize=1)
def get_phase2_store() -> Phase2Store:
    return Phase2Store()
