"""Reusable Phase 3 feature generation for historical training and live inference.

Only readings at or before the requested timestamp are accepted. The module
calls the same Phase 3 feature functions used to create the training CSV, which
prevents a second, divergent implementation of lag and rolling logic.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.phase3_pipeline import build_segments, engineer_features, who_analysis
from app.config import WHO_2021_CONFIG


SAMPLING_INTERVAL_MINUTES = 15
MAJOR_GAP_MINUTES = 90
MAX_BUFFER_HOURS = 30
RAW_NUMERIC = ("pm25", "temperature", "humidity", "no2", "o3", "aqi")
FLAG_DEFAULTS = {
    "missing_flag": False, "sampling_gap_flag": False, "suspicious_value_flag": False,
    "jump_flag": False, "flatline_flag": False, "possible_drift_flag": False,
    "overall_quality_flag": "NORMAL",
}


def generate_realtime_features(history: pd.DataFrame, required_features: list[str] | None = None) -> pd.DataFrame:
    """Generate Phase 3 features from an ordered history ending at current t."""
    if history.empty: raise ValueError("At least one historical reading is required")
    data = history.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    for column in RAW_NUMERIC:
        if column not in data: data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column, default in FLAG_DEFAULTS.items():
        if column not in data: data[column] = default
    if "city_id" not in data: data["city_id"] = data.get("city_name", "UNKNOWN")
    if "city_name" not in data: data["city_name"] = "UNKNOWN"
    if "location_name" not in data: data["location_name"] = "UNKNOWN"
    data = build_segments(data.sort_values(["sensor_id", "timestamp"]))
    features, _, _ = engineer_features(data, {})
    who = json.loads(Path(WHO_2021_CONFIG).read_text(encoding="utf-8"))
    _, _, who_series = who_analysis(features, who)
    for key in ("average", "coverage", "ratio", "exceeded", "status", "interim"):
        features[f"pm25_who_24h_{key}"] = who_series["pm25"][key]
    features["source_aqi_unverified"] = features["aqi"]
    features["airaware_who_health_status"] = who_series["pm25"]["status"]
    features["airaware_who_dominant_pollutant"] = np.where(who_series["pm25"]["status"].isin(["Meets WHO Guideline", "Above WHO Guideline"]), "PM2.5", pd.NA)
    if required_features:
        missing = [column for column in required_features if column not in features]
        if missing: raise ValueError(f"Feature generator cannot produce: {missing}")
    return features


class MultiSensorHistoryBuffer:
    """Independent, gap-aware history buffers for live physical sensor streams."""
    def __init__(self, max_hours: int = MAX_BUFFER_HOURS):
        self.max_rows = int(max_hours * 60 / SAMPLING_INTERVAL_MINUTES) + 1
        self._rows: dict[int, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.max_rows))
        self._last_timestamp: dict[int, pd.Timestamp] = {}
        self._segment_number: dict[int, int] = defaultdict(int)

    def update(self, reading: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
        sensor = int(reading["sensor_id"]); timestamp = pd.to_datetime(reading["timestamp"], utc=True)
        previous = self._last_timestamp.get(sensor)
        reset = previous is None or (timestamp - previous).total_seconds() / 60 > MAJOR_GAP_MINUTES or timestamp <= previous
        if reset:
            self._rows[sensor].clear(); self._segment_number[sensor] += 1
        payload = dict(reading); payload["timestamp"] = timestamp
        payload["live_segment_id"] = f"{sensor}-live-{self._segment_number[sensor]}"
        self._rows[sensor].append(payload); self._last_timestamp[sensor] = timestamp
        return pd.DataFrame(list(self._rows[sensor])), reset

    def reading_count(self, sensor_id: int) -> int:
        return len(self._rows[int(sensor_id)])


def history_requirement(model: str) -> dict[str, Any]:
    requirements = {
        "persistence": ("1 valid reading", 1), "random_forest": ("6 hours", 25),
        "xgboost": ("6 hours", 25), "lightgbm": ("6 hours", 25), "prophet": ("current timestamp and sensor identity", 1),
        "lstm": ("approximately 9 hours", 36), "gru": ("approximately 9 hours", 36),
        "tft": ("30 hours for full feature parity (6-hour encoder plus 24-hour WHO feature lookback)", 121),
    }
    label, samples = requirements[model]
    return {"minimum_history_required": label, "minimum_samples": samples, "sampling_interval": "15 minutes"}
