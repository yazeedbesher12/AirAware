from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = ("1h", "3h", "6h")
TRAIN_END = pd.Timestamp("2026-08-16T00:00:00Z")
VALID_END = pd.Timestamp("2026-08-20T00:00:00Z")
TEST_END = pd.Timestamp("2026-08-24T00:00:00Z")
OOF_SPLITS = (
    (pd.Timestamp("2026-08-16T00:00:00Z"), pd.Timestamp("2026-08-18T00:00:00Z")),
    (pd.Timestamp("2026-08-18T00:00:00Z"), pd.Timestamp("2026-08-20T00:00:00Z")),
)


@dataclass
class Phase4Dataset:
    frame: pd.DataFrame
    features: list[str]

    def valid(self, horizon: str) -> pd.DataFrame:
        flag = f"forecast_sample_valid_{horizon}"
        target = f"target_pm25_{horizon}"
        health = f"target_pm25_who_24h_avg_{horizon}"
        subset = self.frame[
            self.frame[flag].fillna(False).astype(bool)
            & self.frame[target].notna()
            & self.frame[health].notna()
        ].copy()
        return subset.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)

    def splits(self, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = self.valid(horizon)
        return (
            data[data.timestamp < TRAIN_END].copy(),
            data[(data.timestamp >= TRAIN_END) & (data.timestamp < VALID_END)].copy(),
            data[(data.timestamp >= VALID_END) & (data.timestamp < TEST_END)].copy(),
        )

    def oof_folds(self, horizon: str) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
        data = self.valid(horizon)
        folds = []
        for index, (start, end) in enumerate(OOF_SPLITS, 1):
            train = data[data.timestamp < start].copy()
            valid = data[(data.timestamp >= start) & (data.timestamp < end)].copy()
            if len(train) and len(valid):
                folds.append((f"fold_{index}", train, valid))
        return folds


def load_phase4_dataset(outputs_dir: Path) -> Phase4Dataset:
    features = pd.read_csv(outputs_dir / "AirAware_ML_Features.csv", low_memory=False)
    targets = pd.read_csv(outputs_dir / "AirAware_Forecast_Targets.csv", low_memory=False)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True, errors="raise")
    targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True, errors="raise")
    frame = features.merge(
        targets,
        on=["timestamp", "sensor_id", "city_name", "continuous_segment_id"],
        how="inner",
        validate="one_to_one",
    )
    excluded_prefixes = ("target_", "pollution_event_", "forecast_sample_valid_")
    excluded = {
        "timestamp", "city_id", "location_name", "city_name", "continuous_segment_id",
        "source_aqi_unverified", "aqi", "overall_quality_flag", "pm25_who_24h_status",
        "pm25_who_24h_interim", "airaware_who_health_status", "airaware_who_dominant_pollutant",
    }
    feature_columns = [
        column for column in frame.columns
        if column not in excluded
        and not column.startswith(excluded_prefixes)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    # Sensor identity is a static known input. It is intentionally retained as numeric metadata.
    if "sensor_id" not in feature_columns:
        feature_columns.insert(0, "sensor_id")
    frame = frame.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame), dtype=int)
    return Phase4Dataset(frame=frame, features=feature_columns)


def filter_sensor(frame: pd.DataFrame, sensor_id: int | None) -> pd.DataFrame:
    return frame if sensor_id is None else frame[frame.sensor_id == sensor_id].copy()
