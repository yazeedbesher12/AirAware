from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from app.config import OUTPUT_DIR, PHASE3_ANALYSIS_JSON, PHASE3_ARTIFACTS
from app.services.data_store import records


class Phase3Store:
    def __init__(self) -> None:
        if not PHASE3_ANALYSIS_JSON.exists():
            from app.analysis.phase3_pipeline import run_phase3_pipeline
            run_phase3_pipeline()
        self.analysis = json.loads(PHASE3_ANALYSIS_JSON.read_text(encoding="utf-8"))
        self.frames: dict[str, pd.DataFrame] = {}
        for key, filename in PHASE3_ARTIFACTS.items():
            path = OUTPUT_DIR / filename
            header = pd.read_csv(path, nrows=0).columns
            parse_dates = [column for column in ["timestamp"] if column in header]
            self.frames[key] = pd.read_csv(path, parse_dates=parse_dates)

    @staticmethod
    def filter_frame(frame: pd.DataFrame, city: str | None = None, sensor_id: int | None = None, start_time: str | None = None, end_time: str | None = None) -> pd.DataFrame:
        output = frame
        city_column = "city" if "city" in output else "city_name" if "city_name" in output else None
        if city and city_column:
            output = output.loc[output[city_column].astype(str).str.casefold() == city.casefold()]
        if sensor_id is not None and "sensor_id" in output:
            output = output.loc[output.sensor_id == sensor_id]
        if start_time and "timestamp" in output:
            output = output.loc[output.timestamp >= pd.to_datetime(start_time, utc=True)]
        if end_time and "timestamp" in output:
            output = output.loc[output.timestamp <= pd.to_datetime(end_time, utc=True)]
        return output

    def feature_catalog(self, family: str | None = None) -> dict[str, Any]:
        frame = self.frames["feature_catalog"]
        if family:
            frame = frame.loc[frame.family.str.casefold() == family.casefold()]
        return {"total": len(frame), "items": records(frame), "families": self.analysis["feature_engineering"]["families"]}

    def ranking(self, target: str | None, method: str | None, limit: int) -> dict[str, Any]:
        frame = self.frames["feature_ranking"]
        if target:
            frame = frame.loc[frame.target == target]
        if method:
            frame = frame.loc[frame.association_method == method]
        frame = frame.sort_values("score", ascending=False)
        return {"total": len(frame), "items": records(frame.head(limit)), "warning": "Exploratory screening only; repeat feature selection inside Phase 4 training folds."}

    def sample_validity(self, city: str | None, sensor_id: int | None, horizon: str | None) -> dict[str, Any]:
        summary = self.frames["forecast_summary"]
        if city:
            summary = summary.loc[summary.city.str.casefold() == city.casefold()]
        if sensor_id is not None:
            summary = summary.loc[summary.sensor_id == sensor_id]
        if horizon:
            summary = summary.loc[summary.horizon == horizon]
        totals = summary.groupby("horizon", as_index=False).agg(
            total_timestamps=("total_timestamps", "sum"), valid_training_samples=("valid_training_samples", "sum"),
            invalid_due_to_history=("invalid_due_to_history", "sum"), invalid_due_to_future_gap=("invalid_due_to_future_gap", "sum"),
            invalid_due_to_missing_target=("invalid_due_to_missing_target", "sum"), invalid_due_to_quality_issue=("invalid_due_to_quality_issue", "sum"),
            valid_event_labels=("valid_event_labels", "sum"), positive_event_labels=("positive_event_labels", "sum"),
        )
        return {"items": records(summary), "totals": records(totals), "event_definition": self.analysis["forecast_dataset"]["event_definition"]}

    def targets(self, city: str | None, sensor_id: int | None, horizon: str, start_time: str | None, end_time: str | None, max_points: int) -> dict[str, Any]:
        if horizon not in {"1h", "3h", "6h"}:
            raise ValueError("Horizon must be 1h, 3h, or 6h.")
        frame = self.filter_frame(self.frames["forecast_targets"], city, sensor_id, start_time, end_time)
        columns = ["timestamp", "sensor_id", "city_name", f"target_pm25_{horizon}", f"target_pm25_who_24h_avg_{horizon}", f"pollution_event_{horizon}", f"forecast_sample_valid_{horizon}"]
        frame = frame[columns]
        per_sensor = max(25, max_points // max(1, frame.sensor_id.nunique()))
        sampled = []
        for _, group in frame.groupby("sensor_id", sort=True):
            stride = max(1, len(group) // per_sensor)
            sampled.append(group.iloc[::stride].head(per_sensor))
        output = pd.concat(sampled).sort_values(["sensor_id", "timestamp"]) if sampled else frame.head(0)
        return {"horizon": horizon, "items": records(output), "event_definition": self.analysis["forecast_dataset"]["event_definition"]}

    def who_frame(self, pollutant: str, city: str | None, sensor_id: int | None, start_time: str | None = None, end_time: str | None = None) -> pd.DataFrame:
        frame = self.frames["who_analysis"]
        frame = frame.loc[frame.pollutant == pollutant]
        return self.filter_frame(frame, city, sensor_id, start_time, end_time)

    def who_status(self, city: str | None, sensor_id: int | None) -> dict[str, Any]:
        frame = self.filter_frame(self.frames["who_analysis"], city, sensor_id)
        latest_rows = []
        for (_, _), group in frame.groupby(["sensor_id", "pollutant"], sort=True):
            usable = group.loc[~group.status.isin(["Insufficient Data", "Not Available"])]
            row = usable.iloc[-1] if len(usable) else group.iloc[-1]
            latest_rows.append(row)
        latest = pd.DataFrame(latest_rows)
        overall = []
        for sensor, group in latest.groupby("sensor_id"):
            supported = group.loc[group.unit_compatible.astype(str).str.casefold() == "true"]
            above = supported.loc[supported.status == "Above WHO Guideline"]
            selected = above.sort_values("ratio", ascending=False).iloc[0] if len(above) else supported.iloc[0] if len(supported) else None
            overall.append({
                "sensor_id": int(sensor), "city": group.city.iloc[0],
                "overall_status": selected.status if selected is not None else "Insufficient Data",
                "dominant_pollutant": selected.pollutant.upper() if selected is not None else None,
            })
        return {"items": records(latest), "overall": overall, "warning": self.analysis["who"]["warning"], "source": self.analysis["who"]["source"]}


@lru_cache(maxsize=1)
def get_phase3_store() -> Phase3Store:
    return Phase3Store()
