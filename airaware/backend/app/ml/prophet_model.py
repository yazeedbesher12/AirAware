from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

from .base import ForecastModel, ModelOutput, SEED


class ProphetModel(ForecastModel):
    name = "prophet"
    hyperparameters = {"daily_seasonality": True, "weekly_seasonality": False, "yearly_seasonality": False, "seasonality_mode": "additive", "changepoint_prior_scale": 0.05, "uncertainty_samples": 0}

    def fit_predict(self, train: pd.DataFrame, predict: pd.DataFrame, feature_columns: list[str], target_columns: tuple[str, str], artifact_path: Path | None = None) -> ModelOutput:
        result = np.full((len(predict), 2), np.nan, dtype=float)
        saved: dict[str, str] = {}
        for sensor, predict_group in predict.groupby("sensor_id", sort=False):
            train_group = train[train.sensor_id == sensor].sort_values("timestamp")
            if len(train_group) < 96:
                continue
            indices = predict.index.get_indexer(predict_group.index)
            for target_index, target in enumerate(target_columns):
                history = train_group[["timestamp", target]].dropna().rename(columns={"timestamp": "ds", target: "y"})
                history["ds"] = history.ds.dt.tz_localize(None)
                model = Prophet(**self.hyperparameters)
                model.fit(history)
                future = pd.DataFrame({"ds": predict_group.timestamp.dt.tz_localize(None)})
                result[indices, target_index] = model.predict(future).yhat.to_numpy(float)
                if artifact_path:
                    from prophet.serialize import model_to_json
                    model_file = artifact_path.with_name(f"{artifact_path.stem}_sensor_{int(sensor)}_{target}.json")
                    model_file.parent.mkdir(parents=True, exist_ok=True)
                    model_file.write_text(model_to_json(model), encoding="utf-8")
                    # Store portable filenames; callers resolve them relative to the manifest.
                    saved[f"sensor_{int(sensor)}_{target}"] = model_file.name
        if artifact_path:
            artifact_path.write_text(json.dumps({"models": saved, "hyperparameters": self.hyperparameters}, indent=2), encoding="utf-8")
        return ModelOutput(result[:, 0], result[:, 1], warnings=["Prophet uses ds/y per sensor and does not consume the multivariate engineered feature set."])
