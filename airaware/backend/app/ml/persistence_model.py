from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelOutput


class PersistenceModel(ForecastModel):
    name = "persistence"
    hyperparameters = {"rule": "prediction(t+h)=latest valid value at t"}

    def fit_predict(self, train: pd.DataFrame, predict: pd.DataFrame, feature_columns: list[str], target_columns: tuple[str, str], artifact_path: Path | None = None) -> ModelOutput:
        if artifact_path:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(self.hyperparameters, indent=2), encoding="utf-8")
        health = pd.to_numeric(predict["pm25_who_24h_average"], errors="coerce").to_numpy(float)
        fallback = pd.to_numeric(predict["pm25"], errors="coerce").to_numpy(float)
        health = np.where(np.isfinite(health), health, fallback)
        return ModelOutput(pd.to_numeric(predict["pm25"], errors="coerce").to_numpy(float), health)
