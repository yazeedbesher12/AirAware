from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .base import ForecastModel, ModelOutput, SEED


class RandomForestModel(ForecastModel):
    name = "random_forest"
    hyperparameters = {"n_estimators": 180, "max_depth": 14, "min_samples_split": 4, "min_samples_leaf": 2, "max_features": 0.75, "n_jobs": 4, "random_state": SEED}

    def fit_predict(self, train: pd.DataFrame, predict: pd.DataFrame, feature_columns: list[str], target_columns: tuple[str, str], artifact_path: Path | None = None) -> ModelOutput:
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestRegressor(**self.hyperparameters)),
        ])
        pipeline.fit(train[feature_columns], train[list(target_columns)])
        prediction = pipeline.predict(predict[feature_columns])
        model = pipeline.named_steps["model"]
        importances = model.feature_importances_[:len(feature_columns)]
        feature_importance = sorted(
            ({"feature": name, "importance": float(value)} for name, value in zip(feature_columns, importances)),
            key=lambda row: row["importance"], reverse=True,
        )
        if artifact_path:
            artifact_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(pipeline, artifact_path)
        return ModelOutput(prediction[:, 0], prediction[:, 1], feature_importance=feature_importance)
