from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .base import ForecastModel, ModelOutput, SEED


class LightGBMModel(ForecastModel):
    name = "lightgbm"
    hyperparameters = {"n_estimators": 320, "learning_rate": 0.035, "num_leaves": 24, "max_depth": 8, "feature_fraction": 0.8, "bagging_fraction": 0.85, "bagging_freq": 1, "reg_alpha": 0.05, "reg_lambda": 0.3, "n_jobs": 4, "random_state": SEED, "verbosity": -1}

    def fit_predict(self, train: pd.DataFrame, predict: pd.DataFrame, feature_columns: list[str], target_columns: tuple[str, str], artifact_path: Path | None = None) -> ModelOutput:
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train[feature_columns]); x_predict = imputer.transform(predict[feature_columns])
        models = []; predictions = []
        for target in target_columns:
            model = lgb.LGBMRegressor(**self.hyperparameters)
            model.fit(x_train, train[target].to_numpy(float))
            models.append(model); predictions.append(model.predict(x_predict))
        importance = np.mean([m.feature_importances_ for m in models], axis=0)
        feature_importance = sorted(({"feature": name, "importance": float(value)} for name, value in zip(feature_columns, importance)), key=lambda row: row["importance"], reverse=True)
        if artifact_path:
            artifact_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump({"imputer": imputer, "models": models, "features": feature_columns}, artifact_path)
        return ModelOutput(predictions[0], predictions[1], feature_importance=feature_importance)
