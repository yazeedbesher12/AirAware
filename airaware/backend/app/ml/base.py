from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelOutput:
    concentration: np.ndarray
    health_average: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    best_epoch: int | None = None


class ForecastModel(ABC):
    name: str
    hyperparameters: dict[str, Any]

    @abstractmethod
    def fit_predict(
        self,
        train: pd.DataFrame,
        predict: pd.DataFrame,
        feature_columns: list[str],
        target_columns: tuple[str, str],
        artifact_path: Path | None = None,
    ) -> ModelOutput:
        raise NotImplementedError


SEED = 42
