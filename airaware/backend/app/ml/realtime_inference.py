from __future__ import annotations

import json
import warnings
from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .sequence_model import RecurrentNetwork


class SavedModelPredictor:
    """Load a final Phase 4A artifact once and perform inference without retraining."""
    def __init__(self, backend_dir: Path, model: str, horizon: str):
        self.backend_dir = backend_dir; self.model_name = model; self.horizon = horizon
        config_path = backend_dir / "models" / model / f"pm25_{horizon}_config.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.features: list[str] = self.config["features"]
        self.sequence_buffers: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=12))
        self.loaded = self._load()

    def _load(self) -> Any:
        directory = self.backend_dir / "models" / self.model_name
        stem = f"pm25_{self.horizon}"
        if self.model_name == "persistence": return json.loads((directory / f"{stem}.json").read_text(encoding="utf-8"))
        if self.model_name in {"random_forest", "xgboost", "lightgbm"}:
            loaded = joblib.load(directory / f"{stem}.joblib")
            if self.model_name == "random_forest": loaded.named_steps["model"].set_params(n_jobs=1)
            return loaded
        if self.model_name == "prophet":
            from prophet.serialize import model_from_json
            manifest_path = directory / f"{stem}.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")); models = {}
            for key, path in manifest["models"].items():
                model_path = Path(path)
                if not model_path.is_absolute():
                    model_path = manifest_path.parent / model_path
                models[key] = model_from_json(model_path.read_text(encoding="utf-8"))
            return models
        if self.model_name in {"lstm", "gru"}:
            checkpoint = torch.load(directory / f"{stem}.pt", map_location="cpu", weights_only=False)
            network = RecurrentNetwork(checkpoint["input_size"], checkpoint["cell"], checkpoint["hyperparameters"]["hidden_units"], checkpoint["hyperparameters"]["layers"], checkpoint["hyperparameters"]["dropout"])
            network.load_state_dict(checkpoint["state_dict"]); network.eval()
            preprocessing = joblib.load(directory / f"{stem}.preprocessing.joblib")
            return {"checkpoint": checkpoint, "network": network, **preprocessing}
        raise ValueError(f"Streaming predictor is not implemented for {self.model_name}")

    def reset_sensor(self, sensor_id: int) -> None:
        self.sequence_buffers.pop(int(sensor_id), None)

    def predict(self, feature_row: pd.DataFrame) -> tuple[float, float, float]:
        started = perf_counter(); sensor = int(feature_row.sensor_id.iloc[0])
        if self.model_name == "persistence":
            concentration = float(feature_row.pm25.iloc[0]); health = float(feature_row.pm25_who_24h_average.iloc[0]) if pd.notna(feature_row.pm25_who_24h_average.iloc[0]) else concentration
        elif self.model_name == "random_forest":
            values = self.loaded.predict(feature_row[self.features])[0]; concentration, health = float(values[0]), float(values[1])
        elif self.model_name in {"xgboost", "lightgbm"}:
            values = self.loaded["imputer"].transform(feature_row[self.features]); concentration = float(self.loaded["models"][0].predict(values)[0]); health = float(self.loaded["models"][1].predict(values)[0])
        elif self.model_name == "prophet":
            moment = pd.DataFrame({"ds": [feature_row.timestamp.iloc[0].tz_localize(None)]})
            target = f"target_pm25_{self.horizon}"; average = f"target_pm25_who_24h_avg_{self.horizon}"
            concentration = float(self.loaded[f"sensor_{sensor}_{target}"].predict(moment).yhat.iloc[0]); health = float(self.loaded[f"sensor_{sensor}_{average}"].predict(moment).yhat.iloc[0])
        else:
            values = self.loaded["scaler"].transform(self.loaded["imputer"].transform(feature_row[self.features]))[0].astype(np.float32)
            self.sequence_buffers[sensor].append(values)
            if len(self.sequence_buffers[sensor]) < 12: return np.nan, np.nan, (perf_counter() - started) * 1000
            tensor = torch.from_numpy(np.asarray([list(self.sequence_buffers[sensor])], dtype=np.float32))
            with torch.no_grad(): scaled = self.loaded["network"](tensor).numpy()[0]
            output = scaled * self.loaded["checkpoint"]["target_std"] + self.loaded["checkpoint"]["target_mean"]
            concentration, health = float(output[0]), float(output[1])
        return concentration, health, (perf_counter() - started) * 1000


def artifact_size_mb(path: Path) -> float:
    if not path.exists(): return 0.0
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    return sum(item.stat().st_size for item in files) / (1024 * 1024)
