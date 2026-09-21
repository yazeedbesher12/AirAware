from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from app.config import BACKEND_DIR, OUTPUT_DIR
from app.services.data_store import records


class Phase4Store:
    def __init__(self):
        self.outputs = OUTPUT_DIR
        self.models_dir = BACKEND_DIR / "models"

    def json(self, filename: str, default: Any = None) -> Any:
        path = self.outputs / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def model_detail(self, model_name: str) -> dict[str, Any]:
        summary = self.json("model_results.json", {"models": {}}).get("models", {}).get(model_name)
        if summary is None: raise KeyError(model_name)
        detailed = [row for row in self.json("model_results_detailed.json", []) if row.get("model") == model_name]
        importance = {}; histories = {}; model_dir = self.models_dir / model_name
        for horizon in ("1h", "3h", "6h"):
            importance_file = model_dir / f"pm25_{horizon}_feature_importance.csv"
            history_file = model_dir / f"pm25_{horizon}_history.json"
            if importance_file.exists(): importance[horizon] = records(pd.read_csv(importance_file).head(30))
            if history_file.exists(): histories[horizon] = json.loads(history_file.read_text(encoding="utf-8"))
        realtime = self.json("realtime_replay_results.json", {"models": {}}).get("models", {}).get(model_name)
        return {"name": model_name, "summary": summary, "details": detailed, "feature_importance": importance, "training_history": histories, "realtime": realtime}

    def predictions(self, model_name: str, horizon: str, split: str, sensor_id: int | None, max_points: int) -> dict[str, Any]:
        path = self.outputs / "predictions" / model_name / f"{split}_pm25_{horizon}.csv"
        if not path.exists(): raise KeyError(f"{model_name}/{split}/{horizon}")
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        if sensor_id is not None: frame = frame.loc[frame.sensor_id == sensor_id]
        total = len(frame); stride = max(1, total // max_points)
        return {"model": model_name, "horizon": horizon, "split": split, "total": total, "items": records(frame.iloc[::stride].head(max_points))}


@lru_cache(maxsize=1)
def get_phase4_store() -> Phase4Store:
    return Phase4Store()
