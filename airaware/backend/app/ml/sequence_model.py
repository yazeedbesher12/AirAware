from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ForecastModel, ModelOutput, SEED


class RecurrentNetwork(nn.Module):
    def __init__(self, input_size: int, cell: str, hidden_size: int, layers: int, dropout: float):
        super().__init__()
        recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        self.recurrent = recurrent(input_size, hidden_size, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, _ = self.recurrent(x)
        return self.output(self.dropout(values[:, -1, :]))


def build_sequences(frame: pd.DataFrame, x: np.ndarray, target_columns: tuple[str, str], lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []; targets: list[np.ndarray] = []; positions: list[int] = []
    working = frame.reset_index(drop=True)
    for _, group in working.groupby(["sensor_id", "continuous_segment_id"], sort=False):
        positions_group = group.index.to_numpy()
        timestamps = group.timestamp.to_numpy()
        for offset in range(lookback - 1, len(group)):
            window_positions = positions_group[offset - lookback + 1:offset + 1]
            elapsed = (timestamps[offset] - timestamps[offset - lookback + 1]) / np.timedelta64(1, "m")
            if elapsed != (lookback - 1) * 15:
                continue
            target = working.loc[positions_group[offset], list(target_columns)].to_numpy(dtype=float)
            if np.isfinite(target).all():
                sequences.append(x[window_positions]); targets.append(target); positions.append(int(positions_group[offset]))
    if not sequences:
        return np.empty((0, lookback, x.shape[1])), np.empty((0, 2)), np.empty(0, dtype=int)
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(positions, dtype=int)


class TorchSequenceModel(ForecastModel):
    def __init__(self, cell: str):
        self.cell = cell
        self.name = cell
        self.hyperparameters = {"lookback_samples": 12, "lookback_hours": 3, "hidden_units": 32, "layers": 1, "dropout": 0.15, "learning_rate": 0.002, "batch_size": 64, "max_epochs": 12, "patience": 3, "random_seed": SEED}

    def fit_predict(self, train: pd.DataFrame, predict: pd.DataFrame, feature_columns: list[str], target_columns: tuple[str, str], artifact_path: Path | None = None) -> ModelOutput:
        torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
        lookback = int(self.hyperparameters["lookback_samples"])
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_x = scaler.fit_transform(imputer.fit_transform(train[feature_columns]))
        train_seq, train_y, _ = build_sequences(train, train_x, target_columns, lookback)
        if len(train_seq) < 100:
            raise ValueError(f"Only {len(train_seq)} valid sequences; at least 100 are required")
        split = max(1, int(len(train_seq) * 0.85))
        x_fit, y_fit = train_seq[:split], train_y[:split]
        x_valid, y_valid = train_seq[split:], train_y[split:]
        target_mean = y_fit.mean(axis=0); target_std = y_fit.std(axis=0); target_std[target_std < 1e-6] = 1.0
        y_fit_scaled = (y_fit - target_mean) / target_std
        y_valid_scaled = (y_valid - target_mean) / target_std
        train_loader = DataLoader(TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit_scaled.astype(np.float32))), batch_size=int(self.hyperparameters["batch_size"]), shuffle=False, num_workers=0)
        network = RecurrentNetwork(len(feature_columns), self.cell, int(self.hyperparameters["hidden_units"]), int(self.hyperparameters["layers"]), float(self.hyperparameters["dropout"]))
        optimizer = torch.optim.Adam(network.parameters(), lr=float(self.hyperparameters["learning_rate"]))
        loss_fn = nn.MSELoss(); best_loss = float("inf"); best_state = None; stale = 0; history = []
        for epoch in range(1, int(self.hyperparameters["max_epochs"]) + 1):
            network.train(); losses = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad(); loss = loss_fn(network(batch_x), batch_y); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            network.eval()
            with torch.no_grad():
                valid_loss = float(loss_fn(network(torch.from_numpy(x_valid)), torch.from_numpy(y_valid_scaled.astype(np.float32)))) if len(x_valid) else float(np.mean(losses))
            train_loss = float(np.mean(losses)); history.append({"epoch": epoch, "training_loss": train_loss, "validation_loss": valid_loss})
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss; best_state = copy.deepcopy(network.state_dict()); stale = 0
            else:
                stale += 1
                if stale >= int(self.hyperparameters["patience"]): break
        if best_state is not None: network.load_state_dict(best_state)
        context_tail = train.groupby(["sensor_id", "continuous_segment_id"], group_keys=False).tail(lookback)
        context = pd.concat([context_tail, predict], ignore_index=True).drop_duplicates(["timestamp", "sensor_id"], keep="last").sort_values(["sensor_id", "continuous_segment_id", "timestamp"]).reset_index(drop=True)
        context_x = scaler.transform(imputer.transform(context[feature_columns]))
        context_seq, _, context_positions = build_sequences(context.assign(**{target_columns[0]: context[target_columns[0]].fillna(0), target_columns[1]: context[target_columns[1]].fillna(0)}), context_x, target_columns, lookback)
        prediction_lookup: dict[tuple[int, pd.Timestamp], np.ndarray] = {}
        network.eval()
        with torch.no_grad():
            if len(context_seq):
                scaled_predictions = network(torch.from_numpy(context_seq)).numpy()
                values = scaled_predictions * target_std + target_mean
                for position, value in zip(context_positions, values):
                    row = context.iloc[position]; prediction_lookup[(int(row.sensor_id), row.timestamp)] = value
        output = np.full((len(predict), 2), np.nan)
        for position, row in enumerate(predict.itertuples()):
            value = prediction_lookup.get((int(row.sensor_id), row.timestamp))
            if value is not None: output[position] = value
        best_epoch = int(min(history, key=lambda row: row["validation_loss"])["epoch"])
        if artifact_path:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": network.state_dict(), "input_size": len(feature_columns), "cell": self.cell, "hyperparameters": self.hyperparameters, "target_mean": target_mean, "target_std": target_std, "features": feature_columns}, artifact_path)
            joblib.dump({"imputer": imputer, "scaler": scaler}, artifact_path.with_suffix(".preprocessing.joblib"))
        return ModelOutput(output[:, 0], output[:, 1], history=history, best_epoch=best_epoch, warnings=["CPU training with a 3-hour sequence; scalers and imputers were fitted on training rows only."])
