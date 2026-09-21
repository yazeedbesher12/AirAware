from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MultiLoss, RMSE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .base import ForecastModel, ModelOutput, SEED


class TFTModel(ForecastModel):
    name = "tft"
    hyperparameters = {"encoder_length": 24, "prediction_length": 1, "hidden_size": 8, "attention_head_size": 1, "hidden_continuous_size": 8, "dropout": 0.15, "learning_rate": 0.01, "batch_size": 64, "max_epochs": 5, "patience": 2, "random_seed": SEED}

    def fit_predict(self, train: pd.DataFrame, predict: pd.DataFrame, feature_columns: list[str], target_columns: tuple[str, str], artifact_path: Path | None = None) -> ModelOutput:
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")
        seed_everything(SEED, workers=True); torch.set_num_threads(4)
        selected = [column for column in feature_columns if column in {
            "pm25", "temperature", "humidity", "hour", "day_of_week", "is_weekend", "sin_hour", "cos_hour",
            "sin_day_of_week", "cos_day_of_week", "pm25_lag_15m", "pm25_lag_1h", "pm25_lag_3h",
            "pm25_mean_1h", "pm25_mean_3h", "pm25_std_3h", "recent_gap_count_3h", "recent_anomaly_count_3h",
            "pm25_who_24h_average", "pm25_who_24h_coverage",
        }]
        imputer = SimpleImputer(strategy="median"); scaler = StandardScaler()
        train_values = scaler.fit_transform(imputer.fit_transform(train[selected]))
        prepared_train = self._prepare(train, selected, train_values)
        encoder = int(self.hyperparameters["encoder_length"])
        training_dataset = TimeSeriesDataSet(
            prepared_train,
            time_idx="tft_time_idx", target=list(target_columns), group_ids=["series_id"],
            min_encoder_length=12, max_encoder_length=encoder, min_prediction_length=1, max_prediction_length=1,
            static_categoricals=["sensor_key"], time_varying_known_reals=selected,
            target_normalizer=MultiNormalizer([EncoderNormalizer(), EncoderNormalizer()]),
            categorical_encoders={"series_id": NaNLabelEncoder(add_nan=True), "sensor_key": NaNLabelEncoder(add_nan=True)},
            allow_missing_timesteps=True, add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        )
        training_loader = training_dataset.to_dataloader(train=True, batch_size=int(self.hyperparameters["batch_size"]), num_workers=0)
        model = TemporalFusionTransformer.from_dataset(
            training_dataset, learning_rate=float(self.hyperparameters["learning_rate"]),
            hidden_size=int(self.hyperparameters["hidden_size"]), attention_head_size=int(self.hyperparameters["attention_head_size"]),
            hidden_continuous_size=int(self.hyperparameters["hidden_continuous_size"]), dropout=float(self.hyperparameters["dropout"]),
            output_size=[1, 1], loss=MultiLoss([RMSE(), RMSE()]), log_interval=-1, reduce_on_plateau_patience=2,
        )
        trainer = Trainer(
            max_epochs=int(self.hyperparameters["max_epochs"]), accelerator="cpu", devices=1,
            gradient_clip_val=0.1, callbacks=[EarlyStopping(monitor="train_loss_epoch", patience=int(self.hyperparameters["patience"]), mode="min")],
            logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False,
        )
        trainer.fit(model, train_dataloaders=training_loader)
        context_tail = train.groupby(["sensor_id", "continuous_segment_id"], group_keys=False).tail(encoder)
        combined = pd.concat([context_tail, predict], ignore_index=True).drop_duplicates(["timestamp", "sensor_id"], keep="last")
        combined = combined.sort_values(["sensor_id", "continuous_segment_id", "timestamp"]).reset_index(drop=True)
        combined_values = scaler.transform(imputer.transform(combined[selected]))
        prepared_predict = self._prepare(combined, selected, combined_values)
        minimum_time = int(self._time_index(predict.timestamp).min())
        prediction_dataset = TimeSeriesDataSet.from_dataset(training_dataset, prepared_predict, min_prediction_idx=minimum_time, stop_randomization=True)
        prediction_loader = prediction_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
        raw = model.predict(prediction_loader, mode="prediction", return_index=True, trainer_kwargs={"accelerator": "cpu", "devices": 1, "enable_progress_bar": False})
        prediction_values = raw.output
        if isinstance(prediction_values, (list, tuple)):
            concentration = np.asarray(prediction_values[0]).reshape(-1)
            health = np.asarray(prediction_values[1]).reshape(-1)
        else:
            array = np.asarray(prediction_values)
            concentration, health = array[:, 0].reshape(-1), array[:, 1].reshape(-1)
        lookup = {}
        for row, conc, avg in zip(raw.index.itertuples(), concentration, health):
            lookup[(str(row.series_id), int(row.tft_time_idx))] = (float(conc), float(avg))
        output = np.full((len(predict), 2), np.nan)
        for position, row in enumerate(predict.itertuples()):
            key = (f"{int(row.sensor_id)}::{row.continuous_segment_id}", int(self._time_index(pd.Series([row.timestamp])).iloc[0]))
            if key in lookup: output[position] = lookup[key]
        history = [{"epoch": int(trainer.current_epoch + 1), "training_loss": float(trainer.callback_metrics.get("train_loss_epoch", np.nan)), "validation_loss": None}]
        if artifact_path:
            artifact_path.parent.mkdir(parents=True, exist_ok=True); trainer.save_checkpoint(str(artifact_path))
            import joblib
            joblib.dump({"imputer": imputer, "scaler": scaler, "features": selected, "dataset_parameters": training_dataset.get_parameters()}, artifact_path.with_suffix(".preprocessing.joblib"))
        return ModelOutput(output[:, 0], output[:, 1], history=history, best_epoch=int(trainer.current_epoch + 1), warnings=["TFT used an established PyTorch Forecasting implementation with a compact CPU-safe configuration."])

    @staticmethod
    def _time_index(values: pd.Series) -> pd.Series:
        timestamps = pd.to_datetime(values, utc=True)
        origin = pd.Timestamp("2026-08-06T00:00:00Z")
        return ((timestamps - origin).dt.total_seconds() // 900).astype(int)

    def _prepare(self, frame: pd.DataFrame, selected: list[str], values: np.ndarray) -> pd.DataFrame:
        result = frame[["timestamp", "sensor_id", "continuous_segment_id", "target_pm25_1h", "target_pm25_3h", "target_pm25_6h", "target_pm25_who_24h_avg_1h", "target_pm25_who_24h_avg_3h", "target_pm25_who_24h_avg_6h"]].copy()
        result[selected] = values
        result["sensor_key"] = frame.sensor_id.astype(int).astype(str).to_numpy()
        result["series_id"] = frame.sensor_id.astype(int).astype(str).to_numpy() + "::" + frame.continuous_segment_id.astype(str).to_numpy()
        result["tft_time_idx"] = self._time_index(frame.timestamp).to_numpy()
        return result.sort_values(["series_id", "tft_time_idx"]).reset_index(drop=True)
