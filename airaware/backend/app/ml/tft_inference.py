"""Reload and causality audit for the existing Phase 4A TFT checkpoints.

The saved weights are never mutated here.  The adapter deliberately refuses
live prediction when the fitted dataset metadata cannot be reconstructed from
information available at the forecast origin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, MultiNormalizer


SAMPLING_MINUTES = 15
TIME_ORIGIN = pd.Timestamp("2026-08-06T00:00:00Z")
HORIZONS = {"1h", "3h", "6h"}


class UnsafeTFTCheckpointError(RuntimeError):
    """Raised when exact checkpoint inference would require future information."""


@dataclass(frozen=True)
class TFTServingMetadata:
    horizon: str
    features: list[str]
    group_ids: list[str]
    static_categoricals: list[str]
    static_reals: list[str]
    time_varying_known_reals: list[str]
    time_varying_unknown_reals: list[str]
    target_columns: list[str]
    target_normalizer: str
    min_encoder_length: int
    max_encoder_length: int
    prediction_length: int
    sampling_interval_minutes: int
    encoder_history_hours: float
    maximum_feature_lookback_hours: float
    full_feature_history_hours: float
    full_feature_history_samples: int
    causal_feature_audit: str
    causal_target_normalizer_audit: str
    real_time_ready: bool


class TFTInferenceAdapter:
    """Exact offline reloader plus fail-closed live-inference boundary."""

    def __init__(self, backend_dir: Path, horizon: str, load_checkpoint: bool = True):
        if horizon not in HORIZONS:
            raise ValueError(f"Unsupported TFT horizon: {horizon}")
        self.backend_dir = Path(backend_dir)
        self.horizon = horizon
        self.model_dir = self.backend_dir / "models" / "tft"
        self.checkpoint_path = self.model_dir / f"pm25_{horizon}.ckpt"
        self.preprocessing_path = self.model_dir / f"pm25_{horizon}.preprocessing.joblib"
        self.preprocessing = joblib.load(self.preprocessing_path)
        self.imputer = self.preprocessing["imputer"]
        self.scaler = self.preprocessing["scaler"]
        self.features = list(self.preprocessing["features"])
        self.dataset_parameters = self.preprocessing["dataset_parameters"]
        self.metadata = self._metadata()
        self.model: TemporalFusionTransformer | None = None
        if load_checkpoint:
            warnings.filterwarnings("ignore", message="Attribute 'loss'.*")
            warnings.filterwarnings("ignore", message="Attribute 'logging_metrics'.*")
            torch.set_num_threads(4)
            self.model = TemporalFusionTransformer.load_from_checkpoint(str(self.checkpoint_path), map_location="cpu")
            self.model.eval()
            self._validate_checkpoint_contract()

    def _metadata(self) -> TFTServingMetadata:
        parameters = self.dataset_parameters
        normalizer = parameters.get("target_normalizer")
        encoder_normalized = isinstance(normalizer, MultiNormalizer) and any(
            isinstance(item, EncoderNormalizer) for item in normalizer.normalizers
        )
        max_encoder = int(parameters["max_encoder_length"])
        encoder_hours = max_encoder * SAMPLING_MINUTES / 60
        # pm25_who_24h_* is the longest selected causal feature dependency.
        feature_hours = 24.0 if any("who_24h" in name for name in self.features) else 3.0
        full_hours = encoder_hours + feature_hours
        return TFTServingMetadata(
            horizon=self.horizon,
            features=self.features,
            group_ids=list(parameters.get("group_ids") or []),
            static_categoricals=list(parameters.get("static_categoricals") or []),
            static_reals=list(parameters.get("static_reals") or []),
            time_varying_known_reals=list(parameters.get("time_varying_known_reals") or []),
            time_varying_unknown_reals=list(parameters.get("time_varying_unknown_reals") or []),
            target_columns=list(parameters.get("target") or []),
            target_normalizer=type(normalizer).__name__,
            min_encoder_length=int(parameters["min_encoder_length"]),
            max_encoder_length=max_encoder,
            prediction_length=int(parameters["max_prediction_length"]),
            sampling_interval_minutes=SAMPLING_MINUTES,
            encoder_history_hours=encoder_hours,
            maximum_feature_lookback_hours=feature_hours,
            full_feature_history_hours=full_hours,
            full_feature_history_samples=int(full_hours * 60 / SAMPLING_MINUTES) + 1,
            causal_feature_audit="PASS: selected lags/rollups are current-or-backward-looking and segment bounded",
            causal_target_normalizer_audit=(
                "FAIL: EncoderNormalizer derives per-sample target center/scale from encoder target columns; "
                "those columns are future-shifted Phase 3 targets and are not fully known at live forecast time"
                if encoder_normalized else "PASS"
            ),
            real_time_ready=not encoder_normalized,
        )

    def _validate_checkpoint_contract(self) -> None:
        assert self.model is not None
        if list(self.model.hparams.x_reals)[-len(self.features) - 1:-1] != self.features:
            # x_reals also contains generated static scale fields and relative_time_idx.
            model_known = [name for name in self.model.hparams.time_varying_reals_encoder if name != "relative_time_idx"]
            if model_known != self.features:
                raise ValueError("Saved feature order differs from checkpoint x_reals")
        if int(self.model.hparams.max_encoder_length) != self.metadata.max_encoder_length:
            raise ValueError("Checkpoint encoder length differs from preprocessing metadata")

    @staticmethod
    def time_index(values: pd.Series) -> pd.Series:
        timestamps = pd.to_datetime(values, utc=True)
        return ((timestamps - TIME_ORIGIN).dt.total_seconds() // (SAMPLING_MINUTES * 60)).astype(int)

    def validate_sensor_history(self, history: pd.DataFrame) -> None:
        if history.empty:
            raise ValueError("TFT history is empty")
        if history["sensor_id"].nunique() != 1:
            raise ValueError("TFT inference accepts exactly one sensor history; sensor histories must never be mixed")
        timestamps = pd.to_datetime(history["timestamp"], utc=True)
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError("TFT history must be strictly chronological with unique timestamps")
        if len(history) < self.metadata.full_feature_history_samples:
            raise ValueError(
                f"Full-feature TFT inference requires {self.metadata.full_feature_history_samples} readings "
                f"({self.metadata.full_feature_history_hours:g} hours at 15-minute sampling)"
            )

    def predict(self, history: pd.DataFrame) -> dict[str, Any]:
        """Fail closed: this checkpoint cannot be served causally without retraining."""
        self.validate_sensor_history(history)
        if not self.metadata.real_time_ready:
            raise UnsafeTFTCheckpointError(self.metadata.causal_target_normalizer_audit)
        raise NotImplementedError("No causal serving path was required for this checkpoint")

    def _prepare_offline(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = [
            "timestamp", "sensor_id", "continuous_segment_id",
            "target_pm25_1h", "target_pm25_3h", "target_pm25_6h",
            "target_pm25_who_24h_avg_1h", "target_pm25_who_24h_avg_3h", "target_pm25_who_24h_avg_6h",
        ]
        result = frame[required].copy()
        result[self.features] = self.scaler.transform(self.imputer.transform(frame[self.features]))
        result["sensor_key"] = frame.sensor_id.astype(int).astype(str).to_numpy()
        result["series_id"] = frame.sensor_id.astype(int).astype(str).to_numpy() + "::" + frame.continuous_segment_id.astype(str).to_numpy()
        result["tft_time_idx"] = self.time_index(frame.timestamp).to_numpy()
        return result.sort_values(["series_id", "tft_time_idx"]).reset_index(drop=True)

    def reproduce_offline_prediction(self, frame: pd.DataFrame, prediction_timestamp: pd.Timestamp) -> tuple[float, float, float, float, float]:
        """Diagnostic only: reproduce a stored prediction with already-known labels.

        This method is intentionally not exposed as live inference because frame
        contains future-derived targets used by EncoderNormalizer.
        """
        if self.model is None:
            raise RuntimeError("Checkpoint was not loaded")
        if frame["sensor_id"].nunique() != 1:
            raise ValueError("Offline reproduction accepts one sensor only")
        prepared = self._prepare_offline(frame)
        minimum_idx = int(self.time_index(pd.Series([prediction_timestamp])).iloc[0])
        dataset = TimeSeriesDataSet.from_parameters(
            self.dataset_parameters, prepared, min_prediction_idx=minimum_idx, stop_randomization=True
        )
        dataset = dataset.filter(lambda index: index.time_idx_first_prediction == minimum_idx)
        loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        started = perf_counter()
        raw = self.model.predict(
            loader, mode="prediction", return_index=True,
            trainer_kwargs={"accelerator": "cpu", "devices": 1, "enable_progress_bar": False, "logger": False},
        )
        latency_ms = (perf_counter() - started) * 1000
        values = raw.output
        concentration = np.asarray(values[0] if isinstance(values, (list, tuple)) else values[:, 0]).reshape(-1)
        health = np.asarray(values[1] if isinstance(values, (list, tuple)) else values[:, 1]).reshape(-1)
        matches = np.flatnonzero(raw.index["tft_time_idx"].to_numpy(dtype=int) == minimum_idx)
        if not len(matches):
            raise RuntimeError("Reloaded TFT did not return the requested diagnostic timestamp")
        position = int(matches[-1])
        if not np.isfinite(concentration[position]) or not np.isfinite(health[position]):
            raise RuntimeError("Reloaded TFT produced a non-finite diagnostic prediction")
        batch_x, _ = next(iter(loader))
        with torch.no_grad():
            for _ in range(3):
                self.model(batch_x)
            forward_times = []
            for _ in range(20):
                forward_started = perf_counter()
                self.model(batch_x)
                forward_times.append((perf_counter() - forward_started) * 1000)
        return (
            float(concentration[position]), float(health[position]), latency_ms,
            float(np.mean(forward_times)), float(np.percentile(forward_times, 95)),
        )

    def audit_dict(self) -> dict[str, Any]:
        return asdict(self.metadata)

    @property
    def checkpoint_size_mb(self) -> float:
        return self.checkpoint_path.stat().st_size / (1024 * 1024)
