from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)

from pytorch_forecasting.data import (
    NaNLabelEncoder,
    MultiNormalizer,
    TorchNormalizer,
)

from pytorch_forecasting.metrics import (
    MultiLoss,
    RMSE,
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


SEED = 42

SAMPLING_MINUTES = 15

ENCODER_LENGTH = 24       # 6 hours
PREDICTION_LENGTH = 12    # 3 hours

TIME_ORIGIN = pd.Timestamp(
    "2026-08-06T00:00:00Z"
)


# ============================================================
# Feature configuration
# ============================================================

OBSERVED_FEATURES = [
    "pm25",
    "temperature",
    "humidity",

    "pm25_lag_15m",
    "pm25_lag_1h",
    "pm25_lag_3h",

    "pm25_mean_1h",
    "pm25_mean_3h",
    "pm25_std_3h",

    "recent_gap_count_3h",
    "recent_anomaly_count_3h",

    "pm25_who_24h_average",
    "pm25_who_24h_coverage",
]


KNOWN_FUTURE_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",

    "sin_hour",
    "cos_hour",

    "sin_day_of_week",
    "cos_day_of_week",
]


ALL_FEATURES = (
    OBSERVED_FEATURES
    + KNOWN_FUTURE_FEATURES
)


# ============================================================
# Helpers
# ============================================================

def time_index(
    values: pd.Series,
) -> pd.Series:

    timestamps = pd.to_datetime(
        values,
        utc=True,
    )

    return (
        (
            timestamps
            - TIME_ORIGIN
        )
        .dt
        .total_seconds()
        // (
            SAMPLING_MINUTES
            * 60
        )
    ).astype(int)


def prepare_frame(
    frame: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    fit_preprocessor: bool = False,
) -> pd.DataFrame:

    frame = (
        frame
        .copy()
        .sort_values(
            [
                "sensor_id",
                "continuous_segment_id",
                "timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    missing = [
        feature
        for feature
        in ALL_FEATURES
        if feature
        not in frame.columns
    ]

    if missing:

        raise ValueError(
            "Missing TFT causal features:\n"
            + "\n".join(
                missing
            )
        )

    values = frame[
        ALL_FEATURES
    ].copy()

    for column in ALL_FEATURES:

        values[column] = (
            pd.to_numeric(
                values[column],
                errors="coerce",
            )
        )

    if fit_preprocessor:

        values = (
            imputer.fit_transform(
                values
            )
        )

        values = (
            scaler.fit_transform(
                values
            )
        )

    else:

        values = (
            imputer.transform(
                values
            )
        )

        values = (
            scaler.transform(
                values
            )
        )

    result = pd.DataFrame(
        values,
        columns=ALL_FEATURES,
        index=frame.index,
    )

    # --------------------------------------------------------
    # Raw target
    # --------------------------------------------------------

    result[
        "target_pm25"
    ] = pd.to_numeric(
        frame["pm25"],
        errors="coerce",
    ).to_numpy()

    # --------------------------------------------------------
    # Structural columns
    # --------------------------------------------------------

    result[
        "sensor_key"
    ] = (
        frame[
            "sensor_id"
        ]
        .astype(int)
        .astype(str)
        .to_numpy()
    )

    result[
        "series_id"
    ] = (
        frame[
            "sensor_id"
        ]
        .astype(int)
        .astype(str)
        .to_numpy()
        + "::"
        + frame[
            "continuous_segment_id"
        ]
        .astype(str)
        .to_numpy()
    )

    result[
        "tft_time_idx"
    ] = (
        time_index(
            frame[
                "timestamp"
            ]
        )
        .to_numpy()
    )

    result[
        "timestamp"
    ] = pd.to_datetime(
        frame[
            "timestamp"
        ],
        utc=True,
    ).to_numpy()

    return (
        result
        .sort_values(
            [
                "series_id",
                "tft_time_idx",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Dataset
# ============================================================

def create_training_dataset(
    prepared_train: pd.DataFrame,
) -> TimeSeriesDataSet:

    target_normalizer = (
        TorchNormalizer(
            method="standard",
            center=True,
        )
    )

    dataset = TimeSeriesDataSet(
        prepared_train,

        time_idx=
            "tft_time_idx",

        target=
            "target_pm25",

        group_ids=[
            "series_id"
        ],

        min_encoder_length=
            ENCODER_LENGTH,

        max_encoder_length=
            ENCODER_LENGTH,

        min_prediction_length=
            PREDICTION_LENGTH,

        max_prediction_length=
            PREDICTION_LENGTH,

        static_categoricals=[
            "sensor_key"
        ],

        # Calendar values are genuinely known
        # into the future.
        time_varying_known_reals=
            KNOWN_FUTURE_FEATURES,

        # Sensor/environment observations are
        # only known up to the forecast origin.
        time_varying_unknown_reals=
            OBSERVED_FEATURES,

        target_normalizer=
            target_normalizer,

        categorical_encoders={
            "series_id":
                NaNLabelEncoder(
                    add_nan=True
                ),

            "sensor_key":
                NaNLabelEncoder(
                    add_nan=True
                ),
        },

        allow_missing_timesteps=True,

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return dataset


# ============================================================
# Model training
# ============================================================

def train_causal_tft(
    train_df: pd.DataFrame,
    artifact_dir: Path,
):

    warnings.filterwarnings(
        "ignore"
    )

    seed_everything(
        SEED,
        workers=True,
    )

    torch.set_num_threads(
        4
    )

    artifact_dir = Path(
        artifact_dir
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        artifact_dir
        / "pm25_3h_causal.ckpt"
    )

    preprocessing_path = (
        artifact_dir
        / "pm25_3h_causal.preprocessing.joblib"
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    prepared_train = (
        prepare_frame(
            train_df,
            imputer=
                imputer,
            scaler=
                scaler,
            fit_preprocessor=True,
        )
    )

    # --------------------------------------------------------
    # TFT dataset
    # --------------------------------------------------------

    training_dataset = (
        create_training_dataset(
            prepared_train
        )
    )

    training_loader = (
        training_dataset
        .to_dataloader(
            train=True,
            batch_size=64,
            num_workers=0,
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        TemporalFusionTransformer
        .from_dataset(
            training_dataset,

            learning_rate=
                0.01,

            hidden_size=
                8,

            attention_head_size=
                1,

            hidden_continuous_size=
                8,

            dropout=
                0.15,

            output_size=
                1,

            loss=
                RMSE(),

            log_interval=
                -1,

            reduce_on_plateau_patience=
                2,
        )
    )

    early_stop = (
        EarlyStopping(
            monitor=
                "train_loss_epoch",

            patience=
                2,

            mode=
                "min",
        )
    )

    trainer = Trainer(
        max_epochs=8,

        accelerator="cpu",
        devices=1,

        gradient_clip_val=0.1,

        callbacks=[
            early_stop
        ],

        logger=False,

        enable_checkpointing=False,
        enable_model_summary=False,

        enable_progress_bar=True,
    )

    trainer.fit(
        model,
        train_dataloaders=
            training_loader,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trainer.save_checkpoint(
        str(
            checkpoint_path
        )
    )

    joblib.dump(
        {
            "imputer":
                imputer,

            "scaler":
                scaler,

            "features":
                ALL_FEATURES,

            "observed_features":
                OBSERVED_FEATURES,

            "known_future_features":
                KNOWN_FUTURE_FEATURES,

            "dataset_parameters":
                training_dataset
                .get_parameters(),

            "encoder_length":
                ENCODER_LENGTH,

            "prediction_length":
                PREDICTION_LENGTH,

            "sampling_minutes":
                SAMPLING_MINUTES,

            "target":
                "target_pm25",

            "forecast_horizon":
                "3h",

            "causal":
                True,
        },

        preprocessing_path,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "CAUSAL TFT TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Checkpoint:\n"
        f"{checkpoint_path}"
    )

    print()

    print(
        f"Preprocessing:\n"
        f"{preprocessing_path}"
    )

    print()

    print(
        "Encoder:"
        f" {ENCODER_LENGTH} samples"
        f" = "
        f"{ENCODER_LENGTH * 15 / 60:.1f}h"
    )

    print(
        "Prediction:"
        f" {PREDICTION_LENGTH} samples"
        f" = "
        f"{PREDICTION_LENGTH * 15 / 60:.1f}h"
    )

    print()

    print(
        "3h prediction = "
        "last decoder prediction"
    )

    return {
        "model":
            model,

        "dataset":
            training_dataset,

        "checkpoint":
            checkpoint_path,

        "preprocessing":
            preprocessing_path,
    }