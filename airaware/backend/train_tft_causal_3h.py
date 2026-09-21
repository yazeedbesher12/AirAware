from __future__ import annotations

from pathlib import Path
import json
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
    TorchNormalizer,
)

from pytorch_forecasting.metrics import RMSE

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from app.ml.dataset import load_phase4_dataset


# ============================================================
# CONFIG
# ============================================================

BACKEND = Path(__file__).resolve().parent
OUTPUTS = BACKEND / "outputs"

MODEL_DIR = BACKEND / "models" / "tft_causal"
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    MODEL_DIR
    / "pm25_3h_causal.ckpt"
)

PREPROCESSING_PATH = (
    MODEL_DIR
    / "pm25_3h_causal.preprocessing.joblib"
)

PREDICTION_PATH = (
    OUTPUTS
    / "predictions"
    / "tft_causal"
    / "test_pm25_3h.csv"
)

RESULT_PATH = (
    OUTPUTS
    / "tft_causal_3h_results.json"
)

PREDICTION_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42

SAMPLING_MINUTES = 15

# 24 readings × 15 min = 6 hours
ENCODER_LENGTH = 24

# 12 future readings × 15 min = 3 hours
PREDICTION_LENGTH = 12

TIME_ORIGIN = pd.Timestamp(
    "2026-08-06T00:00:00Z"
)


# ============================================================
# FEATURES
# ============================================================

# These values are observations.
# They are known only up to forecast origin.
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


# These are deterministic calendar features.
# They can legitimately be known in the future.
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
# HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 95)
    print(title)
    print("=" * 95)


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


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_frame(
    frame: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    fit: bool = False,
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

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing = [
        col
        for col in ALL_FEATURES
        if col not in frame.columns
    ]

    if missing:

        raise ValueError(
            "Missing required TFT causal features:\n"
            + "\n".join(
                missing
            )
        )

    # --------------------------------------------------------
    # Input features
    # --------------------------------------------------------

    X = frame[
        ALL_FEATURES
    ].copy()

    for col in ALL_FEATURES:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Fit preprocessing ONLY on training data
    # --------------------------------------------------------

    if fit:

        X_imputed = (
            imputer.fit_transform(
                X
            )
        )

        X_scaled = (
            scaler.fit_transform(
                X_imputed
            )
        )

    else:

        X_imputed = (
            imputer.transform(
                X
            )
        )

        X_scaled = (
            scaler.transform(
                X_imputed
            )
        )

    result = pd.DataFrame(
        X_scaled,
        columns=ALL_FEATURES,
        index=frame.index,
    )

    # ========================================================
    # TARGET
    # ========================================================
    #
    # IMPORTANT:
    #
    # This is RAW current PM2.5.
    #
    # We are NOT using:
    #
    # target_pm25_3h
    #
    # as an encoder target.
    #
    # The model learns the real PM2.5 time series and predicts
    # the following 12 × 15-minute observations.
    #
    # ========================================================

    result[
        "target_pm25"
    ] = pd.to_numeric(
        frame[
            "pm25"
        ],
        errors="coerce",
    ).to_numpy()

    # ========================================================
    # TFT STRUCTURAL COLUMNS
    # ========================================================

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
    ] = (
        pd.to_datetime(
            frame[
                "timestamp"
            ],
            utc=True,
        )
        .to_numpy()
    )

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
# TRAINING DATASET
# ============================================================

def make_training_dataset(
    prepared_train: pd.DataFrame,
) -> TimeSeriesDataSet:

    dataset = TimeSeriesDataSet(

        prepared_train,

        # ----------------------------------------------------
        # Main structure
        # ----------------------------------------------------

        time_idx=
            "tft_time_idx",

        target=
            "target_pm25",

        group_ids=[
            "series_id"
        ],

        # ----------------------------------------------------
        # Encoder = past 6 hours
        # ----------------------------------------------------

        min_encoder_length=
            ENCODER_LENGTH,

        max_encoder_length=
            ENCODER_LENGTH,

        # ----------------------------------------------------
        # Decoder = next 3 hours
        # ----------------------------------------------------

        min_prediction_length=
            PREDICTION_LENGTH,

        max_prediction_length=
            PREDICTION_LENGTH,

        # ----------------------------------------------------
        # Static info
        # ----------------------------------------------------

        static_categoricals=[
            "sensor_key"
        ],

        # ----------------------------------------------------
        # Known future information
        # ----------------------------------------------------

        time_varying_known_reals=
            KNOWN_FUTURE_FEATURES,

        # ----------------------------------------------------
        # Observations available only up to forecast origin
        # ----------------------------------------------------

        time_varying_unknown_reals=
            OBSERVED_FEATURES,

        # ----------------------------------------------------
        # CAUSAL NORMALIZATION
        #
        # Unlike old EncoderNormalizer implementation,
        # this normalizer is fitted from TRAINING data.
        # ----------------------------------------------------

        target_normalizer=
            TorchNormalizer(
                method="standard",
                center=True,
            ),

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
# MAIN
# ============================================================

def main() -> None:

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

    # ========================================================
    # LOAD ORIGINAL AIRAWARE DATASET
    # ========================================================

    section(
        "AIRAWARE — CAUSAL TFT 3h TRAINING"
    )

    dataset = (
        load_phase4_dataset(
            OUTPUTS
        )
    )

    # ========================================================
    # SAME ORIGINAL 3h SPLITS
    # ========================================================

    train, validation, test = (
        dataset.splits(
            "3h"
        )
    )

    # Same final-training logic used by Phase4Trainer:
    #
    # final_train = train + validation

    final_train = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    final_train = (
        final_train
        .sort_values(
            [
                "timestamp",
                "sensor_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Train rows      : "
        f"{len(train):,}"
    )

    print(
        f"Validation rows : "
        f"{len(validation):,}"
    )

    print(
        f"Final train rows: "
        f"{len(final_train):,}"
    )

    print(
        f"Test rows       : "
        f"{len(test):,}"
    )

    print()

    print(
        f"Train first     : "
        f"{final_train.timestamp.min()}"
    )

    print(
        f"Train last      : "
        f"{final_train.timestamp.max()}"
    )

    print(
        f"Test first      : "
        f"{test.timestamp.min()}"
    )

    print(
        f"Test last       : "
        f"{test.timestamp.max()}"
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    section(
        "PREPARING TRAINING DATA"
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    prepared_train = (
        prepare_frame(
            final_train,

            imputer=
                imputer,

            scaler=
                scaler,

            fit=True,
        )
    )

    training_dataset = (
        make_training_dataset(
            prepared_train
        )
    )

    print(
        f"Training TFT sequences: "
        f"{len(training_dataset):,}"
    )

    training_loader = (
        training_dataset
        .to_dataloader(

            train=True,

            batch_size=64,

            num_workers=0,
        )
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    section(
        "TRAINING"
    )

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

        max_epochs=
            8,

        accelerator=
            "cpu",

        devices=
            1,

        gradient_clip_val=
            0.1,

        callbacks=[
            early_stop
        ],

        logger=
            False,

        enable_checkpointing=
            False,

        enable_model_summary=
            False,

        enable_progress_bar=
            True,
    )

    trainer.fit(

        model,

        train_dataloaders=
            training_loader,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    section(
        "SAVING CAUSAL MODEL"
    )

    trainer.save_checkpoint(
        str(
            CHECKPOINT_PATH
        )
    )

    preprocessing_bundle = {

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

        "real_time_ready_design":
            True,
    }

    joblib.dump(
        preprocessing_bundle,
        PREPROCESSING_PATH,
    )

    print(
        f"Checkpoint saved:\n"
        f"{CHECKPOINT_PATH}"
    )

    print()

    print(
        f"Preprocessing saved:\n"
        f"{PREPROCESSING_PATH}"
    )

    # ========================================================
    # TEST CONTEXT
    # ========================================================

    section(
        "PREPARING TEST INFERENCE"
    )

    # The decoder needs historical encoder context before
    # the TEST boundary.
    #
    # Keep enough tail rows from each continuous sensor
    # segment.

    context_rows = (
        ENCODER_LENGTH
        + PREDICTION_LENGTH
    )

    context = (
        final_train
        .groupby(
            [
                "sensor_id",
                "continuous_segment_id",
            ],
            group_keys=False,
        )
        .tail(
            context_rows
        )
    )

    combined = pd.concat(
        [
            context,
            test,
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(
            [
                "timestamp",
                "sensor_id",
            ],
            keep="last",
        )
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

    prepared_test = (
        prepare_frame(

            combined,

            imputer=
                imputer,

            scaler=
                scaler,

            fit=False,
        )
    )

    # Earliest TEST origin
    test_min_idx = int(
        time_index(
            test[
                "timestamp"
            ]
        ).min()
    )

    # --------------------------------------------------------
    # IMPORTANT ALIGNMENT
    #
    # If origin is at:
    #
    # t
    #
    # decoder #1 is:
    #
    # t + 15m
    #
    # So min_prediction_idx should allow the decoder to begin
    # immediately after the first possible origin.
    # --------------------------------------------------------

    minimum_decoder_idx = (
        test_min_idx
        + 1
    )

    prediction_dataset = (
        TimeSeriesDataSet
        .from_dataset(

            training_dataset,

            prepared_test,

            min_prediction_idx=
                minimum_decoder_idx,

            stop_randomization=
                True,
        )
    )

    print(
        f"Prediction sequences: "
        f"{len(prediction_dataset):,}"
    )

    prediction_loader = (
        prediction_dataset
        .to_dataloader(

            train=False,

            batch_size=128,

            num_workers=0,
        )
    )

    # ========================================================
    # REAL MODEL INFERENCE
    # ========================================================

    section(
        "TEST INFERENCE"
    )

    raw = model.predict(

        prediction_loader,

        mode=
            "prediction",

        return_index=
            True,

        trainer_kwargs={

            "accelerator":
                "cpu",

            "devices":
                1,

            "enable_progress_bar":
                False,

            "logger":
                False,
        },
    )

    predictions = np.asarray(
        raw.output
    )

    print(
        f"Raw prediction shape: "
        f"{predictions.shape}"
    )

    # Depending on pytorch_forecasting version:
    #
    # possible:
    #
    # (N, 12)
    #
    # or:
    #
    # (N, 12, 1)

    if predictions.ndim == 3:

        predictions = (
            predictions[
                :,
                :,
                0,
            ]
        )

    if predictions.ndim != 2:

        raise RuntimeError(
            "Unexpected TFT prediction dimensions: "
            f"{predictions.shape}"
        )

    if (
        predictions.shape[1]
        != PREDICTION_LENGTH
    ):

        raise RuntimeError(
            "Expected "
            f"{PREDICTION_LENGTH} decoder steps, "
            f"received shape "
            f"{predictions.shape}"
        )

    # ========================================================
    # 3-HOUR FORECAST
    # ========================================================
    #
    # Decoder:
    #
    # Step 1  = origin + 15m
    # Step 2  = origin + 30m
    # ...
    # Step 12 = origin + 180m
    #
    # Therefore last decoder output = +3h forecast.
    # ========================================================

    forecast_3h = (
        predictions[
            :,
            -1
        ]
    )

    prediction_index = (
        raw.index.copy()
    )

    prediction_index[
        "predicted"
    ] = forecast_3h

    # ========================================================
    # CORRECT TEMPORAL ALIGNMENT
    # ========================================================
    #
    # raw.index["tft_time_idx"]
    #
    # = timestamp of FIRST DECODER prediction
    #
    # Therefore:
    #
    # forecast_origin = first_decoder_idx - 1
    #
    # because one interval = 15 minutes.
    #
    # Example:
    #
    # Origin       12:00
    # Decoder #1   12:15
    # Decoder #12  15:00
    #
    # So decoder #12 = origin + 3h.
    # ========================================================

    prediction_index[
        "first_decoder_tft_time_idx"
    ] = (
        prediction_index[
            "tft_time_idx"
        ]
        .astype(int)
    )

    prediction_index[
        "origin_tft_time_idx"
    ] = (
        prediction_index[
            "first_decoder_tft_time_idx"
        ]
        - 1
    )

    prediction_index[
        "target_tft_time_idx"
    ] = (
        prediction_index[
            "first_decoder_tft_time_idx"
        ]
        + (
            PREDICTION_LENGTH
            - 1
        )
    )

    # ========================================================
    # SENSOR ID
    # ========================================================

    prediction_index[
        "sensor_id"
    ] = (
        prediction_index[
            "series_id"
        ]
        .astype(str)
        .str
        .split("::")
        .str[0]
        .astype(int)
    )

    # ========================================================
    # ORIGINAL PHASE 4 TEST TARGETS
    # ========================================================

    test_eval = (
        test.copy()
    )

    test_eval[
        "origin_tft_time_idx"
    ] = (
        time_index(
            test_eval[
                "timestamp"
            ]
        )
    )

    # target_pm25_3h stored at timestamp t means:
    #
    # actual PM2.5 at t + 3 hours.
    #
    # This matches decoder step #12.

    lookup_columns = [
        "timestamp",
        "sensor_id",
        "continuous_segment_id",
        "origin_tft_time_idx",
        "target_pm25_3h",
    ]

    merged = (
        prediction_index
        .merge(

            test_eval[
                lookup_columns
            ],

            on=[
                "sensor_id",
                "origin_tft_time_idx",
            ],

            how=
                "inner",
        )
    )

    merged = (
        merged
        .rename(
            columns={

                "target_pm25_3h":
                    "actual",

                "timestamp":
                    "forecast_origin_timestamp",
            }
        )
    )

    # ========================================================
    # REMOVE INVALID ROWS
    # ========================================================

    merged[
        "predicted"
    ] = pd.to_numeric(
        merged[
            "predicted"
        ],
        errors="coerce",
    )

    merged[
        "actual"
    ] = pd.to_numeric(
        merged[
            "actual"
        ],
        errors="coerce",
    )

    merged = (
        merged[
            np.isfinite(
                merged[
                    "predicted"
                ]
            )
            &
            np.isfinite(
                merged[
                    "actual"
                ]
            )
        ]
        .copy()
    )

    if merged.empty:

        raise RuntimeError(
            "No causal TFT predictions aligned "
            "with Phase 4 3h TEST targets."
        )

    # ========================================================
    # TARGET TIMESTAMP FOR REPORTING
    # ========================================================

    merged[
        "forecast_target_timestamp"
    ] = (
        pd.to_datetime(
            merged[
                "forecast_origin_timestamp"
            ],
            utc=True,
        )
        + pd.Timedelta(
            hours=3
        )
    )

    # ========================================================
    # METRICS
    # ========================================================

    actual = (
        merged[
            "actual"
        ]
        .to_numpy(
            dtype=float
        )
    )

    predicted = (
        merged[
            "predicted"
        ]
        .to_numpy(
            dtype=float
        )
    )

    mae = (
        mean_absolute_error(
            actual,
            predicted,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )
    )

    absolute_errors = np.abs(
        actual
        - predicted
    )

    median_ae = float(
        np.median(
            absolute_errors
        )
    )

    r2 = float(
        r2_score(
            actual,
            predicted,
        )
    )

    # ========================================================
    # SENSOR METRICS
    # ========================================================

    sensor_results = {}

    for sensor_id, group in (
        merged.groupby(
            "sensor_id"
        )
    ):

        sensor_actual = (
            group[
                "actual"
            ]
            .to_numpy(float)
        )

        sensor_predicted = (
            group[
                "predicted"
            ]
            .to_numpy(float)
        )

        sensor_mae = (
            mean_absolute_error(
                sensor_actual,
                sensor_predicted,
            )
        )

        sensor_rmse = float(
            np.sqrt(
                mean_squared_error(
                    sensor_actual,
                    sensor_predicted,
                )
            )
        )

        sensor_results[
            str(
                int(sensor_id)
            )
        ] = {

            "rows":
                int(
                    len(group)
                ),

            "mae":
                float(
                    sensor_mae
                ),

            "rmse":
                sensor_rmse,
        }

    # ========================================================
    # HIGH-PM / P95 ANALYSIS
    # ========================================================

    p95_threshold = float(
        np.percentile(
            actual,
            95
        )
    )

    high_mask = (
        actual
        >= p95_threshold
    )

    p95_actual = (
        actual[
            high_mask
        ]
    )

    p95_predicted = (
        predicted[
            high_mask
        ]
    )

    p95_error = (
        p95_predicted
        - p95_actual
    )

    p95_mae = float(
        np.mean(
            np.abs(
                p95_error
            )
        )
    )

    p95_rmse = float(
        np.sqrt(
            np.mean(
                p95_error ** 2
            )
        )
    )

    under_mask = (
        p95_predicted
        < p95_actual
    )

    underprediction_rate = float(
        np.mean(
            under_mask
        )
    )

    if np.any(
        under_mask
    ):

        under_amounts = (
            p95_actual[
                under_mask
            ]
            - p95_predicted[
                under_mask
            ]
        )

        mean_underprediction = float(
            np.mean(
                under_amounts
            )
        )

        worst_underprediction = float(
            np.max(
                under_amounts
            )
        )

    else:

        mean_underprediction = 0.0
        worst_underprediction = 0.0

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    section(
        "CAUSAL TFT 3h RESULTS"
    )

    print(
        f"TEST rows : "
        f"{len(merged):,}"
    )

    print(
        f"MAE       : "
        f"{mae:.6f}"
    )

    print(
        f"RMSE      : "
        f"{rmse:.6f}"
    )

    print(
        f"Median AE : "
        f"{median_ae:.6f}"
    )

    print(
        f"R²        : "
        f"{r2:.6f}"
    )

    print()

    print(
        f"P95 threshold        : "
        f"{p95_threshold:.6f}"
    )

    print(
        f"P95 rows             : "
        f"{int(high_mask.sum()):,}"
    )

    print(
        f"P95 MAE              : "
        f"{p95_mae:.6f}"
    )

    print(
        f"P95 RMSE             : "
        f"{p95_rmse:.6f}"
    )

    print(
        f"P95 underprediction  : "
        f"{underprediction_rate * 100:.2f}%"
    )

    print(
        f"Mean underprediction : "
        f"{mean_underprediction:.6f}"
    )

    print(
        f"Worst underprediction: "
        f"{worst_underprediction:.6f}"
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    output_columns = [

        "forecast_origin_timestamp",

        "forecast_target_timestamp",

        "sensor_id",

        "continuous_segment_id",

        "origin_tft_time_idx",

        "target_tft_time_idx",

        "actual",

        "predicted",
    ]

    output = (
        merged[
            output_columns
        ]
        .sort_values(
            [
                "forecast_origin_timestamp",
                "sensor_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output[
        "error"
    ] = (
        output[
            "predicted"
        ]
        - output[
            "actual"
        ]
    )

    output[
        "absolute_error"
    ] = np.abs(
        output[
            "error"
        ]
    )

    output.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "project":
            "AirAware",

        "model":
            "tft_causal",

        "horizon":
            "3h",

        "causal":
            True,

        "real_time_ready_design":
            True,

        "encoder_length":
            ENCODER_LENGTH,

        "encoder_hours":
            (
                ENCODER_LENGTH
                * SAMPLING_MINUTES
                / 60
            ),

        "prediction_length":
            PREDICTION_LENGTH,

        "prediction_hours":
            (
                PREDICTION_LENGTH
                * SAMPLING_MINUTES
                / 60
            ),

        "sampling_minutes":
            SAMPLING_MINUTES,

        "train_rows":
            int(
                len(train)
            ),

        "validation_rows":
            int(
                len(validation)
            ),

        "final_train_rows":
            int(
                len(final_train)
            ),

        "original_test_rows":
            int(
                len(test)
            ),

        "evaluated_test_rows":
            int(
                len(merged)
            ),

        "metrics": {

            "mae":
                float(
                    mae
                ),

            "rmse":
                float(
                    rmse
                ),

            "median_absolute_error":
                float(
                    median_ae
                ),

            "r2":
                float(
                    r2
                ),
        },

        "high_pm_p95": {

            "threshold":
                p95_threshold,

            "rows":
                int(
                    high_mask.sum()
                ),

            "mae":
                p95_mae,

            "rmse":
                p95_rmse,

            "underprediction_rate":
                underprediction_rate,

            "mean_underprediction":
                mean_underprediction,

            "worst_underprediction":
                worst_underprediction,
        },

        "sensor_metrics":
            sensor_results,

        "alignment": {

            "decoder_step_1":
                "forecast origin + 15 minutes",

            "decoder_step_12":
                "forecast origin + 3 hours",

            "forecast_origin_formula":
                "first_decoder_tft_time_idx - 1",

            "three_hour_prediction":
                "decoder step 12",
        },

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "preprocessing":
            str(
                PREPROCESSING_PATH
            ),

        "prediction_file":
            str(
                PREDICTION_PATH
            ),
    }

    RESULT_PATH.write_text(

        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),

        encoding=
            "utf-8",
    )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "FILES CREATED"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        PREPROCESSING_PATH
    )

    print(
        PREDICTION_PATH
    )

    print(
        RESULT_PATH
    )

    section(
        "CAUSAL TFT 3h COMPLETE"
    )


if __name__ == "__main__":

    main()