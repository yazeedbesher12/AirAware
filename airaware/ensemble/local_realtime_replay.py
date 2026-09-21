from pathlib import Path
from collections import deque
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# AirAware
# Step 5A-1: Local Real-Time Replay
#
# IMPORTANT:
# This script replays SAVED TEST predictions.
# It does NOT yet run LightGBM/TFT inference from model files.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = (
    BASE_DIR
    / "test_predictions_all_models.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ensemble"
    / "realtime_replay_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "local_realtime_forecasts.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "local_realtime_replay_summary.json"
)


# ============================================================
# Final Forecast Architecture
# ============================================================

FINAL_MODELS = {
    "1h": {
        "method": "lightgbm",
        "prediction_column": "prediction_lightgbm",
    },

    "3h": {
        "method": "tft",
        "prediction_column": "prediction_tft",
    },

    "6h": {
        "method": "lightgbm",
        "prediction_column": "prediction_lightgbm",
    },
}


# ============================================================
# Replay Configuration
# ============================================================

# Number of recent readings kept per sensor.
# At ~15-minute sampling:
# 96 readings ≈ 24 hours.
HISTORY_SIZE = 96

# Set True only if you want visible waiting between readings.
REAL_TIME_DELAY_ENABLED = False

# Example:
# 0.1 sec per replay reading.
REPLAY_DELAY_SECONDS = 0.10


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def normalize_horizon(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    mapping = {
        "1": "1h",
        "1h": "1h",

        "3": "3h",
        "3h": "3h",

        "6": "6h",
        "6h": "6h",
    }

    return mapping.get(
        value,
        value,
    )


def load_test_data():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing TEST file:\n{TEST_FILE}"
        )

    df = pd.read_csv(
        TEST_FILE
    )

    required = [
        "timestamp",
        "sensor_id",
        "actual",
        "horizon",
        "prediction_lightgbm",
        "prediction_tft",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns:\n{missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df["horizon"] = (
        df["horizon"]
        .apply(
            normalize_horizon
        )
    )

    numeric_cols = [
        "actual",
        "prediction_lightgbm",
        "prediction_tft",
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "timestamp",
            "sensor_id",
            "actual",
            "horizon",
        ]
    ).copy()

    return df


# ============================================================
# Convert forecast rows into observation timeline
# ============================================================

def build_observation_stream(df):
    """
    test_predictions_all_models.csv contains one row per
    target horizon.

    For simulated live replay we need one observation event
    per sensor + timestamp.

    We therefore create a unique chronological stream.
    """

    cols = [
        "timestamp",
        "sensor_id",
        "actual",
    ]

    if "city" in df.columns:
        cols.append(
            "city"
        )

    observations = (
        df[cols]
        .drop_duplicates(
            subset=[
                "timestamp",
                "sensor_id",
            ]
        )
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

    return observations


# ============================================================
# Forecast lookup
# ============================================================

def build_forecast_lookup(df):
    """
    Creates:
        (timestamp, sensor_id, horizon)
            ->
        row

    These are PRECOMPUTED TEST predictions.
    """

    lookup = {}

    for _, row in df.iterrows():

        key = (
            row["timestamp"],
            str(row["sensor_id"]),
            row["horizon"],
        )

        lookup[key] = row

    return lookup


# ============================================================
# Sensor history
# ============================================================

class SensorHistory:

    def __init__(
        self,
        maxlen=HISTORY_SIZE,
    ):
        self.buffers = {}

        self.maxlen = maxlen

    def add_reading(
        self,
        sensor_id,
        timestamp,
        value,
    ):

        sensor_id = str(
            sensor_id
        )

        if sensor_id not in self.buffers:

            self.buffers[
                sensor_id
            ] = deque(
                maxlen=self.maxlen
            )

        self.buffers[
            sensor_id
        ].append(
            {
                "timestamp": timestamp,
                "pm25": float(value),
            }
        )

    def count(
        self,
        sensor_id,
    ):

        return len(
            self.buffers.get(
                str(sensor_id),
                [],
            )
        )

    def latest(
        self,
        sensor_id,
    ):

        buffer = self.buffers.get(
            str(sensor_id)
        )

        if not buffer:
            return None

        return buffer[-1]

    def recent_values(
        self,
        sensor_id,
        n=4,
    ):

        buffer = self.buffers.get(
            str(sensor_id)
        )

        if not buffer:
            return []

        return [
            item["pm25"]
            for item
            in list(buffer)[-n:]
        ]


# ============================================================
# Local Replay Forecast Engine
# ============================================================

class ReplayForecastEngine:

    def __init__(
        self,
        forecast_lookup,
    ):

        self.lookup = (
            forecast_lookup
        )

    def predict(
        self,
        timestamp,
        sensor_id,
    ):
        """
        Returns final 1h/3h/6h forecasts.

        Currently these come from SAVED TEST predictions.

        In Step 5A-2 this exact function will be replaced
        with real LightGBM/TFT inference.
        """

        outputs = {}

        for horizon, config in (
            FINAL_MODELS.items()
        ):

            key = (
                timestamp,
                str(sensor_id),
                horizon,
            )

            source_row = (
                self.lookup.get(
                    key
                )
            )

            if source_row is None:

                outputs[
                    horizon
                ] = {
                    "status":
                        "missing",

                    "model":
                        config[
                            "method"
                        ],

                    "prediction":
                        None,
                }

                continue

            prediction_col = (
                config[
                    "prediction_column"
                ]
            )

            prediction = (
                source_row.get(
                    prediction_col
                )
            )

            if pd.isna(
                prediction
            ):

                outputs[
                    horizon
                ] = {
                    "status":
                        "missing",

                    "model":
                        config[
                            "method"
                        ],

                    "prediction":
                        None,
                }

                continue

            outputs[
                horizon
            ] = {
                "status":
                    "ok",

                "model":
                    config[
                        "method"
                    ],

                "prediction":
                    float(
                        prediction
                    ),

                "actual_target":
                    float(
                        source_row[
                            "actual"
                        ]
                    ),
            }

        return outputs


# ============================================================
# Console display
# ============================================================

def display_forecast(
    timestamp,
    sensor_id,
    current_pm25,
    forecasts,
    history_count,
):

    print(
        "\n"
        + "-" * 90
    )

    print(
        f"Timestamp : {timestamp}"
    )

    print(
        f"Sensor    : {sensor_id}"
    )

    print(
        f"Current   : {current_pm25:.2f} µg/m³"
    )

    print(
        f"History   : {history_count} readings"
    )

    print(
        "\nForecasts:"
    )

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        result = forecasts[
            horizon
        ]

        if (
            result[
                "status"
            ]
            != "ok"
        ):

            print(
                f"  {horizon:<3} "
                f"→ MISSING "
                f"({result['model']})"
            )

            continue

        print(
            f"  {horizon:<3} "
            f"→ "
            f"{result['prediction']:.2f} µg/m³ "
            f"[{result['model']}]"
        )


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "AIRAWARE — STEP 5A-1: LOCAL REAL-TIME REPLAY"
    )

    df = load_test_data()

    print(
        f"Forecast rows loaded : "
        f"{len(df):,}"
    )

    stream = (
        build_observation_stream(
            df
        )
    )

    print(
        f"Unique live readings : "
        f"{len(stream):,}"
    )

    lookup = (
        build_forecast_lookup(
            df
        )
    )

    print(
        f"Forecast lookup rows  : "
        f"{len(lookup):,}"
    )

    history = (
        SensorHistory()
    )

    forecast_engine = (
        ReplayForecastEngine(
            lookup
        )
    )

    output_rows = []

    forecast_counts = {
        "1h": 0,
        "3h": 0,
        "6h": 0,
    }

    missing_counts = {
        "1h": 0,
        "3h": 0,
        "6h": 0,
    }

    sensor_counts = {}

    # ========================================================
    # Replay
    # ========================================================

    for index, row in (
        stream.iterrows()
    ):

        timestamp = (
            row[
                "timestamp"
            ]
        )

        sensor_id = str(
            row[
                "sensor_id"
            ]
        )

        current_pm25 = float(
            row[
                "actual"
            ]
        )

        # ----------------------------------------------------
        # Simulate new sensor reading arrival
        # ----------------------------------------------------

        history.add_reading(
            sensor_id=sensor_id,
            timestamp=timestamp,
            value=current_pm25,
        )

        sensor_counts[
            sensor_id
        ] = (
            sensor_counts.get(
                sensor_id,
                0,
            )
            + 1
        )

        # ----------------------------------------------------
        # Forecast
        # ----------------------------------------------------

        forecasts = (
            forecast_engine.predict(
                timestamp=timestamp,
                sensor_id=sensor_id,
            )
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display_forecast(
            timestamp=timestamp,
            sensor_id=sensor_id,
            current_pm25=current_pm25,
            forecasts=forecasts,
            history_count=history.count(
                sensor_id
            ),
        )

        # ----------------------------------------------------
        # Save outputs
        # ----------------------------------------------------

        output = {
            "timestamp":
                timestamp,

            "sensor_id":
                sensor_id,

            "current_pm25":
                current_pm25,

            "history_count":
                history.count(
                    sensor_id
                ),
        }

        if "city" in row.index:

            output[
                "city"
            ] = row[
                "city"
            ]

        for horizon in [
            "1h",
            "3h",
            "6h",
        ]:

            result = (
                forecasts[
                    horizon
                ]
            )

            output[
                f"forecast_{horizon}"
            ] = (
                result[
                    "prediction"
                ]
            )

            output[
                f"model_{horizon}"
            ] = (
                result[
                    "model"
                ]
            )

            output[
                f"status_{horizon}"
            ] = (
                result[
                    "status"
                ]
            )

            if (
                result[
                    "status"
                ]
                == "ok"
            ):

                forecast_counts[
                    horizon
                ] += 1

            else:

                missing_counts[
                    horizon
                ] += 1

        output_rows.append(
            output
        )

        # ----------------------------------------------------
        # Optional replay delay
        # ----------------------------------------------------

        if REAL_TIME_DELAY_ENABLED:

            time.sleep(
                REPLAY_DELAY_SECONDS
            )

    # ========================================================
    # Save
    # ========================================================

    output_df = pd.DataFrame(
        output_rows
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "project":
            "AirAware",

        "phase":
            "Step 5A-1 Local Replay",

        "mode":
            "precomputed_prediction_replay",

        "important_note":
            (
                "This replay validates runtime orchestration "
                "using saved TEST predictions. It does not "
                "yet execute LightGBM or TFT model files."
            ),

        "final_models": {
            horizon:
                config[
                    "method"
                ]
            for horizon, config
            in FINAL_MODELS.items()
        },

        "unique_readings":
            int(
                len(stream)
            ),

        "history_size":
            HISTORY_SIZE,

        "forecast_counts":
            forecast_counts,

        "missing_counts":
            missing_counts,

        "readings_per_sensor":
            sensor_counts,

        "output_file":
            str(
                OUTPUT_FILE
            ),
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Final console summary
    # ========================================================

    print_section(
        "LOCAL REPLAY SUMMARY"
    )

    print(
        f"Total readings: "
        f"{len(stream):,}"
    )

    print(
        "\nFinal architecture:"
    )

    for horizon, config in (
        FINAL_MODELS.items()
    ):

        print(
            f"  {horizon:<3} "
            f"→ {config['method']}"
        )

    print(
        "\nForecast availability:"
    )

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        print(
            f"  {horizon}: "
            f"{forecast_counts[horizon]:,} OK | "
            f"{missing_counts[horizon]:,} missing"
        )

    print(
        "\nPer sensor:"
    )

    for sensor, count in (
        sensor_counts.items()
    ):

        print(
            f"  {sensor}: "
            f"{count:,} readings"
        )

    print_section(
        "FILES CREATED"
    )

    print(
        OUTPUT_FILE
    )

    print(
        SUMMARY_FILE
    )

    print_section(
        "STEP 5A-1 COMPLETE"
    )

    print(
        "Next:\n"
        "Step 5A-2 = replace saved predictions with "
        "real local LightGBM + TFT inference."
    )


if __name__ == "__main__":
    main()