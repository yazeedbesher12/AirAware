from pathlib import Path

import pandas as pd

from app.ml.forecast_engine import ForecastEngine


# ============================================================
# PATHS
# ============================================================

BACKEND = Path(__file__).resolve().parent

FEATURE_FILE = (
    BACKEND
    / "outputs"
    / "AirAware_ML_Features.csv"
)


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "AIRAWARE — FORECAST ENGINE TEST"
    )

    # ========================================================
    # LOAD HISTORICAL ENGINEERED FEATURES
    # ========================================================

    df = pd.read_csv(
        FEATURE_FILE,
        low_memory=False,
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    # ========================================================
    # SELECT ONE REAL HISTORICAL ROW
    # ========================================================
    #
    # Sensor 1
    # TEST period starts at 2026-08-20.
    #
    # We take a row from the middle of the available TEST
    # period instead of the final timestamp.
    # ========================================================

    rows = df[
        (
            df["sensor_id"]
            == 1
        )
        &
        (
            df["timestamp"]
            >= pd.Timestamp(
                "2026-08-20T00:00:00Z"
            )
        )
    ].copy()

    if rows.empty:

        raise RuntimeError(
            "No suitable Sensor 1 TEST row found."
        )

    rows = rows.sort_values(
        "timestamp"
    )

    row = rows.iloc[
        len(rows) // 2
    ]

    # ForecastEngine expects exactly
    # one row as a DataFrame.
    feature_row = pd.DataFrame(
        [row]
    )

    # ========================================================
    # PRINT INPUT
    # ========================================================

    print()
    print("INPUT")
    print("-" * 80)

    print(
        f"Timestamp    : "
        f"{row['timestamp']}"
    )

    print(
        f"Sensor ID    : "
        f"{int(row['sensor_id'])}"
    )

    print(
        f"Current PM2.5: "
        f"{float(row['pm25']):.3f} µg/m³"
    )

    if (
        "temperature" in row.index
        and pd.notna(
            row["temperature"]
        )
    ):

        print(
            f"Temperature  : "
            f"{float(row['temperature']):.3f}"
        )

    if (
        "humidity" in row.index
        and pd.notna(
            row["humidity"]
        )
    ):

        print(
            f"Humidity     : "
            f"{float(row['humidity']):.3f}"
        )

    # ========================================================
    # LOAD FORECAST ENGINE
    # ========================================================

    print()
    print(
        "Loading forecast engine..."
    )

    engine = ForecastEngine()

    print(
        "Models loaded successfully."
    )

    # ========================================================
    # TRUE MODEL INFERENCE
    # ========================================================

    forecasts = engine.predict(
        feature_row
    )

    # ========================================================
    # PRINT FORECASTS
    # ========================================================

    section(
        "FORECAST RESULTS"
    )

    for horizon in (
        "1h",
        "3h",
        "6h",
    ):

        result = forecasts[
            horizon
        ]

        print()
        print(
            f"{horizon} FORECAST"
        )

        print(
            "-" * 60
        )

        # ----------------------------------------------------
        # Main LightGBM PM2.5 prediction
        # ----------------------------------------------------

        print(
            "PM2.5 prediction      : "
            f"{result['pm25']['forecast']:.3f} µg/m³"
        )

        # ----------------------------------------------------
        # OOF-based safety layer
        # ----------------------------------------------------

        print(
            "Safety upper estimate: "
            f"{result['pm25']['safety_upper_estimate']:.3f} µg/m³"
        )

        print(
            "Safety margin        : +"
            f"{result['pm25']['safety_margin']:.3f} µg/m³"
        )

        print(
        "High-PM Risk         : "
        f"{result['high_pm_risk']['level']}"
        )

        print(
            "P90 threshold        : "
            f"{result['high_pm_risk']['p90_threshold']:.3f} µg/m³"
        )

        print(
            "P95 threshold        : "
            f"{result['high_pm_risk']['p95_threshold']:.3f} µg/m³"
        )

        print(
            "Risk trigger         : "
            f"{result['high_pm_risk']['trigger']}"
        )


            # ----------------------------------------------------
        # Predicted future PM2.5 24h rolling average
        # ----------------------------------------------------

        print(
            "Future 24h average   : "
            f"{result['future_24h_average']['forecast']:.3f} µg/m³"
        )

        # ----------------------------------------------------
        # WHO interpretation
        # ----------------------------------------------------

        print(
            "WHO status           : "
            f"{result['who']['status']}"
        )

        print(
            "WHO guideline        : "
            f"{result['who']['guideline']:.3f} µg/m³"
        )

        print(
            "WHO ratio            : "
            f"{result['who']['ratio_to_guideline']:.2f}x"
        )

        # ----------------------------------------------------
        # US EPA PM2.5-derived AQI
        # ----------------------------------------------------

        print(
            "PM2.5-derived EPA AQI: "
            f"{result['epa_aqi']['value']}"
        )

        print(
            "EPA category         : "
            f"{result['epa_aqi']['category']}"
        )

        print(
            "EPA AQI basis        : "
            f"{result['epa_aqi']['basis']}"
        )

        # ----------------------------------------------------
        # Uncertainty method
        # ----------------------------------------------------

        print(
            "Safety method        : "
            f"{result['uncertainty']['method']}"
        )

        print(
            "Safety quantile      : "
            f"{result['uncertainty']['quantile']:.2f}"
        )

    # ========================================================
    # COMPACT SUMMARY
    # ========================================================

    section(
        "COMPACT FORECAST SUMMARY"
    )

    print(
        f"{'Horizon':<10}"
        f"{'PM2.5':>12}"
        f"{'Safety Upper':>16}"
        f"{'24h Avg':>12}"
        f"{'EPA AQI':>10}"
        f"{'WHO':>24}"
    )

    print(
        "-" * 84
    )

    for horizon in (
        "1h",
        "3h",
        "6h",
    ):

        result = forecasts[
            horizon
        ]

        print(
            f"{horizon:<10}"
            f"{result['pm25']['forecast']:>12.3f}"
            f"{result['pm25']['safety_upper_estimate']:>16.3f}"
            f"{result['future_24h_average']['forecast']:>12.3f}"
            f"{result['epa_aqi']['value']:>10}"
            f"{result['who']['status']:>24}"
        )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "FORECAST ENGINE TEST COMPLETE"
    )


if __name__ == "__main__":

    main()