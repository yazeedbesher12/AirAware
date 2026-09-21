from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# AirAware - Ensemble Phase
# Step 1: Load and Align Model Predictions
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# عدّل فقط هذه المسارات إذا ملفاتك موجودة بمكان مختلف
OOF_FILE = BASE_DIR / "stacking_oof_predictions.csv"
TEST_FILE = BASE_DIR / "test_predictions_all_models.csv"

OUTPUT_DIR = BASE_DIR / "ensemble" / "aligned_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper Functions
# ============================================================

def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def find_prediction_columns(df):
    """
    Detect model prediction columns automatically.

    Expected examples:
        prediction_xgboost
        prediction_lightgbm
        prediction_random_forest
        prediction_lstm
        prediction_gru
        prediction_tft
        prediction_prophet
        prediction_persistence

    Also supports:
        pred_xgboost
        xgboost_prediction
    """

    prediction_columns = []

    ignore_columns = {
        "actual",
        "target",
        "horizon",
        "timestamp",
        "sensor_id",
        "city",
        "fold_id",
        "split",
        "pollutant",
    }

    for col in df.columns:
        lower = col.lower()

        if lower in ignore_columns:
            continue

        if (
            lower.startswith("prediction_")
            or lower.startswith("pred_")
            or lower.endswith("_prediction")
            or lower.endswith("_pred")
        ):
            prediction_columns.append(col)

    return prediction_columns


def detect_actual_column(df):
    candidates = [
        "actual",
        "y_true",
        "target_actual",
        "actual_value",
        "true_value",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find the actual/ground-truth column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def normalize_horizon(value):
    """
    Converts horizon formats to:
        1h
        3h
        6h

    Supports:
        1
        3
        6
        '1h'
        '3h'
        '6h'
        '1 hour'
        '3 hours'
    """

    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    mappings = {
        "1": "1h",
        "1h": "1h",
        "1hr": "1h",
        "1 hour": "1h",
        "1 hours": "1h",

        "3": "3h",
        "3h": "3h",
        "3hr": "3h",
        "3 hour": "3h",
        "3 hours": "3h",

        "6": "6h",
        "6h": "6h",
        "6hr": "6h",
        "6 hour": "6h",
        "6 hours": "6h",
    }

    return mappings.get(text, text)


def prepare_dataframe(df, name):
    print_section(f"PREPARING {name}")

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns found:")
    for col in df.columns:
        print(f"  - {col}")

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" not in df.columns:
        raise ValueError(
            f"{name}: required column 'timestamp' was not found."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    invalid_ts = df["timestamp"].isna().sum()

    print(f"\nInvalid timestamps: {invalid_ts:,}")

    if invalid_ts > 0:
        print("Dropping rows with invalid timestamps.")
        df = df.dropna(subset=["timestamp"]).copy()

    # --------------------------------------------------------
    # Horizon
    # --------------------------------------------------------

    if "horizon" not in df.columns:
        raise ValueError(
            f"{name}: required column 'horizon' was not found."
        )

    df["horizon"] = df["horizon"].apply(normalize_horizon)

    print("\nHorizons found:")
    print(df["horizon"].value_counts(dropna=False).to_string())

    # --------------------------------------------------------
    # Actual column
    # --------------------------------------------------------

    actual_column = detect_actual_column(df)

    print(f"\nActual column detected: {actual_column}")

    if actual_column != "actual":
        df = df.rename(columns={actual_column: "actual"})

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Prediction columns
    # --------------------------------------------------------

    prediction_columns = find_prediction_columns(df)

    if not prediction_columns:
        raise ValueError(
            f"{name}: no prediction columns were detected.\n"
            f"Columns: {list(df.columns)}"
        )

    print("\nPrediction columns detected:")

    for col in prediction_columns:
        print(f"  - {col}")

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    sort_columns = []

    if "sensor_id" in df.columns:
        sort_columns.append("sensor_id")

    sort_columns.append("timestamp")

    df = df.sort_values(sort_columns).reset_index(drop=True)

    return df, prediction_columns


def align_horizon(df, horizon, prediction_columns, dataset_name):
    """
    Keep only observations where:
      - actual exists
      - all selected model predictions exist

    This guarantees fair model comparison.
    """

    part = df[df["horizon"] == horizon].copy()

    if part.empty:
        print(f"\nWARNING: No {horizon} rows found in {dataset_name}.")
        return None

    print_section(f"{dataset_name} — {horizon}")

    print(f"Original rows: {len(part):,}")

    before = len(part)

    required = ["actual"] + prediction_columns

    missing_summary = (
        part[required]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nMissing values before alignment:")

    print(missing_summary.to_string())

    # Fair comparison:
    # All included models must have prediction for same observation.
    aligned = part.dropna(
        subset=required
    ).copy()

    after = len(aligned)

    removed = before - after

    print(f"\nAligned rows : {after:,}")
    print(f"Removed rows : {removed:,}")

    if before > 0:
        print(
            f"Coverage     : "
            f"{(after / before) * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicate_keys = ["timestamp"]

    if "sensor_id" in aligned.columns:
        duplicate_keys.append("sensor_id")

    if "target" in aligned.columns:
        duplicate_keys.append("target")

    duplicate_count = aligned.duplicated(
        subset=duplicate_keys,
        keep=False
    ).sum()

    print(f"Potential duplicate observations: {duplicate_count:,}")

    # --------------------------------------------------------
    # Actual summary
    # --------------------------------------------------------

    if len(aligned) > 0:

        print("\nActual target summary:")

        print(
            aligned["actual"]
            .describe()
            .to_string()
        )

        print("\nTime range:")

        print(
            f"Start: {aligned['timestamp'].min()}"
        )

        print(
            f"End  : {aligned['timestamp'].max()}"
        )

        if "sensor_id" in aligned.columns:

            print("\nRows per sensor:")

            print(
                aligned["sensor_id"]
                .value_counts()
                .to_string()
            )

    return aligned


def save_aligned_files(
    df,
    prediction_columns,
    dataset_name
):

    horizons = ["1h", "3h", "6h"]

    results = {}

    for horizon in horizons:

        aligned = align_horizon(
            df=df,
            horizon=horizon,
            prediction_columns=prediction_columns,
            dataset_name=dataset_name,
        )

        if aligned is None:
            continue

        output_file = (
            OUTPUT_DIR
            / f"{dataset_name.lower()}_aligned_{horizon}.csv"
        )

        aligned.to_csv(
            output_file,
            index=False
        )

        results[horizon] = {
            "rows": len(aligned),
            "file": output_file,
        }

        print(
            f"\nSaved:\n{output_file}"
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "AIRAWARE — LOAD & ALIGN MODEL PREDICTIONS"
    )

    print(f"Project directory:\n{BASE_DIR}")

    print(f"\nOOF file:\n{OOF_FILE}")

    print(f"\nTEST file:\n{TEST_FILE}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not OOF_FILE.exists():
        raise FileNotFoundError(
            f"\nOOF file not found:\n{OOF_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"\nTest file not found:\n{TEST_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print_section("LOADING FILES")

    oof = pd.read_csv(OOF_FILE)
    test = pd.read_csv(TEST_FILE)

    print(
        f"OOF loaded : {len(oof):,} rows"
    )

    print(
        f"TEST loaded: {len(test):,} rows"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    oof, oof_prediction_columns = prepare_dataframe(
        oof,
        "OOF"
    )

    test, test_prediction_columns = prepare_dataframe(
        test,
        "TEST"
    )

    # --------------------------------------------------------
    # Compare prediction columns
    # --------------------------------------------------------

    print_section("MODEL COLUMN COMPARISON")

    print("OOF prediction columns:")

    for col in oof_prediction_columns:
        print(f"  {col}")

    print("\nTEST prediction columns:")

    for col in test_prediction_columns:
        print(f"  {col}")

    common_prediction_columns = sorted(
        set(oof_prediction_columns)
        .intersection(test_prediction_columns)
    )

    print("\nCommon model prediction columns:")

    for col in common_prediction_columns:
        print(f"  {col}")

    if not common_prediction_columns:

        raise ValueError(
            "No common model prediction columns exist "
            "between OOF and TEST files."
        )

    missing_from_oof = (
        set(test_prediction_columns)
        - set(oof_prediction_columns)
    )

    missing_from_test = (
        set(oof_prediction_columns)
        - set(test_prediction_columns)
    )

    if missing_from_oof:

        print(
            "\nWARNING — present in TEST but missing from OOF:"
        )

        for col in sorted(missing_from_oof):
            print(f"  {col}")

    if missing_from_test:

        print(
            "\nWARNING — present in OOF but missing from TEST:"
        )

        for col in sorted(missing_from_test):
            print(f"  {col}")

    # --------------------------------------------------------
    # Only common models should be used for fair comparison
    # --------------------------------------------------------

    print_section("ALIGNING OOF")

    oof_results = save_aligned_files(
        oof,
        common_prediction_columns,
        "OOF"
    )

    print_section("ALIGNING TEST")

    test_results = save_aligned_files(
        test,
        common_prediction_columns,
        "TEST"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_section("FINAL SUMMARY")

    print("Common models:")

    for model in common_prediction_columns:
        print(f"  - {model}")

    print("\nOOF:")

    for horizon, info in oof_results.items():

        print(
            f"  {horizon}: "
            f"{info['rows']:,} aligned rows"
        )

    print("\nTEST:")

    for horizon, info in test_results.items():

        print(
            f"  {horizon}: "
            f"{info['rows']:,} aligned rows"
        )

    print(
        "\nAligned files saved to:"
    )

    print(OUTPUT_DIR)

    print_section(
        "STEP 1 COMPLETE"
    )

    print(
        "Do NOT run Weighted Ensemble yet.\n"
        "Review these results first."
    )


if __name__ == "__main__":
    main()