from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ============================================================
# AirAware
# Step 4: Final Comparison
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "test_predictions_all_models.csv"

WEIGHTED_RESULTS = (
    BASE_DIR
    / "ensemble"
    / "weighted_outputs"
    / "weighted_ensemble_results.json"
)

STACKING_RESULTS = (
    BASE_DIR
    / "ensemble"
    / "stacking_outputs"
    / "stacking_results.json"
)

ADVANCED_STACKING_RESULTS = (
    BASE_DIR
    / "ensemble"
    / "stacking_advanced_outputs"
    / "stacking_advanced_results.json"
)

STACKING_DIR = (
    BASE_DIR
    / "ensemble"
    / "stacking_outputs"
)

ADVANCED_STACKING_DIR = (
    BASE_DIR
    / "ensemble"
    / "stacking_advanced_outputs"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ensemble"
    / "final_comparison_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_CSV = (
    OUTPUT_DIR
    / "final_model_comparison.csv"
)

PER_SENSOR_CSV = (
    OUTPUT_DIR
    / "final_per_sensor_comparison.csv"
)

HIGH_PM_CSV = (
    OUTPUT_DIR
    / "high_pm25_comparison.csv"
)

FINAL_JSON = (
    OUTPUT_DIR
    / "final_model_registry.json"
)

FINAL_REPORT_JSON = (
    OUTPUT_DIR
    / "final_comparison_results.json"
)


# ============================================================
# Candidate single models
# ============================================================

SINGLE_MODELS = [
    "prediction_gru",
    "prediction_lightgbm",
    "prediction_lstm",
    "prediction_random_forest",
    "prediction_tft",
    "prediction_xgboost",
    "prediction_prophet",
    "prediction_persistence",
]


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)


def normalize_horizon(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    mapping = {
        "1": "1h",
        "1h": "1h",
        "1hr": "1h",
        "1 hour": "1h",

        "3": "3h",
        "3h": "3h",
        "3hr": "3h",
        "3 hour": "3h",

        "6": "6h",
        "6h": "6h",
        "6hr": "6h",
        "6 hour": "6h",
    }

    return mapping.get(
        text,
        text,
    )


def load_test():
    df = pd.read_csv(
        TEST_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df["horizon"] = df[
        "horizon"
    ].apply(
        normalize_horizon
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    prediction_cols = [
        col
        for col in df.columns
        if col.startswith(
            "prediction_"
        )
    ]

    for col in prediction_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df.dropna(
        subset=[
            "timestamp",
            "actual",
            "horizon",
        ]
    ).copy()


def read_json(path):
    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def calculate_metrics(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "median_ae": None,
            "r2": None,
        }

    return {
        "n": int(
            len(actual)
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),

        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),

        "median_ae": float(
            median_absolute_error(
                actual,
                predicted,
            )
        ),

        "r2": float(
            r2_score(
                actual,
                predicted,
            )
        )
        if len(actual) > 1
        else None,
    }


# ============================================================
# Event Metrics
# ============================================================

def calculate_event_metrics(
    df,
    prediction_column,
):
    """
    Uses existing Phase 3 pollution event data
    if available.

    If health prediction threshold information is
    available, derive event prediction from that.

    Otherwise returns None.
    """

    if (
        "pollution_event_actual"
        not in df.columns
    ):
        return None

    actual_event = pd.to_numeric(
        df[
            "pollution_event_actual"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Try to find matching health prediction column
    # for single models.
    # --------------------------------------------------------

    event_pred = None

    if prediction_column.startswith(
        "prediction_"
    ):

        model_name = prediction_column.replace(
            "prediction_",
            "",
        )

        health_col = (
            f"health_prediction_{model_name}"
        )

        if health_col in df.columns:

            health_values = pd.to_numeric(
                df[
                    health_col
                ],
                errors="coerce",
            )

            # This assumes the existing health_prediction
            # column is already binary/event-compatible.
            unique_values = set(
                health_values.dropna().unique()
            )

            if unique_values.issubset(
                {0, 1}
            ):
                event_pred = health_values

    if event_pred is None:
        return None

    mask = (
        actual_event.notna()
        & event_pred.notna()
    )

    y_true = actual_event[
        mask
    ].astype(int)

    y_pred = event_pred[
        mask
    ].astype(int)

    if len(y_true) == 0:
        return None

    return {
        "event_n":
            int(len(y_true)),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "false_negatives":
            int(
                (
                    (y_true == 1)
                    & (y_pred == 0)
                ).sum()
            ),

        "false_positives":
            int(
                (
                    (y_true == 0)
                    & (y_pred == 1)
                ).sum()
            ),
    }


# ============================================================
# Weighted Ensemble Prediction Loader
# ============================================================

def load_weighted_prediction(
    horizon,
):
    weighted_json = read_json(
        WEIGHTED_RESULTS
    )

    if weighted_json is None:
        return None

    best = (
        weighted_json
        .get(
            "provisional_best_by_horizon",
            {}
        )
        .get(
            horizon
        )
    )

    if not best:
        return None

    models = best[
        "models"
    ]

    candidate_name = "__".join(
        model.replace(
            "prediction_",
            "",
        )
        for model in models
    )

    path = (
        BASE_DIR
        / "ensemble"
        / "weighted_outputs"
        / (
            f"{horizon}_"
            f"{candidate_name}_"
            f"test_predictions.csv"
        )
    )

    if not path.exists():
        return None

    df = pd.read_csv(
        path
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df["horizon"] = df[
        "horizon"
    ].apply(
        normalize_horizon
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df[
        "weighted_prediction"
    ] = pd.to_numeric(
        df[
            "weighted_prediction"
        ],
        errors="coerce",
    )

    return df


# ============================================================
# Original Stacking Loader
# ============================================================

def load_original_stacking(
    horizon,
):
    path = (
        STACKING_DIR
        / (
            f"stacking_{horizon}_"
            f"test_predictions.csv"
        )
    )

    if not path.exists():
        return None

    df = pd.read_csv(
        path
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df[
        "stacking_prediction"
    ] = pd.to_numeric(
        df[
            "stacking_prediction"
        ],
        errors="coerce",
    )

    return df


# ============================================================
# Advanced Stacking Loader
# ============================================================

def load_advanced_stacking(
    horizon,
):
    path = (
        ADVANCED_STACKING_DIR
        / (
            f"advanced_stacking_{horizon}_"
            f"test_predictions.csv"
        )
    )

    if not path.exists():
        return None

    df = pd.read_csv(
        path
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df[
        "stacking_prediction"
    ] = pd.to_numeric(
        df[
            "stacking_prediction"
        ],
        errors="coerce",
    )

    return df


# ============================================================
# Build Common Fair Comparison Table
# ============================================================

def build_common_table(
    test_h,
    horizon,
):
    """
    Build one table containing all candidate predictions.

    Final fair comparison will use rows where all
    included prediction methods exist.

    This ensures apples-to-apples comparison.
    """

    base_cols = [
        "timestamp",
        "sensor_id",
        "city",
        "actual",
    ]

    if (
        "pollution_event_actual"
        in test_h.columns
    ):
        base_cols.append(
            "pollution_event_actual"
        )

    common = test_h[
        base_cols
    ].copy()

    # --------------------------------------------------------
    # Single model predictions
    # --------------------------------------------------------

    for col in SINGLE_MODELS:

        if col in test_h.columns:

            temp = test_h[
                [
                    "timestamp",
                    "sensor_id",
                    col,
                ]
            ].copy()

            common = common.merge(
                temp,
                on=[
                    "timestamp",
                    "sensor_id",
                ],
                how="left",
            )

    # --------------------------------------------------------
    # Weighted
    # --------------------------------------------------------

    weighted = load_weighted_prediction(
        horizon
    )

    if weighted is not None:

        temp = weighted[
            [
                "timestamp",
                "sensor_id",
                "weighted_prediction",
            ]
        ].copy()

        temp = temp.rename(
            columns={
                "weighted_prediction":
                    "prediction_weighted"
            }
        )

        common = common.merge(
            temp,
            on=[
                "timestamp",
                "sensor_id",
            ],
            how="left",
        )

    # --------------------------------------------------------
    # Original Stacking
    # --------------------------------------------------------

    original = load_original_stacking(
        horizon
    )

    if original is not None:

        temp = original[
            [
                "timestamp",
                "sensor_id",
                "stacking_prediction",
            ]
        ].copy()

        temp = temp.rename(
            columns={
                "stacking_prediction":
                    "prediction_stacking_original"
            }
        )

        common = common.merge(
            temp,
            on=[
                "timestamp",
                "sensor_id",
            ],
            how="left",
        )

    # --------------------------------------------------------
    # Advanced Stacking
    # --------------------------------------------------------

    advanced = load_advanced_stacking(
        horizon
    )

    if advanced is not None:

        temp = advanced[
            [
                "timestamp",
                "sensor_id",
                "stacking_prediction",
            ]
        ].copy()

        temp = temp.rename(
            columns={
                "stacking_prediction":
                    "prediction_stacking_advanced"
            }
        )

        common = common.merge(
            temp,
            on=[
                "timestamp",
                "sensor_id",
            ],
            how="left",
        )

    return common


# ============================================================
# Compare all methods
# ============================================================

def compare_methods(
    common,
    horizon,
):
    prediction_cols = [
        col
        for col in common.columns
        if col.startswith(
            "prediction_"
        )
    ]

    # --------------------------------------------------------
    # Strict common rows across ALL methods
    # --------------------------------------------------------

    strict = common.dropna(
        subset=[
            "actual",
            *prediction_cols,
        ]
    ).copy()

    print(
        f"\n{horizon} strict common rows: "
        f"{len(strict):,}"
    )

    rows = []

    for prediction_col in (
        prediction_cols
    ):

        metrics = calculate_metrics(
            strict[
                "actual"
            ],
            strict[
                prediction_col
            ],
        )

        rows.append({
            "horizon":
                horizon,

            "method":
                prediction_col.replace(
                    "prediction_",
                    "",
                ),

            **metrics,
        })

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        "rmse"
    ).reset_index(drop=True)

    results[
        "rank_rmse"
    ] = (
        np.arange(
            len(results)
        )
        + 1
    )

    return results, strict


# ============================================================
# Best Single
# ============================================================

def find_best_single(
    results,
):
    single_names = {
        x.replace(
            "prediction_",
            "",
        )
        for x in SINGLE_MODELS
    }

    subset = results[
        results[
            "method"
        ].isin(
            single_names
        )
    ].copy()

    if subset.empty:
        return None

    return subset.sort_values(
        "rmse"
    ).iloc[0]


# ============================================================
# Per Sensor
# ============================================================

def evaluate_per_sensor(
    strict,
    horizon,
):
    prediction_cols = [
        col
        for col in strict.columns
        if col.startswith(
            "prediction_"
        )
    ]

    rows = []

    for sensor_id, group in (
        strict.groupby(
            "sensor_id"
        )
    ):

        for col in prediction_cols:

            m = calculate_metrics(
                group[
                    "actual"
                ],
                group[
                    col
                ],
            )

            rows.append({
                "horizon":
                    horizon,

                "sensor_id":
                    sensor_id,

                "method":
                    col.replace(
                        "prediction_",
                        "",
                    ),

                **m,
            })

    return rows


# ============================================================
# High PM2.5
# ============================================================

def evaluate_high_pm(
    strict,
    horizon,
):
    prediction_cols = [
        col
        for col in strict.columns
        if col.startswith(
            "prediction_"
        )
    ]

    rows = []

    for percentile in [
        90,
        95,
    ]:

        threshold = np.percentile(
            strict[
                "actual"
            ],
            percentile,
        )

        high = strict[
            strict[
                "actual"
            ] >= threshold
        ].copy()

        for col in prediction_cols:

            m = calculate_metrics(
                high[
                    "actual"
                ],
                high[
                    col
                ],
            )

            rows.append({
                "horizon":
                    horizon,

                "percentile":
                    percentile,

                "threshold":
                    float(
                        threshold
                    ),

                "method":
                    col.replace(
                        "prediction_",
                        "",
                    ),

                **m,
            })

    return rows


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "AIRAWARE — STEP 4: FINAL MODEL COMPARISON"
    )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            TEST_FILE
        )

    test = load_test()

    print(
        f"TEST rows: "
        f"{len(test):,}"
    )

    all_results = []
    per_sensor_rows = []
    high_pm_rows = []

    registry = {}

    comparison_details = {}

    # ========================================================
    # Horizon loop
    # ========================================================

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        print_section(
            f"FINAL COMPARISON — {horizon}"
        )

        test_h = test[
            test[
                "horizon"
            ] == horizon
        ].copy()

        common = build_common_table(
            test_h,
            horizon,
        )

        results, strict = (
            compare_methods(
                common,
                horizon,
            )
        )

        print(
            "\nRanking:"
        )

        print(
            results[
                [
                    "rank_rmse",
                    "method",
                    "n",
                    "rmse",
                    "mae",
                    "median_ae",
                    "r2",
                ]
            ].to_string(
                index=False
            )
        )

        all_results.append(
            results
        )

        # ----------------------------------------------------
        # Best single
        # ----------------------------------------------------

        best_single = find_best_single(
            results
        )

        # ----------------------------------------------------
        # Winner
        # ----------------------------------------------------

        winner = results.iloc[
            0
        ]

        winner_method = winner[
            "method"
        ]

        print(
            f"\nFINAL WINNER {horizon}: "
            f"{winner_method}"
        )

        print(
            f"RMSE = "
            f"{winner['rmse']:.4f}"
        )

        print(
            f"MAE  = "
            f"{winner['mae']:.4f}"
        )

        print(
            f"R²   = "
            f"{winner['r2']:.4f}"
        )

        # ----------------------------------------------------
        # Improvement vs best single
        # ----------------------------------------------------

        improvement = None

        if (
            best_single is not None
            and best_single[
                "rmse"
            ] > 0
        ):

            improvement = (
                (
                    best_single[
                        "rmse"
                    ]
                    - winner[
                        "rmse"
                    ]
                )
                / best_single[
                    "rmse"
                ]
                * 100
            )

            print(
                f"\nBest Single: "
                f"{best_single['method']}"
            )

            print(
                f"Best Single RMSE: "
                f"{best_single['rmse']:.4f}"
            )

            print(
                f"Improvement vs Best Single: "
                f"{improvement:.2f}%"
            )

        # ----------------------------------------------------
        # Per sensor
        # ----------------------------------------------------

        per_sensor_rows.extend(
            evaluate_per_sensor(
                strict,
                horizon,
            )
        )

        # ----------------------------------------------------
        # High PM
        # ----------------------------------------------------

        high_pm_rows.extend(
            evaluate_high_pm(
                strict,
                horizon,
            )
        )

        # ----------------------------------------------------
        # Registry
        # ----------------------------------------------------

        registry[
            horizon
        ] = {
            "method":
                winner_method,

            "rmse":
                float(
                    winner[
                        "rmse"
                    ]
                ),

            "mae":
                float(
                    winner[
                        "mae"
                    ]
                ),

            "r2":
                float(
                    winner[
                        "r2"
                    ]
                ),

            "test_samples":
                int(
                    winner[
                        "n"
                    ]
                ),

            "best_single":
                (
                    best_single[
                        "method"
                    ]
                    if best_single
                    is not None
                    else None
                ),

            "improvement_vs_best_single_percent":
                (
                    float(
                        improvement
                    )
                    if improvement
                    is not None
                    else None
                ),
        }

        comparison_details[
            horizon
        ] = results.to_dict(
            orient="records"
        )

        # ----------------------------------------------------
        # Save strict common dataset
        # ----------------------------------------------------

        strict.to_csv(
            OUTPUT_DIR
            / (
                f"final_common_test_"
                f"{horizon}.csv"
            ),
            index=False,
        )

    # ========================================================
    # Combine results
    # ========================================================

    final_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    final_df.to_csv(
        FINAL_CSV,
        index=False,
    )

    pd.DataFrame(
        per_sensor_rows
    ).to_csv(
        PER_SENSOR_CSV,
        index=False,
    )

    pd.DataFrame(
        high_pm_rows
    ).to_csv(
        HIGH_PM_CSV,
        index=False,
    )

    # ========================================================
    # Final Registry
    # ========================================================

    registry_payload = {
        "project":
            "AirAware",

        "selection_basis":
            "Final comparison on identical unseen TEST timestamps.",

        "forecast_target":
            "PM2.5 concentration",

        "horizons":
            registry,
    }

    with open(
        FINAL_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            registry_payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Detailed report
    # ========================================================

    report_payload = {
        "project":
            "AirAware",

        "phase":
            "Final Model Comparison",

        "important_rule":
            "All methods were compared on identical TEST rows "
            "within each forecast horizon.",

        "comparison":
            comparison_details,

        "final_registry":
            registry,
    }

    with open(
        FINAL_REPORT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report_payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Final summary
    # ========================================================

    print_section(
        "FINAL AIRAWARE FORECAST ARCHITECTURE"
    )

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        info = registry[
            horizon
        ]

        print(
            f"\n{horizon}"
        )

        print(
            f"Winner : "
            f"{info['method']}"
        )

        print(
            f"RMSE   : "
            f"{info['rmse']:.4f}"
        )

        print(
            f"MAE    : "
            f"{info['mae']:.4f}"
        )

        print(
            f"R²     : "
            f"{info['r2']:.4f}"
        )

        print(
            f"Best single: "
            f"{info['best_single']}"
        )

        print(
            f"Improvement: "
            f"{info['improvement_vs_best_single_percent']:.2f}%"
        )

    print_section(
        "FILES CREATED"
    )

    print(
        FINAL_CSV
    )

    print(
        PER_SENSOR_CSV
    )

    print(
        HIGH_PM_CSV
    )

    print(
        FINAL_JSON
    )

    print(
        FINAL_REPORT_JSON
    )

    print_section(
        "STEP 4 COMPLETE"
    )

    print(
        "STOP HERE.\n"
        "Review the final winners before implementing "
        "the live real-time forecast engine."
    )


if __name__ == "__main__":
    main()