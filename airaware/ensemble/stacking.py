from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


# ============================================================
# AirAware
# Step 3: Stacking
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OOF_FILE = BASE_DIR / "stacking_oof_predictions.csv"
TEST_FILE = BASE_DIR / "test_predictions_all_models.csv"

OUTPUT_DIR = BASE_DIR / "ensemble" / "stacking_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = OUTPUT_DIR / "stacking_results.json"
RESULTS_CSV = OUTPUT_DIR / "stacking_candidate_results.csv"


# ============================================================
# Candidate Base Models
# ============================================================

CANDIDATES = {
    "1h": [
        [
            "prediction_gru",
            "prediction_lightgbm",
            "prediction_tft",
        ],
        [
            "prediction_gru",
            "prediction_lightgbm",
            "prediction_tft",
            "prediction_random_forest",
        ],
    ],

    "3h": [
        [
            "prediction_tft",
            "prediction_lightgbm",
            "prediction_gru",
        ],
        [
            "prediction_tft",
            "prediction_lightgbm",
            "prediction_gru",
            "prediction_random_forest",
        ],
    ],

    "6h": [
        [
            "prediction_tft",
            "prediction_gru",
            "prediction_random_forest",
        ],
        [
            "prediction_tft",
            "prediction_gru",
            "prediction_random_forest",
            "prediction_lightgbm",
        ],
    ],
}


# ============================================================
# Meta Models
# ============================================================

META_MODELS = {
    "linear_regression": LinearRegression(),

    "ridge_0.1": Ridge(
        alpha=0.1
    ),

    "ridge_1.0": Ridge(
        alpha=1.0
    ),

    "ridge_10.0": Ridge(
        alpha=10.0
    ),

    "elasticnet": ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=10000,
        random_state=42,
    ),
}


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


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

    return mapping.get(text, text)


def load_data(path):
    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df["horizon"] = df["horizon"].apply(
        normalize_horizon
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    prediction_cols = [
        col
        for col in df.columns
        if col.startswith("prediction_")
    ]

    for col in prediction_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "timestamp",
            "actual",
            "horizon",
        ]
    ).copy()

    sort_cols = []

    if "sensor_id" in df.columns:
        sort_cols.append("sensor_id")

    sort_cols.append("timestamp")

    return df.sort_values(
        sort_cols
    ).reset_index(drop=True)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(actual, predicted):
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
        "n": int(len(actual)),

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
        ) if len(actual) > 1 else None,
    }


# ============================================================
# Strict Candidate Alignment
# ============================================================

def align_candidate(df, models):
    required = [
        "actual",
        *models,
    ]

    return df.dropna(
        subset=required
    ).copy()


# ============================================================
# Chronological Meta Train / Validation Split
# ============================================================

def chronological_meta_split(
    df,
    validation_fraction=0.25,
):
    """
    Important:
    Do NOT random shuffle.

    Earlier OOF rows:
        meta-training

    Later OOF rows:
        meta-validation
    """

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    split_index = int(
        len(df)
        * (1 - validation_fraction)
    )

    if split_index <= 0 or split_index >= len(df):
        raise ValueError(
            "Not enough samples for meta train/validation split."
        )

    train = df.iloc[
        :split_index
    ].copy()

    val = df.iloc[
        split_index:
    ].copy()

    return train, val


# ============================================================
# Fit + Evaluate One Meta Model
# ============================================================

def evaluate_meta_model(
    meta_name,
    meta_model,
    models,
    oof_data,
):
    aligned = align_candidate(
        oof_data,
        models,
    )

    if len(aligned) < 100:
        print(
            f"Skipping {meta_name}: "
            f"only {len(aligned)} aligned rows."
        )
        return None

    meta_train, meta_val = chronological_meta_split(
        aligned,
        validation_fraction=0.25,
    )

    X_train = meta_train[
        models
    ].to_numpy(dtype=float)

    y_train = meta_train[
        "actual"
    ].to_numpy(dtype=float)

    X_val = meta_val[
        models
    ].to_numpy(dtype=float)

    y_val = meta_val[
        "actual"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Fit Meta Model
    # --------------------------------------------------------

    meta_model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Validation Prediction
    # --------------------------------------------------------

    val_prediction = meta_model.predict(
        X_val
    )

    val_metrics = calculate_metrics(
        y_val,
        val_prediction,
    )

    coefficients = None
    intercept = None

    if hasattr(meta_model, "coef_"):
        coefficients = {
            model: float(coef)
            for model, coef
            in zip(
                models,
                np.ravel(meta_model.coef_),
            )
        }

    if hasattr(meta_model, "intercept_"):
        try:
            intercept = float(
                np.ravel(
                    np.asarray(
                        meta_model.intercept_
                    )
                )[0]
            )
        except Exception:
            intercept = None

    return {
        "meta_model": meta_name,
        "models": models,
        "aligned_rows": int(len(aligned)),
        "meta_train_rows": int(len(meta_train)),
        "meta_validation_rows": int(len(meta_val)),
        "validation_metrics": val_metrics,
        "coefficients": coefficients,
        "intercept": intercept,
    }


# ============================================================
# Fit Final Meta Model on Full OOF
# ============================================================

def fit_final_meta_model(
    meta_name,
    models,
    oof_data,
):
    aligned = align_candidate(
        oof_data,
        models,
    )

    # Create fresh model
    if meta_name == "linear_regression":
        model = LinearRegression()

    elif meta_name == "ridge_0.1":
        model = Ridge(alpha=0.1)

    elif meta_name == "ridge_1.0":
        model = Ridge(alpha=1.0)

    elif meta_name == "ridge_10.0":
        model = Ridge(alpha=10.0)

    elif meta_name == "elasticnet":
        model = ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=42,
        )

    else:
        raise ValueError(
            f"Unknown meta-model: {meta_name}"
        )

    X = aligned[
        models
    ].to_numpy(dtype=float)

    y = aligned[
        "actual"
    ].to_numpy(dtype=float)

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# TEST Evaluation
# ============================================================

def evaluate_on_test(
    trained_meta_model,
    models,
    test_data,
):
    """
    For now we evaluate only rows where all selected
    base-model predictions are available.

    This keeps Stacking evaluation scientifically clean.

    Missing-prediction fallback for real-time deployment
    will be handled later after final model selection.
    """

    aligned = align_candidate(
        test_data,
        models,
    )

    if len(aligned) == 0:
        return None, None

    X_test = aligned[
        models
    ].to_numpy(dtype=float)

    y_test = aligned[
        "actual"
    ].to_numpy(dtype=float)

    prediction = trained_meta_model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        prediction,
    )

    output = aligned.copy()

    output[
        "stacking_prediction"
    ] = prediction

    output[
        "stacking_absolute_error"
    ] = np.abs(
        output["actual"]
        - output["stacking_prediction"]
    )

    return metrics, output


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "AIRAWARE — STEP 3: STACKING"
    )

    if not OOF_FILE.exists():
        raise FileNotFoundError(
            f"OOF file missing:\n{OOF_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"TEST file missing:\n{TEST_FILE}"
        )

    oof = load_data(
        OOF_FILE
    )

    test = load_data(
        TEST_FILE
    )

    print(
        f"OOF rows : {len(oof):,}"
    )

    print(
        f"TEST rows: {len(test):,}"
    )

    all_candidate_results = []
    final_results = {}

    # ========================================================
    # Horizon Loop
    # ========================================================

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        print_section(
            f"HORIZON {horizon}"
        )

        oof_h = oof[
            oof["horizon"] == horizon
        ].copy()

        test_h = test[
            test["horizon"] == horizon
        ].copy()

        print(
            f"OOF rows : {len(oof_h):,}"
        )

        print(
            f"TEST rows: {len(test_h):,}"
        )

        horizon_results = []

        # ====================================================
        # Candidate Base Model Groups
        # ====================================================

        for models in CANDIDATES[horizon]:

            print("\nBase models:")
            print(
                " + ".join(
                    m.replace(
                        "prediction_",
                        "",
                    )
                    for m in models
                )
            )

            missing_columns = [
                model
                for model in models
                if (
                    model not in oof.columns
                    or model not in test.columns
                )
            ]

            if missing_columns:
                print(
                    "Skipping because columns are missing:"
                )

                for col in missing_columns:
                    print(
                        f"  {col}"
                    )

                continue

            # ================================================
            # Try Meta Models
            # ================================================

            for meta_name, meta_model in (
                META_MODELS.items()
            ):

                result = evaluate_meta_model(
                    meta_name=meta_name,
                    meta_model=meta_model,
                    models=models,
                    oof_data=oof_h,
                )

                if result is None:
                    continue

                result[
                    "horizon"
                ] = horizon

                horizon_results.append(
                    result
                )

                all_candidate_results.append(
                    result
                )

                metrics = result[
                    "validation_metrics"
                ]

                print(
                    f"  {meta_name:<20} "
                    f"RMSE={metrics['rmse']:.4f} "
                    f"MAE={metrics['mae']:.4f} "
                    f"R²={metrics['r2']:.4f}"
                )

        # ====================================================
        # Select Best using META VALIDATION ONLY
        # ====================================================

        if not horizon_results:
            continue

        best = min(
            horizon_results,
            key=lambda x:
                x[
                    "validation_metrics"
                ]["rmse"],
        )

        print_section(
            f"BEST STACKING — {horizon}"
        )

        print(
            f"Meta Model: "
            f"{best['meta_model']}"
        )

        print(
            "Base Models:"
        )

        for model in best["models"]:
            print(
                f"  - "
                f"{model.replace('prediction_', '')}"
            )

        print(
            f"\nMeta-validation RMSE: "
            f"{best['validation_metrics']['rmse']:.4f}"
        )

        print(
            f"Meta-validation MAE : "
            f"{best['validation_metrics']['mae']:.4f}"
        )

        print(
            f"Meta-validation R²  : "
            f"{best['validation_metrics']['r2']:.4f}"
        )

        # ====================================================
        # Refit chosen meta-model on FULL OOF
        # ====================================================

        final_meta_model = fit_final_meta_model(
            meta_name=best[
                "meta_model"
            ],
            models=best[
                "models"
            ],
            oof_data=oof_h,
        )

        # ====================================================
        # Final Test
        # ====================================================

        test_metrics, test_predictions = (
            evaluate_on_test(
                trained_meta_model=
                    final_meta_model,
                models=best[
                    "models"
                ],
                test_data=test_h,
            )
        )

        if test_metrics is None:
            print(
                "No aligned TEST rows."
            )
            continue

        print(
            f"\nTEST RMSE: "
            f"{test_metrics['rmse']:.4f}"
        )

        print(
            f"TEST MAE : "
            f"{test_metrics['mae']:.4f}"
        )

        print(
            f"TEST R²  : "
            f"{test_metrics['r2']:.4f}"
        )

        # ====================================================
        # Save Test Predictions
        # ====================================================

        prediction_file = (
            OUTPUT_DIR
            / f"stacking_{horizon}_test_predictions.csv"
        )

        test_predictions.to_csv(
            prediction_file,
            index=False,
        )

        coefficients = None
        intercept = None

        if hasattr(
            final_meta_model,
            "coef_",
        ):
            coefficients = {
                model.replace(
                    "prediction_",
                    "",
                ): float(coef)
                for model, coef
                in zip(
                    best["models"],
                    np.ravel(
                        final_meta_model.coef_
                    ),
                )
            }

        if hasattr(
            final_meta_model,
            "intercept_",
        ):
            try:
                intercept = float(
                    np.ravel(
                        np.asarray(
                            final_meta_model.intercept_
                        )
                    )[0]
                )
            except Exception:
                intercept = None

        final_results[
            horizon
        ] = {
            "base_models": [
                model.replace(
                    "prediction_",
                    "",
                )
                for model in best[
                    "models"
                ]
            ],

            "meta_model":
                best["meta_model"],

            "meta_validation_metrics":
                best[
                    "validation_metrics"
                ],

            "test_metrics":
                test_metrics,

            "final_coefficients":
                coefficients,

            "final_intercept":
                intercept,

            "test_prediction_file":
                str(prediction_file),
        }

    # ========================================================
    # Save Candidate CSV
    # ========================================================

    flat_rows = []

    for result in all_candidate_results:

        flat_rows.append({
            "horizon":
                result["horizon"],

            "base_models":
                " + ".join(
                    model.replace(
                        "prediction_",
                        "",
                    )
                    for model in result[
                        "models"
                    ]
                ),

            "meta_model":
                result[
                    "meta_model"
                ],

            "aligned_rows":
                result[
                    "aligned_rows"
                ],

            "meta_train_rows":
                result[
                    "meta_train_rows"
                ],

            "meta_validation_rows":
                result[
                    "meta_validation_rows"
                ],

            "validation_rmse":
                result[
                    "validation_metrics"
                ]["rmse"],

            "validation_mae":
                result[
                    "validation_metrics"
                ]["mae"],

            "validation_r2":
                result[
                    "validation_metrics"
                ]["r2"],
        })

    pd.DataFrame(
        flat_rows
    ).to_csv(
        RESULTS_CSV,
        index=False,
    )

    # ========================================================
    # Save JSON
    # ========================================================

    payload = {
        "project":
            "AirAware",

        "phase":
            "Stacking",

        "selection_rule":
            "Meta-model and base-model combination selected "
            "using chronological meta-validation inside OOF only. "
            "Final TEST is evaluated only after selection.",

        "candidate_results":
            all_candidate_results,

        "final_by_horizon":
            final_results,
    }

    with open(
        RESULTS_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Final Summary
    # ========================================================

    print_section(
        "STACKING FINAL SUMMARY"
    )

    for horizon, result in (
        final_results.items()
    ):

        print(
            f"\n{horizon}"
        )

        print(
            f"Base Models: "
            f"{', '.join(result['base_models'])}"
        )

        print(
            f"Meta Model: "
            f"{result['meta_model']}"
        )

        print(
            f"Meta-validation RMSE: "
            f"{result['meta_validation_metrics']['rmse']:.4f}"
        )

        print(
            f"TEST RMSE: "
            f"{result['test_metrics']['rmse']:.4f}"
        )

        print(
            f"TEST MAE: "
            f"{result['test_metrics']['mae']:.4f}"
        )

        print(
            f"TEST R²: "
            f"{result['test_metrics']['r2']:.4f}"
        )

        if result[
            "final_coefficients"
        ]:

            print(
                "Final coefficients:"
            )

            for model, coef in (
                result[
                    "final_coefficients"
                ].items()
            ):

                print(
                    f"  {model:<20} "
                    f"{coef:.4f}"
                )

        if result[
            "final_intercept"
        ] is not None:

            print(
                f"Intercept: "
                f"{result['final_intercept']:.4f}"
            )

    print_section(
        "FILES CREATED"
    )

    print(
        RESULTS_JSON
    )

    print(
        RESULTS_CSV
    )

    print(
        OUTPUT_DIR
    )

    print_section(
        "STEP 3 COMPLETE"
    )

    print(
        "STOP HERE.\n"
        "Do not choose the final real-time architecture yet.\n"
        "Next step is comparing:\n"
        "Best Single vs Weighted Ensemble vs Stacking."
    )


if __name__ == "__main__":
    main()