from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


# ============================================================
# AirAware
# Step 2: Weighted Ensemble
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OOF_FILE = BASE_DIR / "stacking_oof_predictions.csv"
TEST_FILE = BASE_DIR / "test_predictions_all_models.csv"

OUTPUT_DIR = BASE_DIR / "ensemble" / "weighted_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = OUTPUT_DIR / "weighted_ensemble_results.json"
CANDIDATES_CSV = OUTPUT_DIR / "weighted_candidate_results.csv"


# ============================================================
# Candidate Models
# ============================================================

CANDIDATES = {
    "1h": [
        ["prediction_gru", "prediction_lightgbm"],
        ["prediction_gru", "prediction_tft"],
        ["prediction_lightgbm", "prediction_tft"],
        ["prediction_gru", "prediction_lightgbm", "prediction_tft"],
        [
            "prediction_gru",
            "prediction_lightgbm",
            "prediction_tft",
            "prediction_random_forest",
        ],
        [
            "prediction_gru",
            "prediction_lightgbm",
            "prediction_tft",
            "prediction_xgboost",
        ],
    ],

    "3h": [
        ["prediction_tft", "prediction_lightgbm"],
        ["prediction_tft", "prediction_gru"],
        ["prediction_tft", "prediction_random_forest"],
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
        ["prediction_tft", "prediction_gru"],
        ["prediction_tft", "prediction_random_forest"],
        ["prediction_tft", "prediction_lightgbm"],
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
# Utilities
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

    return mapping.get(text, text)


def load_data(path):
    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df["horizon"] = df["horizon"].apply(normalize_horizon)

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
        subset=["timestamp", "actual", "horizon"]
    ).copy()

    return df


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

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

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    median_ae = median_absolute_error(
        actual,
        predicted,
    )

    if len(actual) > 1:
        r2 = r2_score(
            actual,
            predicted,
        )
    else:
        r2 = None

    return {
        "n": int(len(actual)),
        "rmse": float(rmse),
        "mae": float(mae),
        "median_ae": float(median_ae),
        "r2": (
            float(r2)
            if r2 is not None
            else None
        ),
    }


# ============================================================
# Equal Ensemble
# ============================================================

def equal_average(df, models):
    values = df[models].to_numpy(dtype=float)

    with np.errstate(all="ignore"):
        predictions = np.nanmean(
            values,
            axis=1,
        )

    return predictions


# ============================================================
# Weighted Ensemble
# ============================================================

def strict_alignment(df, models):
    required = ["actual"] + models

    aligned = df.dropna(
        subset=required
    ).copy()

    return aligned


def weighted_prediction_strict(matrix, weights):
    return np.dot(
        matrix,
        weights,
    )


def objective_rmse(weights, matrix, actual):
    prediction = weighted_prediction_strict(
        matrix,
        weights,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction,
        )
    )

    return rmse


def optimize_weights(df, models):
    aligned = strict_alignment(
        df,
        models,
    )

    if len(aligned) == 0:
        raise ValueError(
            f"No aligned OOF rows for models: {models}"
        )

    matrix = aligned[
        models
    ].to_numpy(dtype=float)

    actual = aligned[
        "actual"
    ].to_numpy(dtype=float)

    n_models = len(models)

    initial_weights = np.ones(
        n_models
    ) / n_models

    bounds = [
        (0.0, 1.0)
        for _ in range(n_models)
    ]

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    }

    result = minimize(
        objective_rmse,
        initial_weights,
        args=(matrix, actual),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        print(
            "\nWARNING:"
            f" Weight optimization did not fully converge:\n"
            f"{result.message}"
        )

    weights = result.x

    weights = np.clip(
        weights,
        0,
        None,
    )

    weights = weights / weights.sum()

    prediction = np.dot(
        matrix,
        weights,
    )

    metrics = calculate_metrics(
        actual,
        prediction,
    )

    weight_dict = {
        model: float(weight)
        for model, weight
        in zip(models, weights)
    }

    return {
        "weights": weight_dict,
        "metrics": metrics,
        "aligned_rows": len(aligned),
    }


# ============================================================
# Dynamic Weighted Ensemble
# ============================================================

def dynamic_weighted_prediction(
    row,
    weights,
    minimum_models=1,
):
    """
    Real-time behavior:

    - Ignore models with optimized weight ~= 0.
    - If an ACTIVE model prediction is missing,
      continue using the available active models.
    - Renormalize the remaining weights.
    """

    available_predictions = {}
    available_weights = {}

    missing_models = []
    inactive_models = []

    for model, weight in weights.items():

        # ----------------------------------------------------
        # IMPORTANT FIX:
        #
        # A model with zero optimized weight is INACTIVE.
        # It must NOT be treated as missing.
        # ----------------------------------------------------

        if weight <= 1e-12:
            inactive_models.append(model)
            continue

        value = row.get(model)

        if (
            pd.notna(value)
            and np.isfinite(value)
        ):
            available_predictions[model] = float(value)
            available_weights[model] = float(weight)

        else:
            missing_models.append(model)

    # --------------------------------------------------------
    # Not enough active model predictions
    # --------------------------------------------------------

    if len(available_predictions) < minimum_models:
        return {
            "prediction": np.nan,
            "models_available": [],
            "models_missing": missing_models,
            "models_inactive": inactive_models,
            "effective_weights": {},
            "fallback_used": True,
        }

    total_weight = sum(
        available_weights.values()
    )

    # --------------------------------------------------------
    # Safety fallback:
    # if somehow available active weights sum to zero,
    # use equal weights among available models.
    # --------------------------------------------------------

    if total_weight <= 1e-12:

        n = len(available_predictions)

        effective_weights = {
            model: 1.0 / n
            for model in available_predictions
        }

        fallback_used = True

    else:

        effective_weights = {
            model: weight / total_weight
            for model, weight
            in available_weights.items()
        }

        # True fallback only when an ACTIVE model
        # is actually missing.
        fallback_used = (
            len(missing_models) > 0
        )

    prediction = sum(
        available_predictions[model]
        * effective_weights[model]
        for model in available_predictions
    )

    return {
        "prediction": float(prediction),
        "models_available": list(
            available_predictions.keys()
        ),
        "models_missing": missing_models,
        "models_inactive": inactive_models,
        "effective_weights": effective_weights,
        "fallback_used": fallback_used,
    }


def apply_dynamic_ensemble(
    df,
    weights,
):
    output = df.copy()

    results = []

    for _, row in output.iterrows():

        result = dynamic_weighted_prediction(
            row,
            weights,
            minimum_models=1,
        )

        results.append(result)

    output[
        "weighted_prediction"
    ] = [
        x["prediction"]
        for x in results
    ]

    output[
        "models_available"
    ] = [
        ",".join(
            x["models_available"]
        )
        for x in results
    ]

    output[
        "models_missing"
    ] = [
        ",".join(
            x["models_missing"]
        )
        for x in results
    ]

    output[
        "models_inactive"
    ] = [
        ",".join(
            x["models_inactive"]
        )
        for x in results
    ]

    output[
        "models_available_count"
    ] = [
        len(
            x["models_available"]
        )
        for x in results
    ]

    output[
        "fallback_used"
    ] = [
        x["fallback_used"]
        for x in results
    ]

    output[
        "effective_weights"
    ] = [
        json.dumps(
            x["effective_weights"]
        )
        for x in results
    ]

    return output


# ============================================================
# Evaluate One Candidate
# ============================================================

def evaluate_candidate(
    oof_horizon,
    test_horizon,
    models,
    horizon,
):
    print("\nCandidate:")
    print(" + ".join(models))

    # --------------------------------------------------------
    # 1. Strict OOF alignment for weight optimization
    # --------------------------------------------------------

    aligned_oof = strict_alignment(
        oof_horizon,
        models,
    )

    if len(aligned_oof) == 0:
        print("No valid OOF rows. Skipping.")
        return None

    print(
        f"OOF strict rows: {len(aligned_oof):,}"
    )

    # --------------------------------------------------------
    # 2. Equal Average OOF
    # --------------------------------------------------------

    equal_pred_oof = equal_average(
        aligned_oof,
        models,
    )

    equal_oof_metrics = calculate_metrics(
        aligned_oof["actual"],
        equal_pred_oof,
    )

    # --------------------------------------------------------
    # 3. Optimize Weighted Ensemble using OOF ONLY
    # --------------------------------------------------------

    optimized = optimize_weights(
        oof_horizon,
        models,
    )

    weights = optimized[
        "weights"
    ]

    weighted_oof_metrics = optimized[
        "metrics"
    ]

    print("\nOptimized weights:")

    for model, weight in weights.items():

        clean_name = model.replace(
            "prediction_",
            "",
        )

        print(
            f"  {clean_name:<20} "
            f"{weight:.4f}"
        )

    print(
        f"\nOOF weighted RMSE: "
        f"{weighted_oof_metrics['rmse']:.4f}"
    )

    # --------------------------------------------------------
    # 4. Dynamic OOF evaluation
    # --------------------------------------------------------

    dynamic_oof = apply_dynamic_ensemble(
        oof_horizon,
        weights,
    )

    dynamic_oof_valid = dynamic_oof.dropna(
        subset=[
            "actual",
            "weighted_prediction",
        ]
    )

    dynamic_oof_metrics = calculate_metrics(
        dynamic_oof_valid["actual"],
        dynamic_oof_valid["weighted_prediction"],
    )

    # --------------------------------------------------------
    # 5. TEST
    #
    # Weights already frozen from OOF.
    # TEST is not used for optimization.
    # --------------------------------------------------------

    dynamic_test = apply_dynamic_ensemble(
        test_horizon,
        weights,
    )

    dynamic_test_valid = dynamic_test.dropna(
        subset=[
            "actual",
            "weighted_prediction",
        ]
    )

    test_metrics = calculate_metrics(
        dynamic_test_valid["actual"],
        dynamic_test_valid[
            "weighted_prediction"
        ],
    )

    # --------------------------------------------------------
    # Equal Average TEST
    # --------------------------------------------------------

    test_equal = test_horizon.copy()

    test_equal[
        "equal_prediction"
    ] = equal_average(
        test_equal,
        models,
    )

    equal_test_valid = test_equal.dropna(
        subset=[
            "actual",
            "equal_prediction",
        ]
    )

    equal_test_metrics = calculate_metrics(
        equal_test_valid["actual"],
        equal_test_valid[
            "equal_prediction"
        ],
    )

    # --------------------------------------------------------
    # Correct fallback statistics
    # --------------------------------------------------------

    fallback_count = int(
        dynamic_test_valid[
            "fallback_used"
        ].sum()
    )

    fallback_rate = (
        fallback_count
        / len(dynamic_test_valid)
        * 100
        if len(dynamic_test_valid)
        else 0
    )

    min_models_available = (
        int(
            dynamic_test_valid[
                "models_available_count"
            ].min()
        )
        if len(dynamic_test_valid)
        else 0
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    candidate_name = "__".join(
        model.replace(
            "prediction_",
            "",
        )
        for model in models
    )

    prediction_file = (
        OUTPUT_DIR
        / f"{horizon}_{candidate_name}_test_predictions.csv"
    )

    dynamic_test.to_csv(
        prediction_file,
        index=False,
    )

    print(
        f"TEST weighted RMSE: "
        f"{test_metrics['rmse']:.4f}"
    )

    print(
        f"TEST weighted MAE : "
        f"{test_metrics['mae']:.4f}"
    )

    print(
        f"Dynamic fallback rate: "
        f"{fallback_rate:.2f}%"
    )

    return {
        "horizon": horizon,

        "models": models,

        "weights": weights,

        "oof_strict_rows": int(
            len(aligned_oof)
        ),

        "oof_total_rows": int(
            len(oof_horizon)
        ),

        "equal_oof_metrics":
            equal_oof_metrics,

        "weighted_oof_metrics":
            weighted_oof_metrics,

        "dynamic_oof_metrics":
            dynamic_oof_metrics,

        "equal_test_metrics":
            equal_test_metrics,

        "weighted_test_metrics":
            test_metrics,

        "test_total_rows":
            int(len(test_horizon)),

        "test_valid_predictions":
            int(len(dynamic_test_valid)),

        "fallback_count":
            fallback_count,

        "fallback_rate_percent":
            float(fallback_rate),

        "minimum_models_available":
            min_models_available,

        "prediction_file":
            str(prediction_file),
    }


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "AIRAWARE — STEP 2: WEIGHTED ENSEMBLE"
    )

    print(
        f"OOF:\n{OOF_FILE}"
    )

    print(
        f"\nTEST:\n{TEST_FILE}"
    )

    if not OOF_FILE.exists():
        raise FileNotFoundError(
            f"Missing OOF file:\n{OOF_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing TEST file:\n{TEST_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    oof = load_data(
        OOF_FILE
    )

    test = load_data(
        TEST_FILE
    )

    print(
        f"\nOOF rows : {len(oof):,}"
    )

    print(
        f"TEST rows: {len(test):,}"
    )

    all_results = []

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

        oof_horizon = oof[
            oof["horizon"] == horizon
        ].copy()

        test_horizon = test[
            test["horizon"] == horizon
        ].copy()

        print(
            f"OOF rows : "
            f"{len(oof_horizon):,}"
        )

        print(
            f"TEST rows: "
            f"{len(test_horizon):,}"
        )

        candidates = CANDIDATES[
            horizon
        ]

        for models in candidates:

            missing_cols = [
                model
                for model in models
                if (
                    model not in oof.columns
                    or model not in test.columns
                )
            ]

            if missing_cols:

                print(
                    "\nSkipping candidate."
                )

                print(
                    "Missing columns:"
                )

                for col in missing_cols:
                    print(f"  {col}")

                continue

            result = evaluate_candidate(
                oof_horizon=oof_horizon,
                test_horizon=test_horizon,
                models=models,
                horizon=horizon,
            )

            if result is not None:
                all_results.append(
                    result
                )

    # ========================================================
    # Candidate Table
    # ========================================================

    flat_rows = []

    for result in all_results:

        row = {
            "horizon":
                result["horizon"],

            "models":
                " + ".join(
                    model.replace(
                        "prediction_",
                        "",
                    )
                    for model in result["models"]
                ),

            "oof_rows":
                result["oof_strict_rows"],

            "oof_rmse":
                result[
                    "weighted_oof_metrics"
                ]["rmse"],

            "oof_mae":
                result[
                    "weighted_oof_metrics"
                ]["mae"],

            "oof_r2":
                result[
                    "weighted_oof_metrics"
                ]["r2"],

            "test_rmse":
                result[
                    "weighted_test_metrics"
                ]["rmse"],

            "test_mae":
                result[
                    "weighted_test_metrics"
                ]["mae"],

            "test_r2":
                result[
                    "weighted_test_metrics"
                ]["r2"],

            "equal_test_rmse":
                result[
                    "equal_test_metrics"
                ]["rmse"],

            "fallback_rate_percent":
                result[
                    "fallback_rate_percent"
                ],

            "weights":
                json.dumps(
                    result["weights"]
                ),
        }

        flat_rows.append(
            row
        )

    result_df = pd.DataFrame(
        flat_rows
    )

    if not result_df.empty:

        result_df = result_df.sort_values(
            [
                "horizon",
                "oof_rmse",
            ],
            ascending=[
                True,
                True,
            ],
        )

        result_df.to_csv(
            CANDIDATES_CSV,
            index=False,
        )

    # ========================================================
    # Choose provisional best using OOF ONLY
    # ========================================================

    best_by_horizon = {}

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        candidates = [
            result
            for result in all_results
            if result["horizon"] == horizon
        ]

        if not candidates:
            continue

        best = min(
            candidates,
            key=lambda x:
                x[
                    "weighted_oof_metrics"
                ]["rmse"],
        )

        best_by_horizon[
            horizon
        ] = {
            "models":
                best["models"],

            "weights":
                best["weights"],

            "oof_metrics":
                best[
                    "weighted_oof_metrics"
                ],

            "test_metrics":
                best[
                    "weighted_test_metrics"
                ],

            "fallback_rate_percent":
                best[
                    "fallback_rate_percent"
                ],
        }

    # ========================================================
    # Save JSON
    # ========================================================

    final_json = {
        "project": "AirAware",
        "phase": "Weighted Ensemble",

        "selection_rule":
            "Candidate and weights selected using OOF RMSE only. "
            "TEST was never used for optimization.",

        "dynamic_missing_behavior":
            "Zero-weight models are treated as inactive. "
            "Only missing active model predictions trigger fallback. "
            "Available active model weights are renormalized.",

        "candidates":
            all_results,

        "provisional_best_by_horizon":
            best_by_horizon,
    }

    with open(
        RESULTS_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_json,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Print Summary
    # ========================================================

    print_section(
        "PROVISIONAL BEST WEIGHTED ENSEMBLES"
    )

    for horizon, result in (
        best_by_horizon.items()
    ):

        print(
            f"\n{horizon}"
        )

        print(
            "Models:"
        )

        for model in result[
            "models"
        ]:

            print(
                f"  - "
                f"{model.replace('prediction_', '')}"
            )

        print(
            "\nWeights:"
        )

        for model, weight in (
            result["weights"].items()
        ):

            print(
                f"  "
                f"{model.replace('prediction_', ''):<20}"
                f"{weight:.4f}"
            )

        print(
            "\nOOF:"
        )

        print(
            f"  RMSE = "
            f"{result['oof_metrics']['rmse']:.4f}"
        )

        print(
            f"  MAE  = "
            f"{result['oof_metrics']['mae']:.4f}"
        )

        print(
            f"  R²   = "
            f"{result['oof_metrics']['r2']:.4f}"
        )

        print(
            "\nTEST "
            "(reporting only — not used for selection):"
        )

        print(
            f"  RMSE = "
            f"{result['test_metrics']['rmse']:.4f}"
        )

        print(
            f"  MAE  = "
            f"{result['test_metrics']['mae']:.4f}"
        )

        print(
            f"  R²   = "
            f"{result['test_metrics']['r2']:.4f}"
        )

        print(
            f"\nDynamic fallback rate: "
            f"{result['fallback_rate_percent']:.2f}%"
        )

    print_section(
        "FILES CREATED"
    )

    print(
        RESULTS_JSON
    )

    print(
        CANDIDATES_CSV
    )

    print(
        f"\nPrediction files:\n"
        f"{OUTPUT_DIR}"
    )

    print_section(
        "STEP 2 COMPLETE"
    )

    print(
        "STOP HERE.\n"
        "Review Weighted Ensemble results before "
        "starting Stacking."
    )


if __name__ == "__main__":
    main()