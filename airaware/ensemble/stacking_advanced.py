from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# Optional libraries
# ============================================================

XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    pass

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    pass


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OOF_FILE = BASE_DIR / "stacking_oof_predictions.csv"
TEST_FILE = BASE_DIR / "test_predictions_all_models.csv"

OUTPUT_DIR = BASE_DIR / "ensemble" / "stacking_advanced_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = OUTPUT_DIR / "stacking_advanced_results.json"
RESULTS_CSV = OUTPUT_DIR / "stacking_advanced_candidates.csv"


# ============================================================
# Base-model candidates
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
        [
            "prediction_gru",
            "prediction_lightgbm",
            "prediction_tft",
            "prediction_random_forest",
            "prediction_xgboost",
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
        [
            "prediction_tft",
            "prediction_lightgbm",
            "prediction_gru",
            "prediction_random_forest",
            "prediction_xgboost",
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
        [
            "prediction_tft",
            "prediction_gru",
            "prediction_random_forest",
            "prediction_lightgbm",
            "prediction_xgboost",
        ],
    ],
}


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
        c for c in df.columns
        if c.startswith("prediction_")
    ]

    for col in prediction_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "timestamp",
            "horizon",
            "actual",
        ]
    ).copy()

    return df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "median_ae": None,
            "r2": None,
        }

    return {
        "n": int(len(y_true)),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "median_ae": float(
            median_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ) if len(y_true) > 1 else None,
    }


def align_candidate(df, models):
    return df.dropna(
        subset=["actual"] + models
    ).copy()


# ============================================================
# Chronological meta split
# ============================================================

def chronological_meta_split(
    df,
    validation_fraction=0.25,
):
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    split_index = int(
        len(df)
        * (1 - validation_fraction)
    )

    if (
        split_index <= 0
        or split_index >= len(df)
    ):
        raise ValueError(
            "Invalid meta train/validation split."
        )

    return (
        df.iloc[:split_index].copy(),
        df.iloc[split_index:].copy(),
    )


# ============================================================
# Meta-model definitions
# ============================================================

def build_meta_models():
    models = {
        "linear_regression":
            LinearRegression(),

        "nonnegative_linear":
            LinearRegression(
                positive=True
            ),

        "ridge_0.1":
            Ridge(alpha=0.1),

        "ridge_1.0":
            Ridge(alpha=1.0),

        "ridge_10.0":
            Ridge(alpha=10.0),

        "elasticnet":
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                max_iter=10000,
                random_state=42,
            ),

        "gradient_boosting":
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=10,
                subsample=0.85,
                random_state=42,
            ),

        "hist_gradient_boosting":
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=1.0,
                random_state=42,
            ),

        "random_forest_meta":
            RandomForestRegressor(
                n_estimators=250,
                max_depth=4,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost_meta"] = XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=2.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )

    if LIGHTGBM_AVAILABLE:
        models["lightgbm_meta"] = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=2.0,
            verbosity=-1,
            random_state=42,
        )

    return models


# ============================================================
# Validation
# ============================================================

def evaluate_meta_candidate(
    model_name,
    model,
    models,
    oof_horizon,
):
    aligned = align_candidate(
        oof_horizon,
        models,
    )

    if len(aligned) < 100:
        return None

    train_df, val_df = chronological_meta_split(
        aligned,
        validation_fraction=0.25,
    )

    X_train = train_df[
        models
    ].to_numpy(dtype=float)

    y_train = train_df[
        "actual"
    ].to_numpy(dtype=float)

    X_val = val_df[
        models
    ].to_numpy(dtype=float)

    y_val = val_df[
        "actual"
    ].to_numpy(dtype=float)

    local_model = clone(model)

    local_model.fit(
        X_train,
        y_train,
    )

    pred = local_model.predict(
        X_val
    )

    metrics = calculate_metrics(
        y_val,
        pred,
    )

    return {
        "meta_model": model_name,
        "base_models": models,
        "aligned_rows": int(len(aligned)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "validation_metrics": metrics,
    }


# ============================================================
# Final fit
# ============================================================

def fit_final_model(
    model_template,
    models,
    oof_horizon,
):
    aligned = align_candidate(
        oof_horizon,
        models,
    )

    X = aligned[
        models
    ].to_numpy(dtype=float)

    y = aligned[
        "actual"
    ].to_numpy(dtype=float)

    model = clone(
        model_template
    )

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# Test
# ============================================================

def evaluate_test(
    model,
    models,
    test_horizon,
):
    aligned = align_candidate(
        test_horizon,
        models,
    )

    if aligned.empty:
        return None, None

    X = aligned[
        models
    ].to_numpy(dtype=float)

    y = aligned[
        "actual"
    ].to_numpy(dtype=float)

    pred = model.predict(X)

    metrics = calculate_metrics(
        y,
        pred,
    )

    output = aligned.copy()

    output[
        "stacking_prediction"
    ] = pred

    output[
        "absolute_error"
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
        "AIRAWARE — STEP 3B: ADVANCED STACKING"
    )

    if not OOF_FILE.exists():
        raise FileNotFoundError(
            OOF_FILE
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            TEST_FILE
        )

    oof = load_data(
        OOF_FILE
    )

    test = load_data(
        TEST_FILE
    )

    meta_models = build_meta_models()

    print(
        "Meta models available:"
    )

    for name in meta_models:
        print(f"  - {name}")

    print(
        f"\nXGBoost available: {XGBOOST_AVAILABLE}"
    )

    print(
        f"LightGBM available: {LIGHTGBM_AVAILABLE}"
    )

    candidate_results = []

    winners = {}

    # ========================================================
    # Horizon
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

        horizon_results = []

        for base_models in CANDIDATES[
            horizon
        ]:

            print(
                "\nBase models:"
            )

            print(
                " + ".join(
                    x.replace(
                        "prediction_",
                        "",
                    )
                    for x in base_models
                )
            )

            missing = [
                x
                for x in base_models
                if (
                    x not in oof_h.columns
                    or x not in test_h.columns
                )
            ]

            if missing:
                print(
                    f"Skipping missing columns: {missing}"
                )
                continue

            for meta_name, meta_template in (
                meta_models.items()
            ):

                try:

                    result = evaluate_meta_candidate(
                        model_name=meta_name,
                        model=meta_template,
                        models=base_models,
                        oof_horizon=oof_h,
                    )

                    if result is None:
                        continue

                    result[
                        "horizon"
                    ] = horizon

                    candidate_results.append(
                        result
                    )

                    horizon_results.append(
                        result
                    )

                    m = result[
                        "validation_metrics"
                    ]

                    print(
                        f"  {meta_name:<24}"
                        f" RMSE={m['rmse']:.4f}"
                        f" MAE={m['mae']:.4f}"
                        f" R²={m['r2']:.4f}"
                    )

                except Exception as exc:

                    print(
                        f"  {meta_name:<24}"
                        f" FAILED: {exc}"
                    )

        # ====================================================
        # Choose by meta-validation only
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
            f"BEST ADVANCED STACKING — {horizon}"
        )

        print(
            f"Meta model: "
            f"{best['meta_model']}"
        )

        print(
            "Base models:"
        )

        for x in best[
            "base_models"
        ]:
            print(
                f"  - "
                f"{x.replace('prediction_', '')}"
            )

        print(
            f"\nMeta-validation RMSE: "
            f"{best['validation_metrics']['rmse']:.4f}"
        )

        # ====================================================
        # Fit frozen winner on all OOF
        # ====================================================

        template = meta_models[
            best["meta_model"]
        ]

        final_model = fit_final_model(
            model_template=template,
            models=best[
                "base_models"
            ],
            oof_horizon=oof_h,
        )

        # ====================================================
        # Test
        # ====================================================

        test_metrics, test_preds = evaluate_test(
            model=final_model,
            models=best[
                "base_models"
            ],
            test_horizon=test_h,
        )

        print(
            f"TEST RMSE: "
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

        out_file = (
            OUTPUT_DIR
            / f"advanced_stacking_{horizon}_test_predictions.csv"
        )

        test_preds.to_csv(
            out_file,
            index=False,
        )

        # ----------------------------------------------------
        # Model-specific interpretability
        # ----------------------------------------------------

        coefficients = None
        feature_importances = None

        if hasattr(
            final_model,
            "coef_",
        ):

            coefficients = {
                model.replace(
                    "prediction_",
                    "",
                ): float(coef)
                for model, coef
                in zip(
                    best[
                        "base_models"
                    ],
                    np.ravel(
                        final_model.coef_
                    ),
                )
            }

        if hasattr(
            final_model,
            "feature_importances_",
        ):

            feature_importances = {
                model.replace(
                    "prediction_",
                    "",
                ): float(importance)
                for model, importance
                in zip(
                    best[
                        "base_models"
                    ],
                    final_model.feature_importances_,
                )
            }

        winners[
            horizon
        ] = {
            "meta_model":
                best["meta_model"],

            "base_models": [
                x.replace(
                    "prediction_",
                    "",
                )
                for x in best[
                    "base_models"
                ]
            ],

            "meta_validation_metrics":
                best[
                    "validation_metrics"
                ],

            "test_metrics":
                test_metrics,

            "coefficients":
                coefficients,

            "feature_importances":
                feature_importances,

            "prediction_file":
                str(out_file),
        }

    # ========================================================
    # Save candidates CSV
    # ========================================================

    flat = []

    for r in candidate_results:

        flat.append({
            "horizon":
                r["horizon"],

            "base_models":
                " + ".join(
                    x.replace(
                        "prediction_",
                        "",
                    )
                    for x in r[
                        "base_models"
                    ]
                ),

            "meta_model":
                r[
                    "meta_model"
                ],

            "aligned_rows":
                r[
                    "aligned_rows"
                ],

            "train_rows":
                r[
                    "train_rows"
                ],

            "validation_rows":
                r[
                    "validation_rows"
                ],

            "validation_rmse":
                r[
                    "validation_metrics"
                ]["rmse"],

            "validation_mae":
                r[
                    "validation_metrics"
                ]["mae"],

            "validation_r2":
                r[
                    "validation_metrics"
                ]["r2"],
        })

    df_results = pd.DataFrame(
        flat
    )

    if not df_results.empty:

        df_results = df_results.sort_values(
            [
                "horizon",
                "validation_rmse",
            ]
        )

    df_results.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # ========================================================
    # JSON
    # ========================================================

    payload = {
        "project":
            "AirAware",

        "phase":
            "Advanced Stacking",

        "selection_rule":
            "Advanced stacking model chosen using "
            "chronological meta-validation within OOF only. "
            "Test is evaluated after freezing the winner.",

        "xgboost_available":
            XGBOOST_AVAILABLE,

        "lightgbm_available":
            LIGHTGBM_AVAILABLE,

        "candidate_results":
            candidate_results,

        "final_by_horizon":
            winners,
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
    # Final summary
    # ========================================================

    print_section(
        "ADVANCED STACKING FINAL SUMMARY"
    )

    for horizon, result in (
        winners.items()
    ):

        print(
            f"\n{horizon}"
        )

        print(
            f"Meta Model: "
            f"{result['meta_model']}"
        )

        print(
            f"Base Models: "
            f"{', '.join(result['base_models'])}"
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
        "STEP 3B COMPLETE"
    )

    print(
        "STOP HERE.\n"
        "Next step is final comparison between:\n"
        "Best Single Model\n"
        "Weighted Ensemble\n"
        "Original Stacking\n"
        "Advanced Stacking"
    )


if __name__ == "__main__":
    main()