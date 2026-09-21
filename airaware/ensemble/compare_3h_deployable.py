from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parent.parent

ALL_MODELS_FILE = ROOT / "test_predictions_all_models.csv"

CAUSAL_FILE = (
    ROOT
    / "backend"
    / "outputs"
    / "predictions"
    / "tft_causal"
    / "test_pm25_3h.csv"
)


def metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual
    absolute = np.abs(error)

    return {
        "n": len(actual),

        "mae":
            mean_absolute_error(
                actual,
                predicted,
            ),

        "rmse":
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            ),

        "median_ae":
            np.median(
                absolute
            ),

        "r2":
            r2_score(
                actual,
                predicted,
            ),
    }


def p95_metrics(
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

    threshold = np.percentile(
        actual,
        95,
    )

    mask = (
        actual
        >= threshold
    )

    a = actual[mask]
    p = predicted[mask]

    error = p - a

    under = (
        error < 0
    )

    if under.any():

        amounts = (
            a[under]
            - p[under]
        )

        mean_under = (
            amounts.mean()
        )

        worst_under = (
            amounts.max()
        )

    else:

        mean_under = 0.0
        worst_under = 0.0

    return {
        "p95_threshold":
            threshold,

        "p95_n":
            len(a),

        "p95_mae":
            np.mean(
                np.abs(
                    error
                )
            ),

        "p95_rmse":
            np.sqrt(
                np.mean(
                    error ** 2
                )
            ),

        "underprediction_rate":
            under.mean(),

        "mean_underprediction":
            mean_under,

        "worst_underprediction":
            worst_under,
    }


# ============================================================
# Load original model predictions
# ============================================================

all_models = pd.read_csv(
    ALL_MODELS_FILE,
    low_memory=False,
)

all_models["timestamp"] = (
    pd.to_datetime(
        all_models["timestamp"],
        utc=True,
    )
)

all_models["horizon"] = (
    all_models[
        "horizon"
    ]
    .astype(str)
    .str.lower()
)

all_models = (
    all_models[
        all_models[
            "horizon"
        ]
        .isin(
            [
                "3h",
                "3",
            ]
        )
    ]
    .copy()
)


# ============================================================
# Load causal TFT
# ============================================================

causal = pd.read_csv(
    CAUSAL_FILE,
)

causal[
    "forecast_origin_timestamp"
] = pd.to_datetime(
    causal[
        "forecast_origin_timestamp"
    ],
    utc=True,
)

causal = causal.rename(
    columns={
        "forecast_origin_timestamp":
            "timestamp",

        "predicted":
            "prediction_tft_causal",
    }
)


# ============================================================
# Merge
# ============================================================

base_cols = [
    "timestamp",
    "sensor_id",
    "actual",
]

prediction_candidates = {
    "LightGBM":
        [
            "prediction_lightgbm",
            "lightgbm",
        ],

    "XGBoost":
        [
            "prediction_xgboost",
            "xgboost",
        ],

    "RandomForest":
        [
            "prediction_random_forest",
            "prediction_rf",
            "random_forest",
        ],
}


def find_column(
    dataframe,
    candidates,
):

    for col in candidates:

        if col in dataframe.columns:
            return col

    return None


selected = {}

for model, candidates in (
    prediction_candidates.items()
):

    col = find_column(
        all_models,
        candidates,
    )

    if col is None:

        print(
            f"WARNING: "
            f"Could not find "
            f"{model} prediction column."
        )

    else:

        selected[
            model
        ] = col


needed = (
    base_cols
    + list(
        selected.values()
    )
)

original = (
    all_models[
        needed
    ]
    .copy()
)


merged = original.merge(

    causal[
        [
            "timestamp",
            "sensor_id",
            "prediction_tft_causal",
        ]
    ],

    on=[
        "timestamp",
        "sensor_id",
    ],

    how="inner",
)


# ============================================================
# Strict common rows
# ============================================================

prediction_columns = (
    list(
        selected.values()
    )
    + [
        "prediction_tft_causal"
    ]
)

common = merged.dropna(
    subset=[
        "actual"
    ]
    + prediction_columns
).copy()


print()
print("=" * 95)
print("AIRAWARE — FINAL 3h DEPLOYABLE MODEL COMPARISON")
print("=" * 95)

print(
    f"\nStrict common rows: "
    f"{len(common):,}"
)


# ============================================================
# Evaluate
# ============================================================

models = {
    **selected,
    "CausalTFT":
        "prediction_tft_causal",
}

results = []

for name, col in models.items():

    overall = metrics(
        common[
            "actual"
        ],
        common[
            col
        ],
    )

    high = p95_metrics(
        common[
            "actual"
        ],
        common[
            col
        ],
    )

    result = {
        "model":
            name,

        **overall,

        **high,
    }

    results.append(
        result
    )


results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        "rmse"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# Print
# ============================================================

for row in (
    results_df
    .itertuples(
        index=False
    )
):

    print()
    print(
        "-" * 70
    )

    print(
        row.model
    )

    print(
        "-" * 70
    )

    print(
        f"N                  : "
        f"{row.n}"
    )

    print(
        f"MAE                : "
        f"{row.mae:.6f}"
    )

    print(
        f"RMSE               : "
        f"{row.rmse:.6f}"
    )

    print(
        f"Median AE          : "
        f"{row.median_ae:.6f}"
    )

    print(
        f"R²                 : "
        f"{row.r2:.6f}"
    )

    print(
        f"P95 MAE            : "
        f"{row.p95_mae:.6f}"
    )

    print(
        f"P95 RMSE           : "
        f"{row.p95_rmse:.6f}"
    )

    print(
        f"P95 underprediction: "
        f"{row.underprediction_rate * 100:.2f}%"
    )

    print(
        f"Mean underprediction: "
        f"{row.mean_underprediction:.6f}"
    )

    print(
        f"Worst underprediction: "
        f"{row.worst_underprediction:.6f}"
    )


# ============================================================
# Save
# ============================================================

OUTPUT = (
    ROOT
    / "ensemble"
    / "deployable_3h_comparison.csv"
)

results_df.to_csv(
    OUTPUT,
    index=False,
)

print()
print("=" * 95)
print("RMSE RANKING")
print("=" * 95)

print(
    results_df[
        [
            "model",
            "rmse",
            "mae",
            "r2",
            "p95_rmse",
            "underprediction_rate",
        ]
    ]
    .to_string(
        index=False
    )
)

print()
print(
    f"Saved:\n{OUTPUT}"
)