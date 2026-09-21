from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# AirAware
# Step 4B: Multi-Metric / High-PM-Aware Model Selection
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "ensemble"
    / "final_comparison_outputs"
)

FINAL_FILE = (
    INPUT_DIR
    / "final_model_comparison.csv"
)

HIGH_PM_FILE = (
    INPUT_DIR
    / "high_pm25_comparison.csv"
)

PER_SENSOR_FILE = (
    INPUT_DIR
    / "final_per_sensor_comparison.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ensemble"
    / "health_aware_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SCORES_CSV = (
    OUTPUT_DIR
    / "multi_metric_model_scores.csv"
)

REGISTRY_JSON = (
    OUTPUT_DIR
    / "health_aware_model_registry.json"
)

DETAILS_JSON = (
    OUTPUT_DIR
    / "health_aware_selection_details.json"
)


# ============================================================
# Weights
# ============================================================

WEIGHTS = {
    # Overall accuracy = 40%
    "overall_mae": 0.15,
    "overall_rmse": 0.10,
    "overall_median_ae": 0.10,
    "overall_r2": 0.05,

    # High pollution = 40%
    "p90_mae": 0.12,
    "p90_rmse": 0.08,
    "p95_mae": 0.12,
    "p95_rmse": 0.08,

    # Sensor stability = 20%
    "sensor_mean_mae": 0.08,
    "sensor_worst_mae": 0.07,
    "sensor_rmse_std": 0.05,
}


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def normalize_horizon(value):
    text = str(value).strip().lower()

    mapping = {
        "1": "1h",
        "1h": "1h",
        "3": "3h",
        "3h": "3h",
        "6": "6h",
        "6h": "6h",
    }

    return mapping.get(text, text)


def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    df = pd.read_csv(path)

    if "horizon" in df.columns:
        df["horizon"] = (
            df["horizon"]
            .apply(normalize_horizon)
        )

    if "method" in df.columns:
        df["method"] = (
            df["method"]
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# Min-Max normalization
# ============================================================

def normalize_lower_is_better(series):
    """
    Best (lowest) value -> 1
    Worst (highest) value -> 0
    """

    s = pd.to_numeric(
        series,
        errors="coerce",
    )

    minimum = s.min()
    maximum = s.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
    ):
        return pd.Series(
            np.nan,
            index=series.index,
        )

    if np.isclose(
        maximum,
        minimum,
    ):
        return pd.Series(
            1.0,
            index=series.index,
        )

    return (
        maximum - s
    ) / (
        maximum - minimum
    )


def normalize_higher_is_better(series):
    """
    Best (highest) value -> 1
    Worst (lowest) value -> 0
    """

    s = pd.to_numeric(
        series,
        errors="coerce",
    )

    minimum = s.min()
    maximum = s.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
    ):
        return pd.Series(
            np.nan,
            index=series.index,
        )

    if np.isclose(
        maximum,
        minimum,
    ):
        return pd.Series(
            1.0,
            index=series.index,
        )

    return (
        s - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# Overall Metrics
# ============================================================

def prepare_overall(final_df):

    required = [
        "horizon",
        "method",
        "rmse",
        "mae",
        "median_ae",
        "r2",
    ]

    missing = [
        col
        for col in required
        if col not in final_df.columns
    ]

    if missing:
        raise ValueError(
            f"Final comparison missing columns: {missing}"
        )

    return final_df[
        required
    ].copy()


# ============================================================
# High PM Metrics
# ============================================================

def prepare_high_pm(high_df):

    required = [
        "horizon",
        "method",
        "percentile",
        "rmse",
        "mae",
    ]

    missing = [
        col
        for col in required
        if col not in high_df.columns
    ]

    if missing:
        raise ValueError(
            f"High-PM file missing columns: {missing}"
        )

    high_df = high_df.copy()

    high_df["percentile"] = (
        pd.to_numeric(
            high_df["percentile"],
            errors="coerce",
        )
    )

    rows = []

    for (
        horizon,
        method
    ), group in high_df.groupby(
        [
            "horizon",
            "method",
        ]
    ):

        p90 = group[
            group["percentile"] == 90
        ]

        p95 = group[
            group["percentile"] == 95
        ]

        if (
            p90.empty
            or p95.empty
        ):
            continue

        rows.append({
            "horizon":
                horizon,

            "method":
                method,

            "p90_mae":
                float(
                    p90.iloc[0][
                        "mae"
                    ]
                ),

            "p90_rmse":
                float(
                    p90.iloc[0][
                        "rmse"
                    ]
                ),

            "p95_mae":
                float(
                    p95.iloc[0][
                        "mae"
                    ]
                ),

            "p95_rmse":
                float(
                    p95.iloc[0][
                        "rmse"
                    ]
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Sensor Stability
# ============================================================

def prepare_sensor_stability(sensor_df):

    required = [
        "horizon",
        "sensor_id",
        "method",
        "rmse",
        "mae",
    ]

    missing = [
        col
        for col in required
        if col not in sensor_df.columns
    ]

    if missing:
        raise ValueError(
            f"Per-sensor file missing columns: {missing}"
        )

    rows = []

    for (
        horizon,
        method
    ), group in sensor_df.groupby(
        [
            "horizon",
            "method",
        ]
    ):

        mae_values = pd.to_numeric(
            group["mae"],
            errors="coerce",
        ).dropna()

        rmse_values = pd.to_numeric(
            group["rmse"],
            errors="coerce",
        ).dropna()

        if (
            mae_values.empty
            or rmse_values.empty
        ):
            continue

        rows.append({
            "horizon":
                horizon,

            "method":
                method,

            "sensor_mean_mae":
                float(
                    mae_values.mean()
                ),

            "sensor_worst_mae":
                float(
                    mae_values.max()
                ),

            "sensor_rmse_std":
                float(
                    rmse_values.std(
                        ddof=0
                    )
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Build complete table
# ============================================================

def build_metric_table(
    overall,
    high_pm,
    stability,
):

    merged = overall.merge(
        high_pm,
        on=[
            "horizon",
            "method",
        ],
        how="inner",
    )

    merged = merged.merge(
        stability,
        on=[
            "horizon",
            "method",
        ],
        how="inner",
    )

    merged = merged.rename(
        columns={
            "rmse":
                "overall_rmse",

            "mae":
                "overall_mae",

            "median_ae":
                "overall_median_ae",

            "r2":
                "overall_r2",
        }
    )

    return merged


# ============================================================
# Score one horizon
# ============================================================

def score_horizon(df):

    df = df.copy()

    # --------------------------------------------------------
    # Lower is better
    # --------------------------------------------------------

    lower_metrics = [
        "overall_mae",
        "overall_rmse",
        "overall_median_ae",
        "p90_mae",
        "p90_rmse",
        "p95_mae",
        "p95_rmse",
        "sensor_mean_mae",
        "sensor_worst_mae",
        "sensor_rmse_std",
    ]

    for metric in lower_metrics:

        df[
            f"{metric}_score"
        ] = normalize_lower_is_better(
            df[metric]
        )

    # --------------------------------------------------------
    # Higher is better
    # --------------------------------------------------------

    df[
        "overall_r2_score"
    ] = normalize_higher_is_better(
        df[
            "overall_r2"
        ]
    )

    # --------------------------------------------------------
    # Component scores
    # --------------------------------------------------------

    df[
        "overall_accuracy_score"
    ] = (
        df[
            "overall_mae_score"
        ] * WEIGHTS[
            "overall_mae"
        ]

        + df[
            "overall_rmse_score"
        ] * WEIGHTS[
            "overall_rmse"
        ]

        + df[
            "overall_median_ae_score"
        ] * WEIGHTS[
            "overall_median_ae"
        ]

        + df[
            "overall_r2_score"
        ] * WEIGHTS[
            "overall_r2"
        ]
    ) / 0.40

    df[
        "high_pm_score"
    ] = (
        df[
            "p90_mae_score"
        ] * WEIGHTS[
            "p90_mae"
        ]

        + df[
            "p90_rmse_score"
        ] * WEIGHTS[
            "p90_rmse"
        ]

        + df[
            "p95_mae_score"
        ] * WEIGHTS[
            "p95_mae"
        ]

        + df[
            "p95_rmse_score"
        ] * WEIGHTS[
            "p95_rmse"
        ]
    ) / 0.40

    df[
        "sensor_stability_score"
    ] = (
        df[
            "sensor_mean_mae_score"
        ] * WEIGHTS[
            "sensor_mean_mae"
        ]

        + df[
            "sensor_worst_mae_score"
        ] * WEIGHTS[
            "sensor_worst_mae"
        ]

        + df[
            "sensor_rmse_std_score"
        ] * WEIGHTS[
            "sensor_rmse_std"
        ]
    ) / 0.20

    # --------------------------------------------------------
    # Final weighted score
    # --------------------------------------------------------

    df[
        "final_score"
    ] = (
        df[
            "overall_accuracy_score"
        ] * 0.40

        + df[
            "high_pm_score"
        ] * 0.40

        + df[
            "sensor_stability_score"
        ] * 0.20
    )

    # 0 → 100
    df[
        "final_score_100"
    ] = (
        df[
            "final_score"
        ]
        * 100
    )

    df = df.sort_values(
        "final_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    df[
        "final_rank"
    ] = (
        np.arange(
            len(df)
        )
        + 1
    )

    return df


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "AIRAWARE — STEP 4B: MULTI-METRIC MODEL SELECTION"
    )

    final_df = load_csv(
        FINAL_FILE
    )

    high_df = load_csv(
        HIGH_PM_FILE
    )

    sensor_df = load_csv(
        PER_SENSOR_FILE
    )

    print(
        f"Final comparison rows : "
        f"{len(final_df):,}"
    )

    print(
        f"High-PM rows          : "
        f"{len(high_df):,}"
    )

    print(
        f"Per-sensor rows       : "
        f"{len(sensor_df):,}"
    )

    overall = prepare_overall(
        final_df
    )

    high_pm = prepare_high_pm(
        high_df
    )

    stability = prepare_sensor_stability(
        sensor_df
    )

    metric_table = build_metric_table(
        overall,
        high_pm,
        stability,
    )

    all_scored = []

    registry = {}

    details = {}

    # ========================================================
    # Per horizon
    # ========================================================

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        print_section(
            f"MULTI-METRIC RANKING — {horizon}"
        )

        horizon_df = metric_table[
            metric_table[
                "horizon"
            ] == horizon
        ].copy()

        scored = score_horizon(
            horizon_df
        )

        all_scored.append(
            scored
        )

        display_cols = [
            "final_rank",
            "method",
            "final_score_100",
            "overall_accuracy_score",
            "high_pm_score",
            "sensor_stability_score",
            "overall_rmse",
            "overall_mae",
            "p95_rmse",
            "p95_mae",
            "sensor_worst_mae",
        ]

        print(
            scored[
                display_cols
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

        winner = scored.iloc[
            0
        ]

        runner_up = (
            scored.iloc[1]
            if len(scored) > 1
            else None
        )

        print(
            f"\nFINAL MULTI-METRIC WINNER "
            f"{horizon}: "
            f"{winner['method']}"
        )

        print(
            f"Score: "
            f"{winner['final_score_100']:.2f}/100"
        )

        print(
            f"Overall Accuracy Score: "
            f"{winner['overall_accuracy_score']:.4f}"
        )

        print(
            f"High-PM Score: "
            f"{winner['high_pm_score']:.4f}"
        )

        print(
            f"Sensor Stability Score: "
            f"{winner['sensor_stability_score']:.4f}"
        )

        if runner_up is not None:

            gap = (
                winner[
                    "final_score_100"
                ]
                - runner_up[
                    "final_score_100"
                ]
            )

            print(
                f"Runner-up: "
                f"{runner_up['method']}"
            )

            print(
                f"Score gap: "
                f"{gap:.2f} points"
            )

        registry[
            horizon
        ] = {
            "method":
                winner[
                    "method"
                ],

            "final_score":
                float(
                    winner[
                        "final_score_100"
                    ]
                ),

            "overall_accuracy_score":
                float(
                    winner[
                        "overall_accuracy_score"
                    ]
                ),

            "high_pm_score":
                float(
                    winner[
                        "high_pm_score"
                    ]
                ),

            "sensor_stability_score":
                float(
                    winner[
                        "sensor_stability_score"
                    ]
                ),

            "rmse":
                float(
                    winner[
                        "overall_rmse"
                    ]
                ),

            "mae":
                float(
                    winner[
                        "overall_mae"
                    ]
                ),

            "median_ae":
                float(
                    winner[
                        "overall_median_ae"
                    ]
                ),

            "r2":
                float(
                    winner[
                        "overall_r2"
                    ]
                ),

            "p90_mae":
                float(
                    winner[
                        "p90_mae"
                    ]
                ),

            "p90_rmse":
                float(
                    winner[
                        "p90_rmse"
                    ]
                ),

            "p95_mae":
                float(
                    winner[
                        "p95_mae"
                    ]
                ),

            "p95_rmse":
                float(
                    winner[
                        "p95_rmse"
                    ]
                ),

            "sensor_mean_mae":
                float(
                    winner[
                        "sensor_mean_mae"
                    ]
                ),

            "sensor_worst_mae":
                float(
                    winner[
                        "sensor_worst_mae"
                    ]
                ),

            "sensor_rmse_std":
                float(
                    winner[
                        "sensor_rmse_std"
                    ]
                ),
        }

        details[
            horizon
        ] = scored.to_dict(
            orient="records"
        )

    # ========================================================
    # Save scores
    # ========================================================

    final_scores = pd.concat(
        all_scored,
        ignore_index=True,
    )

    final_scores.to_csv(
        SCORES_CSV,
        index=False,
    )

    # ========================================================
    # Save registry
    # ========================================================

    registry_payload = {
        "project":
            "AirAware",

        "selection_method":
            "Multi-metric forecasting selection",

        "important_note":
            (
                "This registry does not use WHO AQI. "
                "High-PM performance is based on the "
                "upper tail of observed PM2.5 values "
                "(P90 and P95) in the common TEST set."
            ),

        "score_weights": {
            "overall_accuracy":
                0.40,

            "high_pm_performance":
                0.40,

            "sensor_stability":
                0.20,

            "details":
                WEIGHTS,
        },

        "horizons":
            registry,
    }

    with open(
        REGISTRY_JSON,
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
    # Detailed JSON
    # ========================================================

    details_payload = {
        "weights":
            WEIGHTS,

        "ranking_by_horizon":
            details,
    }

    with open(
        DETAILS_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            details_payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Summary
    # ========================================================

    print_section(
        "FINAL MULTI-METRIC AIRAWARE ARCHITECTURE"
    )

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        result = registry[
            horizon
        ]

        print(
            f"\n{horizon}"
        )

        print(
            f"Winner : "
            f"{result['method']}"
        )

        print(
            f"Score  : "
            f"{result['final_score']:.2f}/100"
        )

        print(
            f"RMSE   : "
            f"{result['rmse']:.4f}"
        )

        print(
            f"MAE    : "
            f"{result['mae']:.4f}"
        )

        print(
            f"P95 MAE: "
            f"{result['p95_mae']:.4f}"
        )

    print_section(
        "FILES CREATED"
    )

    print(
        SCORES_CSV
    )

    print(
        REGISTRY_JSON
    )

    print(
        DETAILS_JSON
    )

    print_section(
        "STEP 4B COMPLETE"
    )

    print(
        "STOP HERE.\n"
        "Review the multi-metric winners before "
        "freezing the real-time forecast architecture."
    )


if __name__ == "__main__":
    main()