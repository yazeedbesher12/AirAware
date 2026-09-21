from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# AirAware
# Step 4C: Robustness + Underprediction Check
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FINAL_DIR = (
    BASE_DIR
    / "ensemble"
    / "final_comparison_outputs"
)

COMMON_FILES = {
    "1h": FINAL_DIR / "final_common_test_1h.csv",
    "3h": FINAL_DIR / "final_common_test_3h.csv",
    "6h": FINAL_DIR / "final_common_test_6h.csv",
}

MULTI_METRIC_FILE = (
    BASE_DIR
    / "ensemble"
    / "health_aware_outputs"
    / "multi_metric_model_scores.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ensemble"
    / "robustness_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

UNDERPREDICTION_CSV = (
    OUTPUT_DIR
    / "underprediction_analysis.csv"
)

SENSITIVITY_CSV = (
    OUTPUT_DIR
    / "weight_sensitivity_results.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "robustness_summary.json"
)


# ============================================================
# Weight Schemes
# ============================================================

WEIGHT_SCHEMES = {
    # Current
    "balanced_40_40_20": {
        "overall": 0.40,
        "high_pm": 0.40,
        "stability": 0.20,
    },

    # More general accuracy
    "accuracy_focused_50_30_20": {
        "overall": 0.50,
        "high_pm": 0.30,
        "stability": 0.20,
    },

    # More health / pollution focused
    "health_focused_35_45_20": {
        "overall": 0.35,
        "high_pm": 0.45,
        "stability": 0.20,
    },

    # Strong high-pollution priority
    "high_pm_heavy_30_50_20": {
        "overall": 0.30,
        "high_pm": 0.50,
        "stability": 0.20,
    },

    # Slightly more sensor-stability emphasis
    "stability_focused_40_35_25": {
        "overall": 0.40,
        "high_pm": 0.35,
        "stability": 0.25,
    },
}


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    return pd.read_csv(path)


def safe_mean(series):
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(s) == 0:
        return np.nan

    return float(s.mean())


def safe_max(series):
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(s) == 0:
        return np.nan

    return float(s.max())


# ============================================================
# Underprediction Analysis
# ============================================================

def analyze_underprediction(
    df,
    horizon,
):
    """
    Analyze all prediction columns on:
    - all rows
    - top 10% PM2.5
    - top 5% PM2.5

    Error convention:
        prediction_error = predicted - actual

    Negative = underprediction
    Positive = overprediction
    """

    prediction_cols = [
        c
        for c in df.columns
        if c.startswith("prediction_")
    ]

    actual = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    p90 = float(
        actual.quantile(0.90)
    )

    p95 = float(
        actual.quantile(0.95)
    )

    subsets = {
        "all": df,
        "p90": df[
            actual >= p90
        ].copy(),
        "p95": df[
            actual >= p95
        ].copy(),
    }

    rows = []

    for subset_name, subset in subsets.items():

        for col in prediction_cols:

            local = subset[
                [
                    "actual",
                    col,
                ]
            ].copy()

            local["actual"] = pd.to_numeric(
                local["actual"],
                errors="coerce",
            )

            local[col] = pd.to_numeric(
                local[col],
                errors="coerce",
            )

            local = local.dropna()

            if local.empty:
                continue

            local["error"] = (
                local[col]
                - local["actual"]
            )

            local["underprediction"] = (
                local["error"] < 0
            )

            local["underprediction_amount"] = np.where(
                local["error"] < 0,
                -local["error"],
                0.0,
            )

            under_mask = (
                local["underprediction"]
            )

            under_values = local.loc[
                under_mask,
                "underprediction_amount",
            ]

            bias = float(
                local["error"].mean()
            )

            underprediction_rate = float(
                under_mask.mean()
            )

            mean_underprediction = (
                float(
                    under_values.mean()
                )
                if len(under_values) > 0
                else 0.0
            )

            worst_underprediction = (
                float(
                    under_values.max()
                )
                if len(under_values) > 0
                else 0.0
            )

            rows.append({
                "horizon":
                    horizon,

                "subset":
                    subset_name,

                "method":
                    col.replace(
                        "prediction_",
                        "",
                    ),

                "n":
                    int(len(local)),

                "threshold_p90":
                    p90,

                "threshold_p95":
                    p95,

                "bias_pred_minus_actual":
                    bias,

                "underprediction_rate":
                    underprediction_rate,

                "mean_underprediction":
                    mean_underprediction,

                "worst_underprediction":
                    worst_underprediction,
            })

    return rows


# ============================================================
# Optional Per-Sensor Worst Underprediction
# ============================================================

def analyze_per_sensor_underprediction(
    df,
    horizon,
):
    if "sensor_id" not in df.columns:
        return []

    prediction_cols = [
        c
        for c in df.columns
        if c.startswith("prediction_")
    ]

    rows = []

    p95 = float(
        pd.to_numeric(
            df["actual"],
            errors="coerce",
        ).quantile(0.95)
    )

    high = df[
        pd.to_numeric(
            df["actual"],
            errors="coerce",
        ) >= p95
    ].copy()

    for sensor_id, group in high.groupby(
        "sensor_id"
    ):

        for col in prediction_cols:

            local = group[
                [
                    "actual",
                    col,
                ]
            ].copy()

            local["actual"] = pd.to_numeric(
                local["actual"],
                errors="coerce",
            )

            local[col] = pd.to_numeric(
                local[col],
                errors="coerce",
            )

            local = local.dropna()

            if local.empty:
                continue

            error = (
                local[col]
                - local["actual"]
            )

            under_amount = np.where(
                error < 0,
                -error,
                0.0,
            )

            rows.append({
                "horizon":
                    horizon,

                "subset":
                    "p95_per_sensor",

                "sensor_id":
                    sensor_id,

                "method":
                    col.replace(
                        "prediction_",
                        "",
                    ),

                "n":
                    int(len(local)),

                "underprediction_rate":
                    float(
                        np.mean(
                            error < 0
                        )
                    ),

                "mean_underprediction":
                    float(
                        np.mean(
                            under_amount[
                                under_amount > 0
                            ]
                        )
                    )
                    if np.any(
                        under_amount > 0
                    )
                    else 0.0,

                "worst_underprediction":
                    float(
                        np.max(
                            under_amount
                        )
                    ),
            })

    return rows


# ============================================================
# Weight Sensitivity
# ============================================================

def run_weight_sensitivity(
    scores_df,
):
    """
    Uses the already-computed normalized component scores:

    - overall_accuracy_score
    - high_pm_score
    - sensor_stability_score

    Then re-ranks models under several sensible
    weighting schemes.
    """

    required = [
        "horizon",
        "method",
        "overall_accuracy_score",
        "high_pm_score",
        "sensor_stability_score",
    ]

    missing = [
        c
        for c in required
        if c not in scores_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in multi-metric scores: {missing}"
        )

    rows = []

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        hdf = scores_df[
            scores_df["horizon"] == horizon
        ].copy()

        for scheme_name, weights in (
            WEIGHT_SCHEMES.items()
        ):

            hdf[
                "sensitivity_score"
            ] = (
                hdf[
                    "overall_accuracy_score"
                ] * weights[
                    "overall"
                ]

                + hdf[
                    "high_pm_score"
                ] * weights[
                    "high_pm"
                ]

                + hdf[
                    "sensor_stability_score"
                ] * weights[
                    "stability"
                ]
            )

            ranked = hdf.sort_values(
                "sensitivity_score",
                ascending=False,
            ).reset_index(
                drop=True
            )

            ranked[
                "rank"
            ] = (
                np.arange(
                    len(ranked)
                )
                + 1
            )

            for _, row in ranked.iterrows():

                rows.append({
                    "horizon":
                        horizon,

                    "scheme":
                        scheme_name,

                    "method":
                        row["method"],

                    "score":
                        float(
                            row[
                                "sensitivity_score"
                            ]
                        ),

                    "rank":
                        int(
                            row["rank"]
                        ),

                    "overall_weight":
                        weights[
                            "overall"
                        ],

                    "high_pm_weight":
                        weights[
                            "high_pm"
                        ],

                    "stability_weight":
                        weights[
                            "stability"
                        ],
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Winner Robustness
# ============================================================

def summarize_sensitivity(
    sensitivity_df,
):
    summaries = {}

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        hdf = sensitivity_df[
            sensitivity_df[
                "horizon"
            ] == horizon
        ].copy()

        winners = hdf[
            hdf["rank"] == 1
        ]

        winner_counts = (
            winners[
                "method"
            ]
            .value_counts()
            .to_dict()
        )

        total_schemes = int(
            winners[
                "scheme"
            ].nunique()
        )

        best_method = (
            winners[
                "method"
            ]
            .value_counts()
            .idxmax()
        )

        win_count = int(
            winner_counts[
                best_method
            ]
        )

        robustness = (
            win_count
            / total_schemes
            if total_schemes > 0
            else 0
        )

        summaries[
            horizon
        ] = {
            "winner_counts":
                {
                    str(k): int(v)
                    for k, v
                    in winner_counts.items()
                },

            "most_robust_method":
                best_method,

            "wins":
                win_count,

            "total_schemes":
                total_schemes,

            "robustness_fraction":
                float(
                    robustness
                ),
        }

    return summaries


# ============================================================
# Underprediction Summary for Key Candidates
# ============================================================

def summarize_underprediction(
    under_df,
):
    summary = {}

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        p95 = under_df[
            (
                under_df[
                    "horizon"
                ] == horizon
            )
            &
            (
                under_df[
                    "subset"
                ] == "p95"
            )
        ].copy()

        if p95.empty:
            continue

        # Lower is safer for these metrics
        safest_rate = p95.sort_values(
            "underprediction_rate"
        ).iloc[0]

        safest_mean = p95.sort_values(
            "mean_underprediction"
        ).iloc[0]

        safest_worst = p95.sort_values(
            "worst_underprediction"
        ).iloc[0]

        summary[
            horizon
        ] = {
            "lowest_underprediction_rate":
                {
                    "method":
                        safest_rate[
                            "method"
                        ],

                    "value":
                        float(
                            safest_rate[
                                "underprediction_rate"
                            ]
                        ),
                },

            "lowest_mean_underprediction":
                {
                    "method":
                        safest_mean[
                            "method"
                        ],

                    "value":
                        float(
                            safest_mean[
                                "mean_underprediction"
                            ]
                        ),
                },

            "lowest_worst_underprediction":
                {
                    "method":
                        safest_worst[
                            "method"
                        ],

                    "value":
                        float(
                            safest_worst[
                                "worst_underprediction"
                            ]
                        ),
                },
        }

    return summary


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "AIRAWARE — STEP 4C: ROBUSTNESS + UNDERPREDICTION CHECK"
    )

    if not MULTI_METRIC_FILE.exists():
        raise FileNotFoundError(
            f"Run Step 4B first:\n{MULTI_METRIC_FILE}"
        )

    scores_df = load_csv(
        MULTI_METRIC_FILE
    )

    under_rows = []
    per_sensor_rows = []

    # ========================================================
    # Analyze each horizon
    # ========================================================

    for horizon, path in (
        COMMON_FILES.items()
    ):

        print_section(
            f"UNDERPREDICTION ANALYSIS — {horizon}"
        )

        df = load_csv(
            path
        )

        print(
            f"Rows: {len(df):,}"
        )

        rows = analyze_underprediction(
            df,
            horizon,
        )

        under_rows.extend(
            rows
        )

        per_sensor_rows.extend(
            analyze_per_sensor_underprediction(
                df,
                horizon,
            )
        )

        temp = pd.DataFrame(
            rows
        )

        p95 = temp[
            temp[
                "subset"
            ] == "p95"
        ].copy()

        p95 = p95.sort_values(
            [
                "underprediction_rate",
                "mean_underprediction",
            ]
        )

        print(
            "\nP95 Underprediction Ranking:"
        )

        print(
            p95[
                [
                    "method",
                    "n",
                    "bias_pred_minus_actual",
                    "underprediction_rate",
                    "mean_underprediction",
                    "worst_underprediction",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # ========================================================
    # Save Underprediction
    # ========================================================

    under_df = pd.DataFrame(
        under_rows
    )

    under_df.to_csv(
        UNDERPREDICTION_CSV,
        index=False,
    )

    if per_sensor_rows:

        pd.DataFrame(
            per_sensor_rows
        ).to_csv(
            OUTPUT_DIR
            / "p95_per_sensor_underprediction.csv",
            index=False,
        )

    # ========================================================
    # Sensitivity
    # ========================================================

    print_section(
        "WEIGHT SENSITIVITY ANALYSIS"
    )

    sensitivity_df = (
        run_weight_sensitivity(
            scores_df
        )
    )

    sensitivity_df.to_csv(
        SENSITIVITY_CSV,
        index=False,
    )

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        print(
            f"\n{horizon}"
        )

        winners = sensitivity_df[
            (
                sensitivity_df[
                    "horizon"
                ] == horizon
            )
            &
            (
                sensitivity_df[
                    "rank"
                ] == 1
            )
        ]

        print(
            winners[
                [
                    "scheme",
                    "method",
                    "score",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # ========================================================
    # Robustness Summary
    # ========================================================

    sensitivity_summary = (
        summarize_sensitivity(
            sensitivity_df
        )
    )

    underprediction_summary = (
        summarize_underprediction(
            under_df
        )
    )

    print_section(
        "ROBUSTNESS SUMMARY"
    )

    for horizon in [
        "1h",
        "3h",
        "6h",
    ]:

        s = sensitivity_summary[
            horizon
        ]

        print(
            f"\n{horizon}"
        )

        print(
            f"Most robust method: "
            f"{s['most_robust_method']}"
        )

        print(
            f"Wins: "
            f"{s['wins']}/"
            f"{s['total_schemes']}"
        )

        print(
            f"Robustness: "
            f"{s['robustness_fraction'] * 100:.1f}%"
        )

        u = underprediction_summary.get(
            horizon
        )

        if u:

            print(
                "P95 safest underprediction rate: "
                f"{u['lowest_underprediction_rate']['method']} "
                f"({u['lowest_underprediction_rate']['value']:.4f})"
            )

            print(
                "P95 smallest mean underprediction: "
                f"{u['lowest_mean_underprediction']['method']} "
                f"({u['lowest_mean_underprediction']['value']:.4f})"
            )

            print(
                "P95 smallest worst underprediction: "
                f"{u['lowest_worst_underprediction']['method']} "
                f"({u['lowest_worst_underprediction']['value']:.4f})"
            )

    # ========================================================
    # JSON
    # ========================================================

    payload = {
        "project":
            "AirAware",

        "phase":
            "Robustness and Underprediction Check",

        "weight_schemes":
            WEIGHT_SCHEMES,

        "sensitivity_summary":
            sensitivity_summary,

        "underprediction_summary":
            underprediction_summary,

        "important_note":
            (
                "Negative prediction bias means systematic "
                "underprediction. P95 analysis focuses on "
                "the highest 5% observed PM2.5 values in "
                "the final common TEST subset."
            ),
    }

    with open(
        SUMMARY_JSON,
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
    # Files
    # ========================================================

    print_section(
        "FILES CREATED"
    )

    print(
        UNDERPREDICTION_CSV
    )

    print(
        SENSITIVITY_CSV
    )

    print(
        SUMMARY_JSON
    )

    print(
        OUTPUT_DIR
        / "p95_per_sensor_underprediction.csv"
    )

    print_section(
        "STEP 4C COMPLETE"
    )

    print(
        "STOP HERE.\n"
        "Do not freeze the final architecture until "
        "the robustness and P95 underprediction results "
        "are reviewed."
    )


if __name__ == "__main__":
    main()