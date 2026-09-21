from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.analysis.phase3_pipeline import (
    build_segments,
    engineer_features,
    who_analysis,
)

from app.config import (
    CLEANED_CSV,
    OUTPUT_DIR,
)


BACKEND = Path(__file__).resolve().parent

EXPECTED_FEATURES = (
    OUTPUT_DIR
    / "AirAware_ML_Features.csv"
)

WHO_CONFIG = (
    BACKEND
    / "config"
    / "who_2021_guidelines.json"
)

PHASE2_JSON = (
    OUTPUT_DIR
    / "phase2_analysis.json"
)

TEST_START = pd.Timestamp(
    "2026-08-20T00:00:00Z"
)

TEST_END = pd.Timestamp(
    "2026-08-23T00:00:00Z"
)

# We do not need to test every row yet.
# 12 timestamps per sensor is enough for this causality/parity audit.
SAMPLES_PER_SENSOR = 12

TOLERANCE = 1e-8


def section(title: str):

    print()
    print("=" * 95)
    print(title)
    print("=" * 95)


def load_inputs():

    cleaned = pd.read_csv(
        CLEANED_CSV,
        low_memory=False,
    )

    expected = pd.read_csv(
        EXPECTED_FEATURES,
        low_memory=False,
    )

    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        utc=True,
    )

    expected["timestamp"] = pd.to_datetime(
        expected["timestamp"],
        utc=True,
    )

    with open(
        WHO_CONFIG,
        "r",
        encoding="utf-8",
    ) as f:
        who = json.load(f)

    if PHASE2_JSON.exists():

        with open(
            PHASE2_JSON,
            "r",
            encoding="utf-8",
        ) as f:
            phase2 = json.load(f)

    else:
        # engineer_features currently does not require
        # Phase 2 values to calculate the numerical features.
        phase2 = {}

    return (
        cleaned,
        expected,
        phase2,
        who,
    )


def build_features_using_history_only(
    history: pd.DataFrame,
    phase2: dict,
    who: dict,
):

    # --------------------------------------------------------
    # IMPORTANT:
    # history contains ONLY timestamp <= forecast origin.
    # --------------------------------------------------------

    data = history.copy()

    data = build_segments(
        data
    )

    features, _, _ = engineer_features(
        data,
        phase2,
    )

    _, _, who_series = who_analysis(
        features,
        who,
    )

    pm25 = who_series["pm25"]

    features[
        "pm25_who_24h_average"
    ] = pm25["average"]

    features[
        "pm25_who_24h_coverage"
    ] = pm25["coverage"]

    features[
        "pm25_who_24h_ratio"
    ] = pm25["ratio"]

    return features


def compare_value(
    generated,
    expected,
):

    if pd.isna(generated) and pd.isna(expected):
        return True, 0.0

    if pd.isna(generated) != pd.isna(expected):
        return False, np.inf

    try:

        generated = float(generated)
        expected = float(expected)

        difference = abs(
            generated
            - expected
        )

        return (
            difference <= TOLERANCE,
            difference,
        )

    except Exception:

        return (
            str(generated)
            == str(expected),
            0.0,
        )


def main():

    section(
        "AIRAWARE — STEP 5A-3 FEATURE PARITY"
    )

    (
        cleaned,
        expected,
        phase2,
        who,
    ) = load_inputs()

    # --------------------------------------------------------
    # Only features used by final LightGBM models
    # --------------------------------------------------------

    import joblib

    model_bundle = joblib.load(
        BACKEND
        / "models"
        / "lightgbm"
        / "pm25_1h.joblib"
    )

    required_features = list(
        model_bundle["features"]
    )

    print(
        f"LightGBM required features : "
        f"{len(required_features)}"
    )

    print(
        f"Cleaned rows               : "
        f"{len(cleaned):,}"
    )

    print(
        f"Expected feature rows      : "
        f"{len(expected):,}"
    )

    # --------------------------------------------------------
    # Pick timestamps from TEST period
    # --------------------------------------------------------

    candidates = expected[
        (
            expected["timestamp"]
            >= TEST_START
        )
        &
        (
            expected["timestamp"]
            < TEST_END
        )
    ].copy()

    sampled_rows = []

    for sensor_id, group in candidates.groupby(
        "sensor_id"
    ):

        group = group.sort_values(
            "timestamp"
        )

        if len(group) <= SAMPLES_PER_SENSOR:

            sample = group

        else:

            indexes = np.linspace(
                0,
                len(group) - 1,
                SAMPLES_PER_SENSOR,
                dtype=int,
            )

            sample = group.iloc[
                indexes
            ]

        sampled_rows.append(
            sample
        )

    samples = pd.concat(
        sampled_rows,
        ignore_index=True,
    )

    print(
        f"Parity timestamps          : "
        f"{len(samples)}"
    )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    total_values = 0
    passed_values = 0

    failures = []

    for number, expected_row in enumerate(
        samples.itertuples(),
        1,
    ):

        timestamp = expected_row.timestamp
        sensor_id = int(
            expected_row.sensor_id
        )

        print(
            f"\rTesting "
            f"{number}/{len(samples)} "
            f"| sensor={sensor_id} "
            f"| {timestamp}",
            end="",
            flush=True,
        )

        # ====================================================
        # CAUSAL BOUNDARY
        #
        # Absolutely no rows after timestamp are supplied.
        # ====================================================

        history = cleaned[
            cleaned["timestamp"]
            <= timestamp
        ].copy()

        generated = (
            build_features_using_history_only(
                history,
                phase2,
                who,
            )
        )

        generated_row = generated[
            (
                generated[
                    "sensor_id"
                ]
                == sensor_id
            )
            &
            (
                generated[
                    "timestamp"
                ]
                == timestamp
            )
        ]

        if len(generated_row) != 1:

            failures.append(
                {
                    "timestamp":
                        timestamp,

                    "sensor_id":
                        sensor_id,

                    "feature":
                        "__ROW__",

                    "reason":
                        f"generated rows={len(generated_row)}",
                }
            )

            continue

        generated_row = (
            generated_row.iloc[0]
        )

        expected_match = expected[
            (
                expected[
                    "sensor_id"
                ]
                == sensor_id
            )
            &
            (
                expected[
                    "timestamp"
                ]
                == timestamp
            )
        ]

        if len(expected_match) != 1:

            failures.append(
                {
                    "timestamp":
                        timestamp,

                    "sensor_id":
                        sensor_id,

                    "feature":
                        "__EXPECTED_ROW__",

                    "reason":
                        f"expected rows={len(expected_match)}",
                }
            )

            continue

        expected_match = (
            expected_match.iloc[0]
        )

        for feature in required_features:

            total_values += 1

            if feature not in generated_row.index:

                failures.append(
                    {
                        "timestamp":
                            timestamp,

                        "sensor_id":
                            sensor_id,

                        "feature":
                            feature,

                        "reason":
                            "missing generated feature",
                    }
                )

                continue

            ok, difference = compare_value(
                generated_row[
                    feature
                ],
                expected_match[
                    feature
                ],
            )

            if ok:

                passed_values += 1

            else:

                failures.append(
                    {
                        "timestamp":
                            timestamp,

                        "sensor_id":
                            sensor_id,

                        "feature":
                            feature,

                        "generated":
                            generated_row[
                                feature
                            ],

                        "expected":
                            expected_match[
                                feature
                            ],

                        "absolute_difference":
                            difference,
                    }
                )

    print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    section(
        "FEATURE PARITY RESULTS"
    )

    failed_values = (
        total_values
        - passed_values
    )

    parity = (
        passed_values
        / total_values
        if total_values
        else 0
    )

    print(
        f"Values checked : "
        f"{total_values:,}"
    )

    print(
        f"Passed         : "
        f"{passed_values:,}"
    )

    print(
        f"Failed         : "
        f"{failed_values:,}"
    )

    print(
        f"Parity         : "
        f"{parity * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Save failures
    # --------------------------------------------------------

    failure_file = (
        OUTPUT_DIR
        / "realtime_feature_parity_failures.csv"
    )

    pd.DataFrame(
        failures
    ).to_csv(
        failure_file,
        index=False,
    )

    print()

    print(
        f"Failures saved:\n"
        f"{failure_file}"
    )

    section(
        "RESULT"
    )

    if failed_values == 0:

        print(
            "PASS"
        )

        print(
            "Historical features can be reproduced "
            "using only information available at or "
            "before the forecast origin."
        )

    else:

        print(
            "MISMATCH FOUND"
        )

        print(
            "Do NOT build the final realtime serving "
            "pipeline until these differences are explained."
        )


if __name__ == "__main__":
    main()