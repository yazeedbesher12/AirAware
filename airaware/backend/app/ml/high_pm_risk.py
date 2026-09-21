from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[2]

PREDICTION_DIR = (
    BACKEND_DIR
    / "outputs"
    / "predictions"
    / "lightgbm"
)


class HighPMRisk:

    def __init__(self):

        self.thresholds = {}

        for horizon in (
            "1h",
            "3h",
            "6h",
        ):
            self.thresholds[horizon] = (
                self._load_thresholds(
                    horizon
                )
            )


    def _load_thresholds(
        self,
        horizon: str,
    ) -> dict:

        path = (
            PREDICTION_DIR
            / f"oof_pm25_{horizon}.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"OOF prediction file not found:\n{path}"
            )

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        if "actual" not in df.columns:

            raise ValueError(
                f"{path.name} does not contain "
                "'actual' column."
            )

        actual = pd.to_numeric(
            df["actual"],
            errors="coerce",
        )

        actual = (
            actual[
                actual.notna()
            ]
            .to_numpy(
                dtype=float
            )
        )

        actual = actual[
            np.isfinite(
                actual
            )
        ]

        if len(actual) == 0:

            raise ValueError(
                f"No valid OOF actual values "
                f"for {horizon}"
            )

        p90 = float(
            np.percentile(
                actual,
                90
            )
        )

        p95 = float(
            np.percentile(
                actual,
                95
            )
        )

        return {

            "p90":
                p90,

            "p95":
                p95,

            "samples":
                int(
                    len(actual)
                ),
        }


    @staticmethod
    def _level(
        value: float,
        p90: float,
        p95: float,
    ) -> str:

        if value >= p95:

            return "High"

        if value >= p90:

            return "Elevated"

        return "Normal"


    def assess(
        self,
        horizon: str,
        point_forecast: float,
        safety_upper_estimate: float,
    ) -> dict:

        if horizon not in self.thresholds:

            raise ValueError(
                f"Unsupported horizon: {horizon}"
            )

        threshold = (
            self.thresholds[
                horizon
            ]
        )

        p90 = float(
            threshold[
                "p90"
            ]
        )

        p95 = float(
            threshold[
                "p95"
            ]
        )

        point_forecast = float(
            point_forecast
        )

        safety_upper_estimate = float(
            safety_upper_estimate
        )

        point_level = (
            self._level(
                point_forecast,
                p90,
                p95,
            )
        )

        safety_level = (
            self._level(
                safety_upper_estimate,
                p90,
                p95,
            )
        )

        # Final risk uses the safety-adjusted estimate
        # because this layer is intended to catch possible
        # underprediction.
        final_level = safety_level

        if (
            point_level == "Normal"
            and safety_level
            in (
                "Elevated",
                "High",
            )
        ):

            trigger = (
                "Safety upper estimate crossed "
                "historical high-PM threshold"
            )

        elif point_level == "Elevated" and (
            safety_level == "High"
        ):

            trigger = (
                "Safety adjustment raised risk "
                "from Elevated to High"
            )

        else:

            trigger = (
                "Point forecast and safety estimate "
                "are in compatible risk ranges"
            )

        return {

            "level":
                final_level,

            "point_forecast_level":
                point_level,

            "safety_adjusted_level":
                safety_level,

            "p90_threshold":
                p90,

            "p95_threshold":
                p95,

            "reference_value":
                safety_upper_estimate,

            "reference":
                "safety_upper_estimate",

            "trigger":
                trigger,

            "method":
                (
                    "OOF empirical PM2.5 "
                    "P90/P95 thresholds"
                ),

            "oof_samples":
                threshold[
                    "samples"
                ],

            "note":
                (
                    "This is a model-risk indicator based "
                    "on the historical OOF PM2.5 distribution. "
                    "It is not a WHO or US EPA health category."
                ),
        }