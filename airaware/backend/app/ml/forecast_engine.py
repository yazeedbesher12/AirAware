from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .epa_aqi import pm25_to_aqi

from .forecast_interpreter import (
    interpret_who_pm25,
)

from .forecast_safety import (
    ForecastSafety,
)

from .high_pm_risk import (
    HighPMRisk,
)


BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_DIR = (
    BACKEND_DIR
    / "models"
    / "lightgbm"
)


MODEL_PATHS = {

    "1h":
        MODEL_DIR
        / "pm25_1h.joblib",

    "3h":
        MODEL_DIR
        / "pm25_3h.joblib",

    "6h":
        MODEL_DIR
        / "pm25_6h.joblib",
}


class ForecastEngine:

    def __init__(self):

        self.models = {}

        # ====================================================
        # LOAD THREE LIGHTGBM MODELS
        # ====================================================

        for horizon, path in (
            MODEL_PATHS.items()
        ):

            if not path.exists():

                raise FileNotFoundError(
                    f"Model not found:\n{path}"
                )

            self.models[
                horizon
            ] = joblib.load(
                path
            )

        # ====================================================
        # SAFETY CALIBRATION
        # ====================================================

        self.safety = ForecastSafety(
            quantile=0.95
        )

        # ====================================================
        # HIGH-PM RISK CALIBRATION
        # ====================================================

        self.high_pm_risk = (
            HighPMRisk()
        )


    def _predict_one(
        self,
        horizon: str,
        feature_row: pd.DataFrame,
    ) -> dict:

        bundle = (
            self.models[
                horizon
            ]
        )

        features = (
            bundle[
                "features"
            ]
        )

        # ====================================================
        # CHECK FEATURES
        # ====================================================

        missing = [

            feature

            for feature
            in features

            if feature
            not in feature_row.columns
        ]

        if missing:

            raise ValueError(
                f"{horizon} missing features:\n"
                + "\n".join(
                    missing
                )
            )

        # ====================================================
        # PREPARE INPUT
        # ====================================================

        X = feature_row[
            features
        ].copy()

        for column in features:

            X[
                column
            ] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        # Same exact imputer fitted during training
        X = (
            bundle[
                "imputer"
            ]
            .transform(
                X
            )
        )

        # ====================================================
        # MODEL 0
        #
        # Future PM2.5 concentration
        # ====================================================

        pm25_forecast = float(

            bundle[
                "models"
            ][0]

            .predict(
                X
            )[0]
        )

        # ====================================================
        # MODEL 1
        #
        # Future PM2.5 24-hour average
        # ====================================================

        pm25_24h_average = float(

            bundle[
                "models"
            ][1]

            .predict(
                X
            )[0]
        )

        # ====================================================
        # GENERAL SAFETY LAYER
        # ====================================================

        safety = (
            self.safety.apply(

                horizon=
                    horizon,

                prediction=
                    pm25_forecast,
            )
        )

        # ====================================================
        # HIGH-PM RISK
        # ====================================================

        high_pm = (
            self.high_pm_risk.assess(

                horizon=
                    horizon,

                point_forecast=
                    pm25_forecast,

                safety_upper_estimate=
                    safety[
                        "safety_upper_estimate"
                    ],
            )
        )

        # ====================================================
        # WHO INTERPRETATION
        # ====================================================

        who = (
            interpret_who_pm25(
                pm25_24h_average
            )
        )

        # ====================================================
        # US EPA PM2.5 AQI
        # ====================================================

        epa = (
            pm25_to_aqi(
                pm25_24h_average
            )
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        return {

            "horizon":
                horizon,

            # ------------------------------------------------
            # Main forecast
            # ------------------------------------------------

            "pm25": {

                "forecast":
                    pm25_forecast,

                "unit":
                    "µg/m³",

                "safety_upper_estimate":
                    safety[
                        "safety_upper_estimate"
                    ],

                "safety_margin":
                    safety[
                        "safety_margin"
                    ],
            },

            # ------------------------------------------------
            # Future health-average prediction
            # ------------------------------------------------

            "future_24h_average": {

                "forecast":
                    pm25_24h_average,

                "unit":
                    "µg/m³",
            },

            # ------------------------------------------------
            # High-PM statistical risk
            # ------------------------------------------------

            "high_pm_risk": {

                "level":
                    high_pm[
                        "level"
                    ],

                "point_forecast_level":
                    high_pm[
                        "point_forecast_level"
                    ],

                "safety_adjusted_level":
                    high_pm[
                        "safety_adjusted_level"
                    ],

                "p90_threshold":
                    high_pm[
                        "p90_threshold"
                    ],

                "p95_threshold":
                    high_pm[
                        "p95_threshold"
                    ],

                "trigger":
                    high_pm[
                        "trigger"
                    ],

                "method":
                    high_pm[
                        "method"
                    ],
            },

            # ------------------------------------------------
            # WHO
            # ------------------------------------------------

            "who": {

                "status":
                    who[
                        "status"
                    ],

                "guideline":
                    who[
                        "guideline"
                    ],

                "ratio_to_guideline":
                    who[
                        "ratio_to_guideline"
                    ],

                "averaging_period":
                    "24h",
            },

            # ------------------------------------------------
            # EPA AQI
            # ------------------------------------------------

            "epa_aqi": {

                "value":
                    epa[
                        "aqi"
                    ],

                "category":
                    epa[
                        "category"
                    ],

                "basis":
                    "PM2.5",

                # We currently do not forecast
                # all possible AQI pollutants.
                "overall_aqi":
                    False,
            },

            # ------------------------------------------------
            # Uncertainty metadata
            # ------------------------------------------------

            "uncertainty": {

                "method":
                    safety[
                        "method"
                    ],

                "quantile":
                    safety[
                        "quantile"
                    ],

                "note":
                    safety[
                        "note"
                    ],
            },
        }


    def predict(
        self,
        feature_row: pd.DataFrame,
    ) -> dict:

        if len(
            feature_row
        ) != 1:

            raise ValueError(
                "ForecastEngine expects "
                "exactly one feature row."
            )

        forecasts = {}

        for horizon in (
            "1h",
            "3h",
            "6h",
        ):

            forecasts[
                horizon
            ] = (
                self._predict_one(

                    horizon=
                        horizon,

                    feature_row=
                        feature_row,
                )
            )

        return forecasts