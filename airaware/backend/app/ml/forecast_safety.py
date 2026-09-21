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


class ForecastSafety:

    def __init__(
        self,
        quantile: float = 0.95,
    ):

        self.quantile = quantile
        self.margins = {}

        for horizon in (
            "1h",
            "3h",
            "6h",
        ):
            self.margins[horizon] = (
                self._load_margin(
                    horizon
                )
            )


    def _load_margin(
        self,
        horizon: str,
    ) -> float:

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

        required = {
            "actual",
            "predicted",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{path.name} missing columns: "
                f"{sorted(missing)}"
            )

        actual = pd.to_numeric(
            df["actual"],
            errors="coerce",
        )

        predicted = pd.to_numeric(
            df["predicted"],
            errors="coerce",
        )

        valid = (
            actual.notna()
            & predicted.notna()
        )

        actual = actual[valid].to_numpy(
            dtype=float
        )

        predicted = predicted[valid].to_numpy(
            dtype=float
        )

        if len(actual) == 0:
            raise ValueError(
                f"No valid OOF rows for {horizon}"
            )

        # ----------------------------------------------------
        # Positive residual:
        #
        # actual - prediction
        #
        # Positive means model UNDERESTIMATED pollution.
        #
        # We only care about the upper-risk direction here.
        # ----------------------------------------------------

        underprediction_residual = (
            actual
            - predicted
        )

        positive_residual = np.maximum(
            underprediction_residual,
            0.0,
        )

        margin = float(
            np.quantile(
                positive_residual,
                self.quantile,
            )
        )

        return margin


    def apply(
        self,
        horizon: str,
        prediction: float,
    ) -> dict:

        prediction = float(
            prediction
        )

        margin = float(
            self.margins[horizon]
        )

        upper = (
            prediction
            + margin
        )

        return {
            "method":
                "OOF one-sided empirical residual",

            "quantile":
                self.quantile,

            "safety_margin":
                margin,

            "safety_upper_estimate":
                upper,

            "note":
                (
                    "Upper estimate uses the "
                    "95th percentile of positive "
                    "OOF residuals (actual - predicted). "
                    "It is a safety estimate, not a formal "
                    "probabilistic confidence interval."
                ),
        }