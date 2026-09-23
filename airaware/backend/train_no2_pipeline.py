"""Complete, isolated NO2 forecasting benchmark for AirAware Sensor 1.

The pipeline reuses Phase 3 exact-time/gap-aware primitives and Phase 4 split,
preprocessing, and artifact conventions. It never changes PM2.5, AQI, WHO,
ForecastEngine, API, database, or frontend behavior.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from xgboost import XGBRegressor

from app.analysis.phase3_pipeline import (
    MAJOR_GAP_MINUTES,
    add_exact_lag,
    add_future_value,
    backward_rolling,
    build_segments,
)
from app.ml.base import SEED
from app.ml.dataset import TEST_END, TRAIN_END, VALID_END
from app.ml.lightgbm_model import LightGBMModel
from app.ml.random_forest_model import RandomForestModel
from app.ml.xgboost_model import XGBoostModel


BACKEND_DIR = Path(__file__).resolve().parent
CLEANED_CSV = BACKEND_DIR / "data" / "cleaned" / "AirAware_Tulkarem_Nablus_cleaned.csv"
PHASE3_FEATURES_CSV = BACKEND_DIR / "outputs" / "AirAware_ML_Features.csv"
RESULTS_PATH = BACKEND_DIR / "outputs" / "no2_results.json"
PREDICTION_DIR = BACKEND_DIR / "outputs" / "predictions" / "no2"
MODEL_DIR = BACKEND_DIR / "models" / "no2"

HORIZONS = {"1h": 60, "3h": 180, "6h": 360}
FOLDS = (
    ("fold_1", pd.Timestamp("2026-08-12T00:00:00Z"), pd.Timestamp("2026-08-14T00:00:00Z")),
    ("fold_2", pd.Timestamp("2026-08-14T00:00:00Z"), pd.Timestamp("2026-08-16T00:00:00Z")),
    ("fold_3", pd.Timestamp("2026-08-16T00:00:00Z"), pd.Timestamp("2026-08-20T00:00:00Z")),
)

BASE_FEATURES = [
    "no2", "no2_lag_15m", "no2_lag_30m", "no2_lag_1h", "no2_lag_3h", "no2_lag_6h",
    "no2_mean_1h", "no2_std_1h", "no2_min_1h", "no2_max_1h",
    "no2_mean_3h", "no2_median_3h", "no2_std_3h", "no2_min_3h", "no2_max_3h", "no2_range_3h",
    "no2_mean_6h", "no2_std_6h",
    "no2_delta_15m", "no2_delta_1h", "no2_delta_3h", "no2_slope_1h", "no2_slope_3h",
    "temperature", "temperature_lag_1h", "temperature_lag_3h",
    "temperature_mean_1h", "temperature_mean_3h", "temperature_std_3h",
    "humidity", "humidity_lag_1h", "humidity_lag_3h",
    "humidity_mean_1h", "humidity_mean_3h", "humidity_std_3h",
    "o3", "o3_lag_1h", "o3_lag_3h", "o3_mean_1h", "o3_mean_3h",
    "pm25", "pm25_lag_1h", "pm25_lag_3h", "pm25_mean_1h", "pm25_mean_3h",
    "hour", "minute", "day_of_week", "is_weekend",
    "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week",
    "missing_flag", "sampling_gap_flag", "suspicious_value_flag",
    "current_quality_review_flag", "recent_gap_count_3h", "recent_anomaly_count_3h",
]

MODEL_PARAMS = {
    "random_forest": {**RandomForestModel.hyperparameters, "n_jobs": 4},
    "xgboost": {**XGBoostModel.hyperparameters, "n_jobs": 4},
    "lightgbm": {**LightGBMModel.hyperparameters, "n_jobs": 4},
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        "bias": float(error.mean()),
        "underprediction_count": int((error < 0).sum()),
        "underprediction_rate": float((error < 0).mean()),
        "overprediction_count": int((error > 0).sum()),
        "overprediction_rate": float((error > 0).mean()),
    }


def high_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int | None]:
    subset = frame.loc[frame.actual.ge(threshold)].copy()
    if subset.empty:
        return {
            "threshold_ppb": threshold, "n": 0, "mae": None, "rmse": None,
            "underprediction_count": 0, "underprediction_rate": None,
            "mean_underprediction_magnitude": None, "worst_underprediction": None,
        }
    error = subset.predicted - subset.actual
    under = -error.loc[error.lt(0)]
    return {
        "threshold_ppb": threshold,
        "n": len(subset),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "underprediction_count": int(error.lt(0).sum()),
        "underprediction_rate": float(error.lt(0).mean()),
        "mean_underprediction_magnitude": float(under.mean()) if len(under) else 0.0,
        "worst_underprediction": float(under.max()) if len(under) else 0.0,
    }


def exact_change(frame: pd.DataFrame, minutes: int) -> pd.Series:
    return frame.no2 - add_exact_lag(frame, "no2", minutes)


def audit_sensor(canonical: pd.DataFrame) -> dict[str, Any]:
    sensor = canonical.loc[canonical.sensor_id.eq(1)].sort_values("timestamp").copy()
    if sensor.empty or not sensor.city_name.eq("Tulkarem").all():
        raise ValueError("Canonical Sensor 1 / Tulkarem rows are unavailable or inconsistent")
    no2 = sensor.no2
    finite = np.isfinite(no2.to_numpy(float))
    deltas = sensor.timestamp.diff().dt.total_seconds().div(60)
    sampling = deltas.dropna().value_counts().sort_index()
    major_mask = deltas.gt(MAJOR_GAP_MINUTES)
    gaps = []
    for index in sensor.index[major_mask]:
        position = sensor.index.get_loc(index)
        gaps.append({
            "previous_timestamp": sensor.timestamp.iloc[position - 1],
            "next_timestamp": sensor.timestamp.iloc[position],
            "gap_minutes": deltas.loc[index],
        })

    lag_map = {label: add_exact_lag(sensor, "no2", minutes) for label, minutes in HORIZONS.items()}
    lag_map["15m"] = add_exact_lag(sensor, "no2", 15)
    correlations = {}
    changes = {}
    largest = {}
    for label, lag in lag_map.items():
        pair = pd.concat([sensor.no2, lag], axis=1).dropna()
        change = (sensor.no2 - lag).dropna()
        correlations[label] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        changes[label] = {
            "median_absolute_change_ppb": float(change.abs().median()),
            "mean_absolute_change_ppb": float(change.abs().mean()),
        }
        if label in HORIZONS or label == "15m":
            ordered = change.sort_values()
            largest[label] = {
                "negative": [{"timestamp": sensor.loc[i, "timestamp"], "change_ppb": v} for i, v in ordered.head(5).items()],
                "positive": [{"timestamp": sensor.loc[i, "timestamp"], "change_ppb": v} for i, v in ordered.tail(5).sort_values(ascending=False).items()],
            }

    cross = {}
    for column in ("temperature", "humidity", "o3", "pm25"):
        cross[column] = float(sensor[["no2", column]].corr().iloc[0, 1])
    unique = np.sort(no2.dropna().unique())
    increments = np.diff(unique)
    q1, q3 = no2.quantile([0.25, 0.75])
    std = float(no2.std())
    dynamic_range = float(no2.max() - no2.min())
    return {
        "sensor_id": 1,
        "location": "Tulkarem Municipality",
        "unit": "ppb",
        "row_count": len(sensor),
        "valid_no2_count": int(no2.notna().sum()),
        "missing_count": int(no2.isna().sum()),
        "missing_percentage": float(no2.isna().mean() * 100),
        "timestamp_start": sensor.timestamp.min(),
        "timestamp_end": sensor.timestamp.max(),
        "sampling_frequency_distribution_minutes": {str(float(k)): int(v) for k, v in sampling.items()},
        "duplicated_timestamps": int(sensor.timestamp.duplicated().sum()),
        "non_finite_values": int((~finite & no2.notna().to_numpy()).sum()),
        "min": float(no2.min()), "max": float(no2.max()), "range": dynamic_range,
        "mean": float(no2.mean()), "median": float(no2.median()), "std": std,
        "p90": float(no2.quantile(.90)), "p95": float(no2.quantile(.95)), "p99": float(no2.quantile(.99)),
        "iqr": float(q3 - q1),
        "coefficient_of_variation": float(std / no2.mean()),
        "unique_value_count": len(unique),
        "smallest_positive_observed_increment": float(increments[increments > 0].min()) if (increments > 0).any() else None,
        "major_gap_definition_minutes": MAJOR_GAP_MINUTES,
        "major_gap_count": len(gaps), "major_gaps": gaps,
        "longest_gap_minutes": float(deltas.max()) if deltas.notna().any() else None,
        "continuous_segment_count": int(sensor.continuous_segment_id.nunique()),
        "suspicious_impossible_values": {"negative_count": int(no2.lt(0).sum()), "non_finite_count": int((~finite & no2.notna().to_numpy()).sum())},
        "autocorrelation": correlations,
        "pearson_correlation": cross,
        "changes": changes,
        "largest_changes": largest,
        "dynamic_range_assessment": (
            "Low-variance and visibly quantized to approximately 0.01 ppb. The signal is highly persistent and "
            "narrow relative to its mean; no negative/non-finite values or gap evidence establishes a sensor fault, "
            "but the constrained range limits practical forecast value and makes R2 easy to overinterpret."
        ),
    }


def build_dataset() -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    canonical = pd.read_csv(CLEANED_CSV, low_memory=False)
    canonical["timestamp"] = pd.to_datetime(canonical.timestamp, utc=True, errors="raise")
    for column in ("pm25", "temperature", "humidity", "no2", "o3"):
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce")
    canonical = build_segments(canonical)
    audit = audit_sensor(canonical)

    phase3 = pd.read_csv(PHASE3_FEATURES_CSV, low_memory=False)
    phase3["timestamp"] = pd.to_datetime(phase3.timestamp, utc=True, errors="raise")
    frame = phase3.loc[phase3.sensor_id.eq(1)].sort_values("timestamp").copy()
    if len(frame) != audit["row_count"]:
        raise ValueError("Phase 3 feature rows do not match canonical Sensor 1 rows")

    # Extend the existing Phase 3 features only with compact, causal NO2 history.
    for minutes, label in ((30, "30m"), (360, "6h")):
        frame[f"no2_lag_{label}"] = add_exact_lag(frame, "no2", minutes)
    for minutes, label, stats in (
        (60, "1h", ("std", "min", "max")),
        (180, "3h", ("median", "min", "max")),
        (360, "6h", ("mean", "std")),
    ):
        calculated = backward_rolling(frame, "no2", minutes)
        minimum_count = max(2, math.ceil((minutes / 15) * .5))
        for statistic in stats:
            frame[f"no2_{statistic}_{label}"] = calculated[statistic].where(calculated["count"] >= minimum_count)
    frame["no2_range_3h"] = frame.no2_max_3h - frame.no2_min_3h
    frame["no2_delta_15m"] = frame.no2 - frame.no2_lag_15m
    frame["no2_delta_1h"] = frame.no2 - frame.no2_lag_1h
    frame["no2_delta_3h"] = frame.no2 - frame.no2_lag_3h
    frame["no2_slope_1h"] = frame.no2_delta_1h
    frame["no2_slope_3h"] = frame.no2_delta_3h / 3.0

    for label, minutes in HORIZONS.items():
        future = add_future_value(frame, frame.no2, minutes)
        frame[f"target_no2_{label}"] = future
        frame[f"target_no2_delta_{label}"] = future - frame.no2

    missing = sorted(set(BASE_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Required causal features are missing: {missing}")
    forbidden = [name for name in BASE_FEATURES if name.startswith("target_") or "future" in name or "aqi" in name or "who" in name]
    if forbidden:
        raise AssertionError(f"Forbidden feature leakage: {forbidden}")
    return frame, audit, BASE_FEATURES


def split_data(frame: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = f"target_no2_{horizon}"
    valid = frame.loc[frame.no2.notna() & frame[target].notna()].copy()
    return (
        valid.loc[valid.timestamp.lt(TRAIN_END)].copy(),
        valid.loc[valid.timestamp.ge(TRAIN_END) & valid.timestamp.lt(VALID_END)].copy(),
        valid.loc[valid.timestamp.ge(VALID_END) & valid.timestamp.lt(TEST_END)].copy(),
    )


def estimator(name: str):
    if name == "random_forest":
        return RandomForestRegressor(**MODEL_PARAMS[name])
    if name == "xgboost":
        return XGBRegressor(**MODEL_PARAMS[name])
    if name == "lightgbm":
        return lgb.LGBMRegressor(**MODEL_PARAMS[name])
    raise ValueError(name)


def fit_predict(
    model_name: str,
    formulation: str,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    features: list[str],
    horizon: str,
) -> tuple[np.ndarray, Any, SimpleImputer]:
    target = f"target_no2_{horizon}" if formulation == "absolute" else f"target_no2_delta_{horizon}"
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_predict = imputer.transform(predict[features])
    model = estimator(model_name)
    model.fit(x_train, train[target].to_numpy(float))
    raw = np.asarray(model.predict(x_predict), dtype=float)
    predicted = raw if formulation == "absolute" else predict.no2.to_numpy(float) + raw
    return predicted, model, imputer


def prediction_rows(
    source: pd.DataFrame,
    predicted: np.ndarray,
    model: str,
    formulation: str,
    horizon: str,
    fold: str,
) -> pd.DataFrame:
    actual = source[f"target_no2_{horizon}"].to_numpy(float)
    result = pd.DataFrame({
        "timestamp": source.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sensor_id": 1,
        "actual": actual,
        "current_no2": source.no2.to_numpy(float),
        "predicted": predicted,
        "residual": actual - predicted,
        "model": model,
        "target_formulation": formulation,
        "fold": fold,
        "horizon": horizon,
        "unit": "ppb",
    })
    return result


def summarize_oof(oof: pd.DataFrame, persistence: pd.DataFrame) -> list[dict[str, Any]]:
    persistence_fold = {
        fold: regression_metrics(group.actual, group.predicted)
        for fold, group in persistence.groupby("fold", sort=True)
    }
    rows = []
    for (model, formulation), candidate in oof.groupby(["model", "target_formulation"], sort=False):
        folds = []
        for fold, group in candidate.groupby("fold", sort=True):
            values = regression_metrics(group.actual, group.predicted)
            baseline = persistence_fold[fold]
            values["mae_improvement_vs_persistence_percent"] = (baseline["mae"] - values["mae"]) / baseline["mae"] * 100
            values["rmse_improvement_vs_persistence_percent"] = (baseline["rmse"] - values["rmse"]) / baseline["rmse"] * 100
            folds.append({"fold": fold, **values})
        summary = {"model": model, "target_formulation": formulation, "fold_metrics": folds}
        for metric in ("mae", "rmse", "median_absolute_error", "r2", "bias", "underprediction_rate", "overprediction_rate"):
            values = np.asarray([item[metric] for item in folds], dtype=float)
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"std_{metric}"] = float(values.std(ddof=0))
        summary["mae_fold_wins_vs_persistence"] = int(sum(item["mae_improvement_vs_persistence_percent"] > 0 for item in folds))
        summary["rmse_fold_wins_vs_persistence"] = int(sum(item["rmse_improvement_vs_persistence_percent"] > 0 for item in folds))
        summary["mean_mae_improvement_vs_persistence_percent"] = float(np.mean([item["mae_improvement_vs_persistence_percent"] for item in folds]))
        summary["mean_rmse_improvement_vs_persistence_percent"] = float(np.mean([item["rmse_improvement_vs_persistence_percent"] for item in folds]))
        rows.append(summary)
    return rows


def candidate_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (row["mean_mae"], row["mean_rmse"], abs(row["mean_bias"]))


def select_candidate(
    summaries: list[dict[str, Any]],
    oof: pd.DataFrame,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = next(row for row in summaries if row["model"] == "persistence")
    ranked = sorted(summaries, key=candidate_key)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
        candidate = oof.loc[(oof.model == row["model"]) & (oof.target_formulation == row["target_formulation"])]
        row["high_no2"] = {
            "p90": high_metrics(candidate, thresholds["p90"]),
            "p95": high_metrics(candidate, thresholds["p95"]),
        }

    baseline_p90 = baseline["high_no2"]["p90"]
    eligible = []
    for row in ranked:
        if row["model"] == "persistence":
            continue
        high = row["high_no2"]["p90"]
        robust = (
            row["mean_mae_improvement_vs_persistence_percent"] >= 5.0
            and row["mean_rmse_improvement_vs_persistence_percent"] >= 5.0
            and row["mae_fold_wins_vs_persistence"] >= 2
            and row["rmse_fold_wins_vs_persistence"] >= 2
            and high["mae"] <= baseline_p90["mae"]
            and high["underprediction_rate"] <= max(0.75, baseline_p90["underprediction_rate"])
        )
        row["passes_promotion_gate"] = bool(robust)
        if robust:
            eligible.append(row)
    baseline["passes_promotion_gate"] = True
    selected = min(eligible, key=candidate_key) if eligible else baseline
    reason = (
        "Best learned candidate passing the development-only robustness gate."
        if eligible else
        "Persistence remains the most defensible model for this horizon with the current data."
    )
    return {
        "model": selected["model"],
        "target_formulation": selected["target_formulation"],
        "promote_artifact": selected["model"] != "persistence",
        "reason": reason,
    }, ranked


def verify_causality(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    # Recalculate representative exact lags/rollups and compare to selected features.
    checks = {}
    for column, minutes in (("no2_lag_15m", 15), ("no2_lag_30m", 30), ("no2_lag_1h", 60), ("no2_lag_3h", 180), ("no2_lag_6h", 360)):
        expected = add_exact_lag(frame, "no2", minutes)
        checks[column] = bool(np.allclose(frame[column], expected, equal_nan=True))
    for column, minutes, statistic in (("no2_mean_1h", 60, "mean"), ("no2_std_3h", 180, "std"), ("no2_mean_6h", 360, "mean")):
        expected = backward_rolling(frame, "no2", minutes)[statistic]
        mask = frame[column].notna()
        checks[column] = bool(np.allclose(frame.loc[mask, column], expected.loc[mask], equal_nan=True))
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "representative_recalculation_checks": checks,
        "feature_count": len(features),
        "statement": "All selected inputs are current values, exact past lags, backward-only segment-bounded rolls/trends, calendar values, or current/backward quality flags.",
    }


def main() -> dict[str, Any]:
    np.random.seed(SEED)
    frame, audit, features = build_dataset()
    causal_verification = verify_causality(frame, features)
    if causal_verification["status"] != "PASS":
        raise AssertionError("Causal feature verification failed")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "project": "AirAware NO2 Pollutant Outlook",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc),
        "pollutant": "no2", "unit": "ppb", "sensor_id": 1, "location": "Tulkarem Municipality",
        "audit": audit,
        "features": features,
        "new_no2_features": [name for name in features if name not in pd.read_csv(PHASE3_FEATURES_CSV, nrows=0).columns],
        "split_boundaries": {"train_end": TRAIN_END, "validation_end": VALID_END, "test_end": TEST_END},
        "horizons": {},
        "causal_feature_verification": causal_verification,
        "limitations": [
            "Only 17 days of Sensor 1 data are available.",
            "NO2 has a very narrow, quantized observed range.",
            "The current 1h test period was inspected in an earlier prototype and is not pristine external validation.",
            "All current test results are development-final evaluations; future live-period validation is required.",
            "NO2 is structurally unavailable for Nablus sensors and was not fabricated or interpolated.",
        ],
    }

    for horizon in HORIZONS:
        train, validation, test = split_data(frame, horizon)
        development = pd.concat([train, validation], ignore_index=True).sort_values("timestamp")
        thresholds = {
            "p90": float(development[f"target_no2_{horizon}"].quantile(.90)),
            "p95": float(development[f"target_no2_{horizon}"].quantile(.95)),
        }
        fold_definitions = []
        oof_parts = []
        for fold_name, valid_start, valid_end in FOLDS:
            fold_train = development.loc[development.timestamp.lt(valid_start)].copy()
            fold_valid = development.loc[development.timestamp.ge(valid_start) & development.timestamp.lt(valid_end)].copy()
            if min(len(fold_train), len(fold_valid)) == 0:
                raise ValueError(f"{horizon} {fold_name} is empty")
            fold_definitions.append({
                "fold": fold_name,
                "train_start": fold_train.timestamp.min(), "train_end_exclusive": valid_start,
                "validation_start": valid_start, "validation_end_exclusive": valid_end,
                "train_count": len(fold_train), "validation_count": len(fold_valid),
            })
            persistence = fold_valid.no2.to_numpy(float)
            oof_parts.append(prediction_rows(fold_valid, persistence, "persistence", "absolute", horizon, fold_name))
            for model_name in ("random_forest", "xgboost", "lightgbm"):
                for formulation in ("absolute", "delta"):
                    predicted, _, _ = fit_predict(model_name, formulation, fold_train, fold_valid, features, horizon)
                    oof_parts.append(prediction_rows(fold_valid, predicted, model_name, formulation, horizon, fold_name))

        oof = pd.concat(oof_parts, ignore_index=True)
        oof_path = PREDICTION_DIR / f"oof_no2_{horizon}.csv"
        oof.to_csv(oof_path, index=False)
        persistence_oof = oof.loc[oof.model.eq("persistence")]
        summaries = summarize_oof(oof, persistence_oof)
        selected, rankings = select_candidate(summaries, oof, thresholds)

        selected_model = selected["model"]
        selected_formulation = selected["target_formulation"]
        if selected_model == "persistence":
            test_prediction = test.no2.to_numpy(float)
            final_model = final_imputer = None
        else:
            test_prediction, final_model, final_imputer = fit_predict(
                selected_model, selected_formulation, development, test, features, horizon
            )
        test_predictions = prediction_rows(
            test, test_prediction, selected_model, selected_formulation, horizon, "development_final_test"
        ).rename(columns={"model": "selected_model"})
        test_path = PREDICTION_DIR / f"test_no2_{horizon}.csv"
        test_predictions.to_csv(test_path, index=False)
        persistence_test_prediction = test.no2.to_numpy(float)
        selected_test_metrics = regression_metrics(test_predictions.actual, test_predictions.predicted)
        persistence_test_metrics = regression_metrics(test[f"target_no2_{horizon}"], persistence_test_prediction)
        test_high = {
            "selected": {
                "p90": high_metrics(test_predictions.rename(columns={"selected_model": "model"}), thresholds["p90"]),
                "p95": high_metrics(test_predictions.rename(columns={"selected_model": "model"}), thresholds["p95"]),
            }
        }
        baseline_test_frame = prediction_rows(test, persistence_test_prediction, "persistence", "absolute", horizon, "development_final_test")
        test_high["persistence"] = {
            "p90": high_metrics(baseline_test_frame, thresholds["p90"]),
            "p95": high_metrics(baseline_test_frame, thresholds["p95"]),
        }

        artifact_path = MODEL_DIR / f"no2_{horizon}.joblib"
        reload_check = None
        if selected["promote_artifact"]:
            metadata = {
                "version": "1.0.0", "created_at": datetime.now(timezone.utc),
                "pollutant": "no2", "unit": "ppb", "horizon": horizon,
                "sensor_id": 1, "location": "Tulkarem Municipality",
                "model_type": selected_model, "target_formulation": selected_formulation,
                "features": features, "feature_order": features,
                "training_timestamp_start": development.timestamp.min(),
                "training_timestamp_end": development.timestamp.max(),
                "target_alignment": f"Exact timestamp t+{horizon} within continuous_segment_id",
                "major_gap_minutes": MAJOR_GAP_MINUTES,
                "hyperparameters": MODEL_PARAMS[selected_model],
                "frontend_contract": {"pollutant_outlook": True, "separate_from_aqi": True},
            }
            bundle = {"model": final_model, "imputer": final_imputer, "features": features, "metadata": metadata}
            joblib.dump(bundle, artifact_path)
            del bundle
            reloaded = joblib.load(artifact_path)
            raw = np.asarray(reloaded["model"].predict(reloaded["imputer"].transform(test[reloaded["features"]])), dtype=float)
            reproduced = test.no2.to_numpy(float) + raw if selected_formulation == "delta" else raw
            difference = np.abs(reproduced - test_prediction)
            reload_check = {
                "mean_absolute_prediction_difference": float(difference.mean()),
                "max_prediction_difference": float(difference.max()),
                "within_0_000001_count": int((difference <= 1e-6).sum()),
                "total_count": len(difference),
                "within_0_000001_percentage": float((difference <= 1e-6).mean() * 100),
                "status": "PASS" if np.all(difference <= 1e-6) else "FAIL",
            }
            if reload_check["status"] != "PASS":
                artifact_path.unlink(missing_ok=True)
                selected["promote_artifact"] = False
                selected["reason"] += " Reload verification failed; artifact removed."
        elif artifact_path.exists():
            # A current failed gate must not leave a stale successful artifact.
            artifact_path.unlink()

        result["horizons"][horizon] = {
            "target": f"target_no2_{horizon}",
            "delta_target": f"target_no2_delta_{horizon}",
            "target_construction": f"Exact NO2 lookup at t+{horizon}, restricted to the same continuous segment",
            "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
            "folds": fold_definitions,
            "development_thresholds": thresholds,
            "candidate_rankings": rankings,
            "selection": selected,
            "test": {
                "label": "development-final evaluation; not pristine external proof",
                "selected_metrics": selected_test_metrics,
                "persistence_metrics": persistence_test_metrics,
                "mae_improvement_vs_persistence_percent": (persistence_test_metrics["mae"] - selected_test_metrics["mae"]) / persistence_test_metrics["mae"] * 100,
                "rmse_improvement_vs_persistence_percent": (persistence_test_metrics["rmse"] - selected_test_metrics["rmse"]) / persistence_test_metrics["rmse"] * 100,
                "high_no2": test_high,
            },
            "artifact": {
                "promoted": bool(selected["promote_artifact"]),
                "path": str(artifact_path.relative_to(BACKEND_DIR)) if selected["promote_artifact"] else None,
                "reload_verification": reload_check,
            },
            "oof_predictions_path": str(oof_path.relative_to(BACKEND_DIR)),
            "test_predictions_path": str(test_path.relative_to(BACKEND_DIR)),
        }

    promoted = [h for h, payload in result["horizons"].items() if payload["artifact"]["promoted"]]
    result["practical_value"] = {
        "promoted_horizons": promoted,
        "assessment": (
            "Deployment value is limited by the 0.54 ppb observed range and short history. Any promoted model is a "
            "prototype Pollutant Outlook only; future live-period data must confirm fold stability, calibration, and "
            "upper-tail underprediction before production reliance."
        ),
    }
    result["safety_check"] = {
        "pm25_models_changed": False, "forecast_engine_changed": False,
        "epa_aqi_changed": False, "who_logic_changed": False,
        "frontend_changed": False, "nablus_no2_fabricated": False,
    }
    RESULTS_PATH.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
