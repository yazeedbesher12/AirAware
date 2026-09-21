from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


WHO_PM25_24H = 15.0
MIN_SENSOR_TEST_SAMPLES = 20


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    y, p = actual[mask], predicted[mask]
    if not len(y):
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "mape": None, "median_absolute_error": None}
    mape_mask = np.abs(y) >= 1.0
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) >= 2 else None,
        "mape": float(np.mean(np.abs((y[mape_mask] - p[mape_mask]) / y[mape_mask])) * 100) if mape_mask.any() else None,
        "median_absolute_error": float(median_absolute_error(y, p)),
    }


def event_metrics(actual_average: np.ndarray, predicted_average: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(actual_average) & np.isfinite(predicted_average)
    actual = actual_average[mask] > WHO_PM25_24H
    predicted = predicted_average[mask] > WHO_PM25_24H
    if not len(actual):
        return {"event_n": 0, "event_accuracy": None, "event_precision": None, "event_recall": None, "event_f1": None, "event_specificity": None, "false_positives": 0, "false_negatives": 0}
    tp = int(np.sum(actual & predicted)); tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted)); fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "event_n": int(len(actual)), "event_accuracy": float((tp + tn) / len(actual)),
        "event_precision": precision, "event_recall": recall, "event_f1": f1,
        "event_specificity": specificity, "false_positives": fp, "false_negatives": fn,
    }


def evaluate(actual: np.ndarray, predicted: np.ndarray, actual_average: np.ndarray, predicted_average: np.ndarray) -> dict[str, Any]:
    return {**regression_metrics(actual, predicted), **event_metrics(actual_average, predicted_average)}


def sensor_metrics(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for sensor, group in predictions.groupby("sensor_id"):
        if len(group) < MIN_SENSOR_TEST_SAMPLES:
            results.append({"sensor_id": int(sensor), "status": "INSUFFICIENT_TEST_DATA", "n": int(len(group))})
            continue
        metrics = evaluate(group.actual.to_numpy(float), group.predicted.to_numpy(float), group.actual_health_average.to_numpy(float), group.predicted_health_average.to_numpy(float))
        results.append({"sensor_id": int(sensor), "city": str(group.city.iloc[0]), "status": "OK", **metrics})
    return results


def prediction_frame(source: pd.DataFrame, output: Any, model: str, horizon: str, split: str, fold_id: str) -> pd.DataFrame:
    result = pd.DataFrame({
        "timestamp": source.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sensor_id": source.sensor_id.astype(int), "city": source.city_name.astype(str),
        "target": "pm25", "horizon": horizon,
        "actual": source[f"target_pm25_{horizon}"].to_numpy(float),
        "predicted": output.concentration,
        "actual_health_average": source[f"target_pm25_who_24h_avg_{horizon}"].to_numpy(float),
        "predicted_health_average": output.health_average,
        "split": split, "fold_id": fold_id, "model": model,
    })
    result["absolute_error"] = (result.actual - result.predicted).abs()
    result["residual"] = result.actual - result.predicted
    result["pollution_event_actual"] = result.actual_health_average > WHO_PM25_24H
    result["pollution_event_predicted"] = result.predicted_health_average > WHO_PM25_24H
    return result
