"""AirAware Phase 2 advanced EDA pipeline.

The cleaned Phase 1 dataset is the sole observation source. Phase 2 never
changes or deletes observations and precomputes expensive analyses for the API.
"""

from __future__ import annotations

import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import pacf

from app.config import (
    ANALYSIS_JSON, CLEANED_CSV, ENVIRONMENTAL_FEATURES, FEATURE_UNITS,
    OUTPUT_DIR, PHASE2_ANALYSIS_JSON, PHASE2_ARTIFACTS, REPORT_DIR,
    ensure_directories,
)

PHASE2_VERSION = "2.0.0"
LAG_MINUTES = [15, 30, 60, 120, 180, 360, 720, 1440]
ROLLING_WINDOWS = ["1h", "3h", "6h", "12h", "24h"]
ARTIFACTS = PHASE2_ARTIFACTS


def _clean_text(value: str) -> str:
    """Normalize legacy mojibake sequences without relying on source encoding."""
    return (
        value.replace("\u00e2\u20ac\u201d", "-")
        .replace("\u00e2\u2020\u201d", "<->")
        .replace("\u00e2\u2020\u2019", "->")
    )


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        return _clean_text(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _write_csv(name: str, frame: pd.DataFrame) -> None:
    output = frame.copy()
    for column in output.select_dtypes(include="object"):
        output[column] = output[column].map(lambda value: _clean_text(value) if isinstance(value, str) else value)
    output.to_csv(OUTPUT_DIR / ARTIFACTS[name], index=False, date_format="%Y-%m-%dT%H:%M:%SZ")


def _load() -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_csv(CLEANED_CSV)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for feature in ENVIRONMENTAL_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    phase1 = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    return data.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True), phase1


def _provided_features(phase1: dict[str, Any], sensor_id: int) -> list[str]:
    availability = next(row for row in phase1["availability"] if row["sensor_id"] == int(sensor_id))
    return [feature for feature in ENVIRONMENTAL_FEATURES if availability[feature] != "STRUCTURALLY_UNAVAILABLE"]


def _mode_count(values: pd.Series) -> int:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 40 or values.nunique() < 5:
        return 1
    counts, _ = np.histogram(values, bins=min(40, max(12, int(np.sqrt(len(values))))))
    smoothed = gaussian_filter1d(counts.astype(float), sigma=1.2)
    prominence = max(1.0, smoothed.max() * 0.08)
    peaks, _ = find_peaks(smoothed, prominence=prominence)
    return max(1, int(len(peaks)))


def descriptive_statistics(data: pd.DataFrame, phase1: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    interpretations: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("global", "All sensors", data)]
    scopes += [("city", city, group) for city, group in data.groupby("city_name")]
    scopes += [("sensor", str(int(sensor)), group) for sensor, group in data.groupby("sensor_id")]
    for scope_type, scope_value, group in scopes:
        sensor_id = int(scope_value) if scope_type == "sensor" else None
        allowed = _provided_features(phase1, sensor_id) if sensor_id is not None else ENVIRONMENTAL_FEATURES
        for feature in allowed:
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            q = values.quantile([.01, .05, .10, .25, .50, .75, .90, .95, .99])
            mean = float(values.mean())
            std = float(values.std()) if len(values) > 1 else 0.0
            skew = float(values.skew()) if len(values) > 2 else 0.0
            kurt = float(values.kurt()) if len(values) > 3 else 0.0
            modes = _mode_count(values)
            row = {
                "scope_type": scope_type, "scope_value": scope_value, "feature": feature,
                "count": len(values), "mean": mean, "median": values.median(),
                "variance": values.var(), "std": std, "min": values.min(), "max": values.max(),
                "q1": q.loc[.25], "q3": q.loc[.75], "iqr": q.loc[.75] - q.loc[.25],
                "p01": q.loc[.01], "p05": q.loc[.05], "p10": q.loc[.10],
                "p25": q.loc[.25], "p50": q.loc[.50], "p75": q.loc[.75],
                "p90": q.loc[.90], "p95": q.loc[.95], "p99": q.loc[.99],
                "skewness": skew, "kurtosis": kurt,
                "coefficient_of_variation": std / abs(mean) if mean else None,
                "estimated_modes": modes,
            }
            rows.append(row)
            traits = []
            if abs(skew) >= 1:
                traits.append(f"strongly {'right' if skew > 0 else 'left'}-skewed")
            elif abs(skew) >= .5:
                traits.append(f"moderately {'right' if skew > 0 else 'left'}-skewed")
            if kurt >= 3:
                traits.append("heavy-tailed")
            if modes >= 2:
                traits.append(f"possibly multimodal ({modes} density peaks)")
            if std <= max(abs(mean) * .01, 1e-9):
                traits.append("low-variance")
            if not traits:
                traits.append("no strong skew or tail signal")
            interpretations.append({
                "scope_type": scope_type, "scope_value": scope_value, "feature": feature,
                "text": f"{feature.upper()} is " + ", ".join(traits) + f"; P95={q.loc[.95]:.2f}, P99={q.loc[.99]:.2f}."
            })
    return pd.DataFrame(rows), interpretations


def temporal_patterns(data: pd.DataFrame, phase1: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for sensor_id, group in data.groupby("sensor_id", sort=True):
        for feature in _provided_features(phase1, int(sensor_id)):
            base = group[["timestamp", feature]].dropna().copy()
            if base.empty:
                continue
            base["hour"] = base.timestamp.dt.hour
            base["day_of_week"] = base.timestamp.dt.dayofweek
            base["date"] = base.timestamp.dt.floor("D")
            base["week"] = base.timestamp.dt.tz_localize(None).dt.to_period("W").astype(str)
            base["day_type"] = np.where(base.day_of_week >= 5, "Weekend", "Weekday")
            for hour, values in base.groupby("hour")[feature]:
                rows.append({"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "feature": feature, "pattern_type": "hour", "period": int(hour), "mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "max": values.max(), "count": len(values)})
            for dow, values in base.groupby("day_of_week")[feature]:
                rows.append({"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "feature": feature, "pattern_type": "day_of_week", "period": dow_names[int(dow)], "mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "max": values.max(), "count": len(values)})
            for dtype, values in base.groupby("day_type")[feature]:
                rows.append({"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "feature": feature, "pattern_type": "day_type", "period": dtype, "mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "max": values.max(), "count": len(values)})
            for date, values in base.groupby("date")[feature]:
                rows.append({"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "feature": feature, "pattern_type": "daily", "period": date, "mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "max": values.max(), "count": len(values)})
            for week, values in base.groupby("week")[feature]:
                rows.append({"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "feature": feature, "pattern_type": "weekly", "period": week, "mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "max": values.max(), "count": len(values)})
            hourly = base.groupby("hour")[feature].mean()
            peak_hour, low_hour = int(hourly.idxmax()), int(hourly.idxmin())
            findings.append({"sensor_id": int(sensor_id), "feature": feature, "text": f"Sensor {sensor_id} {feature.upper()} has its highest hourly mean at {peak_hour:02d}:00 UTC ({hourly.max():.2f}) and lowest at {low_hour:02d}:00 UTC ({hourly.min():.2f})."})
    return pd.DataFrame(rows), findings


def _grid_series(group: pd.DataFrame, feature: str, minutes: int = 15) -> pd.Series:
    series = group.set_index("timestamp")[feature].sort_index()
    return series.resample(f"{minutes}min").mean()


def rolling_summary(data: pd.DataFrame, phase1: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for sensor_id, group in data.groupby("sensor_id", sort=True):
        for feature in _provided_features(phase1, int(sensor_id)):
            values = group.set_index("timestamp")[feature].dropna().sort_index()
            if len(values) < 8:
                continue
            row: dict[str, Any] = {"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "feature": feature}
            for window in ROLLING_WINDOWS:
                roll = values.rolling(window, min_periods=2)
                row[f"{window}_std_median"] = roll.std().median()
                row[f"{window}_variance_median"] = roll.var().median()
                row[f"{window}_range_p95"] = (roll.max() - roll.min()).quantile(.95)
            output.append(row)
    return output


def decomposition(data: pd.DataFrame, phase1: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outputs: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for sensor_id, group in data.groupby("sensor_id", sort=True):
        for feature in _provided_features(phase1, int(sensor_id)):
            if feature not in {"pm25", "no2", "o3", "aqi"}:
                continue
            grid = _grid_series(group, feature)
            missing_rate = float(grid.isna().mean())
            if len(grid) < 7 * 96 or missing_rate > .08:
                outputs.append({"sensor_id": int(sensor_id), "feature": feature, "status": "SKIPPED", "reason": f"Insufficient continuous coverage for reliable daily STL (n={len(grid)}, grid missing={missing_rate:.1%}).", "points": []})
                continue
            prepared = grid.interpolate(limit=2, limit_direction="both")
            valid_runs = prepared.notna().ne(prepared.notna().shift()).cumsum()
            segments = [segment for _, segment in prepared.groupby(valid_runs) if segment.notna().all()]
            longest = max(segments, key=len, default=pd.Series(dtype=float))
            if len(longest) < 7 * 96:
                outputs.append({"sensor_id": int(sensor_id), "feature": feature, "status": "SKIPPED", "reason": "No continuous segment of at least seven days after limited short-gap interpolation.", "points": []})
                continue
            fitted = STL(longest, period=96, robust=True).fit()
            sample = pd.DataFrame({"timestamp": longest.index, "observed": longest.values, "trend": fitted.trend, "seasonal": fitted.seasonal, "residual": fitted.resid}).iloc[::4]
            seasonal_strength = max(0.0, 1 - np.var(fitted.resid) / np.var(fitted.resid + fitted.seasonal))
            trend_strength = max(0.0, 1 - np.var(fitted.resid) / np.var(fitted.resid + fitted.trend))
            outputs.append({"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "feature": feature, "status": "AVAILABLE", "period_samples": 96, "seasonal_strength": seasonal_strength, "trend_strength": trend_strength, "points": _records(sample)})
            findings.append({"sensor_id": int(sensor_id), "feature": feature, "text": f"Daily STL for Sensor {sensor_id} {feature.upper()} has seasonal strength {seasonal_strength:.2f} and trend strength {trend_strength:.2f}; residual spikes remain for anomaly review."})
    return outputs, findings


def autocorrelation_analysis(data: pd.DataFrame, phase1: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sensor_id, group in data.groupby("sensor_id", sort=True):
        for feature in _provided_features(phase1, int(sensor_id)):
            grid = _grid_series(group, feature)
            prepared = grid.interpolate(limit=2)
            segments = [seg.dropna() for _, seg in prepared.groupby(prepared.isna().cumsum()) if seg.notna().sum() >= 100]
            longest = max(segments, key=len, default=pd.Series(dtype=float))
            max_lag = min(96, max(1, len(longest) // 4))
            pacf_values = pacf(longest.to_numpy(), nlags=max_lag, method="ywm") if len(longest) > max_lag + 5 and longest.nunique() > 2 else np.full(max_lag + 1, np.nan)
            for minutes in LAG_MINUTES:
                steps = minutes // 15
                paired = pd.concat([grid.rename("current"), grid.shift(steps).rename("lagged")], axis=1).dropna()
                if len(paired) < 30:
                    continue
                corr = paired.current.corr(paired.lagged)
                rows.append({"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "feature": feature, "lag_minutes": minutes, "lag": f"{minutes} min" if minutes < 60 else f"{minutes//60} h", "autocorrelation": corr, "absolute_autocorrelation": abs(corr), "pacf": pacf_values[steps] if steps < len(pacf_values) else None, "n_observations": len(paired)})
    return pd.DataFrame(rows).sort_values("absolute_autocorrelation", ascending=False)


def correlations(data: pd.DataFrame, phase1: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson_rows, spearman_rows = [], []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("global", "All sensors", data)]
    scopes += [("city", city, group) for city, group in data.groupby("city_name")]
    scopes += [("sensor", str(int(sensor)), group) for sensor, group in data.groupby("sensor_id")]
    for scope_type, scope_value, group in scopes:
        features = [f for f in ENVIRONMENTAL_FEATURES if group[f].notna().sum() >= 30]
        for left, right in itertools.combinations(features, 2):
            pair = group[[left, right]].dropna()
            if len(pair) < 30 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
                continue
            p = pair[left].corr(pair[right], method="pearson")
            s = pair[left].corr(pair[right], method="spearman")
            common = {"scope_type": scope_type, "scope_value": scope_value, "feature_a": left, "feature_b": right, "n_observations": len(pair)}
            pearson_rows.append({**common, "correlation": p, "absolute_correlation": abs(p)})
            spearman_rows.append({**common, "correlation": s, "absolute_correlation": abs(s), "pearson_correlation": p, "spearman_minus_pearson": abs(s) - abs(p)})
    return pd.DataFrame(pearson_rows).sort_values("absolute_correlation", ascending=False), pd.DataFrame(spearman_rows).sort_values("absolute_correlation", ascending=False)


def cross_correlations(data: pd.DataFrame, phase1: dict[str, Any]) -> pd.DataFrame:
    rows = []
    lag_minutes = [-360, -180, -120, -60, -30, -15, 0, 15, 30, 60, 120, 180, 360]
    for sensor_id, group in data.groupby("sensor_id", sort=True):
        features = _provided_features(phase1, int(sensor_id))
        grids = {feature: _grid_series(group, feature) for feature in features}
        for source, target in itertools.permutations(features, 2):
            for minutes in lag_minutes:
                steps = minutes // 15
                # Positive lag means source(t) is paired with target(t + lag).
                pair = pd.concat([grids[source].rename("source"), grids[target].shift(-steps).rename("target")], axis=1).dropna()
                if len(pair) < 60 or pair.source.nunique() < 2 or pair.target.nunique() < 2:
                    continue
                corr = pair.source.corr(pair.target)
                rows.append({"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "source_feature": source, "target_feature": target, "lag_minutes": minutes, "lag": f"{minutes:+d} min" if abs(minutes) < 60 else f"{minutes/60:+g} h", "correlation": corr, "absolute_correlation": abs(corr), "n_observations": len(pair)})
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False)


def mutual_information(data: pd.DataFrame, phase1: dict[str, Any]) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("global", "All sensors", data)] + [("sensor", str(int(s)), g) for s, g in data.groupby("sensor_id")]
    for scope_type, scope_value, group in scopes:
        features = [f for f in ENVIRONMENTAL_FEATURES if group[f].notna().sum() >= 100]
        for source, target in itertools.permutations(features, 2):
            pair = group[[source, target]].dropna()
            if len(pair) < 100 or pair[source].nunique() < 5 or pair[target].nunique() < 5:
                continue
            sample = pair.iloc[::max(1, len(pair) // 2500)]
            mi = mutual_info_regression(sample[[source]], sample[target], random_state=42)[0]
            pearson = sample[source].corr(sample[target], method="pearson")
            spearman = sample[source].corr(sample[target], method="spearman")
            rows.append({"scope_type": scope_type, "scope_value": scope_value, "source_feature": source, "target_feature": target, "mutual_information": mi, "pearson": pearson, "spearman": spearman, "n_observations": len(pair)})
    return pd.DataFrame(rows).sort_values("mutual_information", ascending=False)


def nablus_comparison(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    nablus = data.loc[data.city_name.str.casefold() == "nablus"]
    rows, differences = [], []
    sensors = sorted(nablus.sensor_id.unique())
    for sensor_a, sensor_b in itertools.combinations(sensors, 2):
        left = nablus.loc[nablus.sensor_id == sensor_a].set_index("timestamp")
        right = nablus.loc[nablus.sensor_id == sensor_b].set_index("timestamp")
        for feature in ["pm25", "temperature", "humidity", "aqi"]:
            pair = pd.concat([left[feature].rename("a"), right[feature].rename("b")], axis=1, join="inner").dropna()
            if pair.empty:
                continue
            delta = pair.a - pair.b
            rows.append({
                "feature": feature, "sensor_a": int(sensor_a), "sensor_b": int(sensor_b),
                "sensor_pair": f"{int(sensor_a)} ↔ {int(sensor_b)}", "overlap_count": len(pair),
                "pearson": pair.a.corr(pair.b, method="pearson"), "spearman": pair.a.corr(pair.b, method="spearman"),
                "mae": delta.abs().mean(), "median_absolute_difference": delta.abs().median(),
                "mean_difference_a_minus_b": delta.mean(), "difference_std": delta.std(),
            })
            sampled = pair.assign(difference=delta).iloc[::max(1, len(pair) // 400)].reset_index()
            differences.append({"feature": feature, "sensor_a": int(sensor_a), "sensor_b": int(sensor_b), "sensor_pair": f"{int(sensor_a)} ↔ {int(sensor_b)}", "points": _records(sampled[["timestamp", "a", "b", "difference"]])})
    return pd.DataFrame(rows), differences


def pca_analysis(data: pd.DataFrame, phase1: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    variance_rows, loading_rows, results = [], [], []
    groups = [("Tulkarem", data.loc[data.city_name == "Tulkarem"]), ("Nablus", data.loc[data.city_name == "Nablus"])]
    for group_name, group in groups:
        features = [f for f in ENVIRONMENTAL_FEATURES if group[f].notna().mean() >= .95]
        valid = group[["timestamp", "sensor_id", *features]].dropna()
        if len(valid) < 100 or len(features) < 2:
            results.append({"group": group_name, "status": "SKIPPED", "reason": "Insufficient complete multivariate observations."})
            continue
        scaled = StandardScaler().fit_transform(valid[features])
        model = PCA(random_state=42).fit(scaled)
        scores = model.transform(scaled)
        cumulative = np.cumsum(model.explained_variance_ratio_)
        thresholds = {str(threshold): int(np.searchsorted(cumulative, threshold) + 1) for threshold in [.8, .9, .95]}
        for component_index, (ratio, cumulative_ratio) in enumerate(zip(model.explained_variance_ratio_, cumulative), start=1):
            variance_rows.append({"group": group_name, "component": f"PC{component_index}", "explained_variance_ratio": ratio, "cumulative_explained_variance": cumulative_ratio})
            for feature, loading in zip(features, model.components_[component_index - 1]):
                loading_rows.append({"group": group_name, "component": f"PC{component_index}", "feature": feature, "loading": loading, "absolute_loading": abs(loading)})
        sample_indices = np.arange(0, len(valid), max(1, len(valid) // 1200))
        points = pd.DataFrame({
            "timestamp": valid.timestamp.iloc[sample_indices].to_numpy(), "sensor_id": valid.sensor_id.iloc[sample_indices].to_numpy(),
            "pc1": scores[sample_indices, 0], "pc2": scores[sample_indices, 1] if scores.shape[1] > 1 else 0,
        })
        dominant = []
        for idx in range(min(3, len(features))):
            order = np.argsort(np.abs(model.components_[idx]))[::-1][:2]
            dominant.append({"component": f"PC{idx+1}", "features": [features[i] for i in order], "text": f"PC{idx+1} is statistically dominated by " + " and ".join(features[i] for i in order) + " variation."})
        results.append({"group": group_name, "status": "AVAILABLE", "features": features, "n_observations": len(valid), "components_for_threshold": thresholds, "dominant_components": dominant, "points": _records(points)})
    return pd.DataFrame(variance_rows), pd.DataFrame(loading_rows), results


def clustering(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    profile_rows, results, assignments_output = [], [], []
    for group_name, group in [("Tulkarem", data.loc[data.city_name == "Tulkarem"]), ("Nablus", data.loc[data.city_name == "Nablus"])]:
        features = [f for f in ENVIRONMENTAL_FEATURES if group[f].notna().mean() >= .95]
        valid = group[["timestamp", "sensor_id", "city_name", *features]].dropna().copy()
        if len(valid) < 200:
            results.append({"group": group_name, "status": "SKIPPED", "reason": "Insufficient complete observations."})
            continue
        scaled = StandardScaler().fit_transform(valid[features])
        evaluations = []
        best_model, best_score, best_k = None, -1.0, None
        for k in range(2, 7):
            model = KMeans(n_clusters=k, random_state=42, n_init=20).fit(scaled)
            score = silhouette_score(scaled, model.labels_, sample_size=min(2000, len(valid)), random_state=42)
            evaluations.append({"method": "KMeans", "parameter": f"k={k}", "cluster_count": k, "noise_count": 0, "inertia": model.inertia_, "silhouette": score})
            if score > best_score:
                best_model, best_score, best_k = model, score, k
        dbscan_evaluations = []
        for eps in [.4, .6, .8, 1.0, 1.2]:
            labels = DBSCAN(eps=eps, min_samples=12).fit_predict(scaled)
            cluster_labels = set(labels) - {-1}
            score = silhouette_score(scaled[labels != -1], labels[labels != -1]) if len(cluster_labels) >= 2 and (labels != -1).sum() >= 50 else None
            dbscan_evaluations.append({"method": "DBSCAN", "parameter": f"eps={eps}", "cluster_count": len(cluster_labels), "noise_count": int((labels == -1).sum()), "inertia": None, "silhouette": score})
        labels = best_model.labels_
        valid["cluster"] = labels
        overall_means = valid[features].mean()
        overall_stds = valid[features].std().replace(0, 1)
        cluster_details = []
        for cluster_id, cluster in valid.groupby("cluster"):
            standardized_profile = (cluster[features].mean() - overall_means) / overall_stds
            distinguishing = standardized_profile.abs().sort_values(ascending=False).index[:2].tolist()
            detail = {
                "cluster": int(cluster_id), "name": f"Cluster {int(cluster_id)}", "count": len(cluster),
                "percentage": 100 * len(cluster) / len(valid),
                "sensor_distribution": {str(int(k)): int(v) for k, v in cluster.sensor_id.value_counts().items()},
                "typical_hours": [int(v) for v in cluster.timestamp.dt.hour.value_counts().head(3).index],
                "distinguishing_features": distinguishing,
                "interpretation": f"Cluster {int(cluster_id)} is most distinguished by " + " and ".join(distinguishing) + " relative to this city group.",
            }
            cluster_details.append(detail)
            for feature in features:
                profile_rows.append({"group": group_name, "method": "KMeans", "cluster": int(cluster_id), "feature": feature, "count": len(cluster), "mean": cluster[feature].mean(), "median": cluster[feature].median(), "std": cluster[feature].std(), "min": cluster[feature].min(), "max": cluster[feature].max()})
        sample_indices = np.arange(0, len(valid), max(1, len(valid) // 1500))
        assignments = valid.iloc[sample_indices][["timestamp", "sensor_id", "cluster", *features]]
        assignments_output.append({"group": group_name, "features": features, "points": _records(assignments)})
        results.append({"group": group_name, "status": "AVAILABLE", "method": "KMeans", "selected_k": best_k, "silhouette_score": best_score, "evaluations": evaluations, "dbscan_sensitivity": dbscan_evaluations, "clusters": cluster_details})
    return pd.DataFrame(profile_rows), results, assignments_output


def anomaly_analysis(data: pd.DataFrame, phase1: dict[str, Any]) -> pd.DataFrame:
    rows = []
    multivariate_flags: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sensor_id, group in data.groupby("sensor_id", sort=True):
        features = _provided_features(phase1, int(sensor_id))
        valid = group[features].dropna()
        iso_flags = np.zeros(len(group), dtype=bool)
        lof_flags = np.zeros(len(group), dtype=bool)
        if len(valid) >= 100:
            scaled = StandardScaler().fit_transform(valid)
            iso = IsolationForest(contamination=.015, random_state=42, n_estimators=200).fit_predict(scaled) == -1
            neighbors = min(35, max(10, len(valid) // 40))
            lof = LocalOutlierFactor(n_neighbors=neighbors, contamination=.015).fit_predict(scaled) == -1
            positions = group.index.get_indexer(valid.index)
            iso_flags[positions] = iso
            lof_flags[positions] = lof
        multivariate_flags[int(sensor_id)] = (iso_flags, lof_flags)
        for feature in features:
            values = group[feature]
            finite = values.dropna()
            if len(finite) < 30:
                continue
            median = finite.median()
            mad = stats.median_abs_deviation(finite, scale="normal")
            robust_z = (values - median).abs() / mad if mad > 0 else pd.Series(0, index=values.index)
            q1, q3 = finite.quantile([.25, .75]); iqr = q3 - q1
            iqr_flag = (values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)
            mean, std = finite.mean(), finite.std()
            z_flag = ((values - mean).abs() / std > 4) if std > 0 else pd.Series(False, index=values.index)
            rolling_mean = values.rolling(96, min_periods=24).median()
            rolling_mad = (values - rolling_mean).abs().rolling(96, min_periods=24).median() * 1.4826
            rolling_z = (values - rolling_mean).abs() / rolling_mad.replace(0, np.nan)
            position_map = {idx: pos for pos, idx in enumerate(group.index)}
            candidates = z_flag.fillna(False) | (robust_z > 4.5).fillna(False) | iqr_flag.fillna(False) | (rolling_z > 5).fillna(False)
            candidates |= pd.Series(iso_flags | lof_flags, index=group.index)
            for idx in group.index[candidates]:
                pos = position_map[idx]
                flags = {
                    "zscore_flag": bool(z_flag.loc[idx]), "mad_flag": bool(robust_z.loc[idx] > 4.5),
                    "iqr_flag": bool(iqr_flag.loc[idx]), "rolling_zscore_flag": bool(rolling_z.loc[idx] > 5) if pd.notna(rolling_z.loc[idx]) else False,
                    "isolation_forest_flag": bool(iso_flags[pos]), "lof_flag": bool(lof_flags[pos]),
                }
                methods = [name.replace("_flag", "") for name, active in flags.items() if active]
                rows.append({
                    "row_index": int(idx), "timestamp": group.at[idx, "timestamp"], "sensor_id": int(sensor_id),
                    "city": group.at[idx, "city_name"], "feature": feature, "value": values.loc[idx],
                    **flags, "anomaly_methods_triggered": len(methods), "methods": ",".join(methods),
                    "quality_flag": group.at[idx, "overall_quality_flag"], "analytical_category": "UNCERTAIN",
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    timestamp_feature_counts = frame.groupby(["sensor_id", "timestamp"]).feature.transform("nunique")
    isolated = frame.groupby(["sensor_id", "feature"])["timestamp"].transform(lambda s: s.diff().dt.total_seconds().gt(1800) & s.shift(-1).sub(s).dt.total_seconds().gt(1800))
    frame.loc[timestamp_feature_counts >= 2, "analytical_category"] = "MULTIVARIATE_ANOMALY"
    frame.loc[(frame.anomaly_methods_triggered >= 2) & isolated.fillna(False), "analytical_category"] = "ISOLATED_SPIKE"
    # Agreement across Nablus sensors at the same feature/timestamp supports a shared event category.
    nablus = frame.loc[frame.city == "Nablus"]
    shared = nablus.groupby(["timestamp", "feature"]).sensor_id.nunique()
    shared_keys = set(shared[shared >= 2].index)
    shared_mask = [(row.timestamp, row.feature) in shared_keys for row in frame.itertuples()]
    frame.loc[shared_mask, "analytical_category"] = "LIKELY_ENVIRONMENTAL_EVENT"
    return frame.sort_values(["anomaly_methods_triggered", "timestamp"], ascending=[False, True])


def change_point_analysis(data: pd.DataFrame, phase1: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sensor_id, group in data.groupby("sensor_id", sort=True):
        for feature in _provided_features(phase1, int(sensor_id)):
            grid = _grid_series(group, feature)
            prepared = grid.interpolate(limit=2)
            segments = [seg.dropna() for _, seg in prepared.groupby(prepared.isna().cumsum()) if seg.notna().sum() >= 5 * 96]
            for segment in segments:
                values = segment.to_numpy(dtype=float)
                scale = np.std(values)
                if len(values) < 5 * 96 or scale <= 0:
                    continue
                normalized = ((values - np.mean(values)) / scale).reshape(-1, 1)
                breakpoints = rpt.Pelt(model="rbf", min_size=96, jump=8).fit(normalized).predict(pen=8 * np.log(len(values)))
                for point in breakpoints[:-1][:8]:
                    before = values[max(0, point - 96):point]
                    after = values[point:min(len(values), point + 96)]
                    if len(before) < 48 or len(after) < 48:
                        continue
                    mean_delta = float(np.mean(after) - np.mean(before))
                    variance_delta = float(np.var(after) - np.var(before))
                    if abs(mean_delta) / scale >= abs(variance_delta) / max(np.var(values), 1e-9):
                        change_type, before_stat, after_stat, magnitude = "MEAN_SHIFT", np.mean(before), np.mean(after), mean_delta
                    else:
                        change_type, before_stat, after_stat, magnitude = "VARIANCE_SHIFT", np.var(before), np.var(after), variance_delta
                    rows.append({"sensor_id": int(sensor_id), "city": group.city_name.iloc[0], "feature": feature, "change_timestamp": segment.index[point], "change_type": change_type, "before_statistic": before_stat, "after_statistic": after_stat, "magnitude": magnitude, "standardized_magnitude": abs(mean_delta) / scale, "label": "Behavioral Regime Shift"})
    return pd.DataFrame(rows).sort_values("standardized_magnitude", ascending=False) if rows else pd.DataFrame(columns=["sensor_id", "city", "feature", "change_timestamp", "change_type", "before_statistic", "after_statistic", "magnitude", "standardized_magnitude", "label"])


def _write_reports(result: dict[str, Any], frames: dict[str, pd.DataFrame], phase1: dict[str, Any]) -> None:
    stats_frame = frames["statistics"]
    autocorr = frames["autocorrelation"]
    pearson = frames["pearson"]
    spearman = frames["spearman"]
    cross = frames["cross_correlation"]
    mi = frames["mutual_information"]
    anomalies = frames["anomalies"]
    changes = frames["change_points"]
    sensor_compare = frames["sensor_comparison"]
    global_stats = stats_frame.loc[stats_frame.scope_type == "global"]
    top_corr = pearson.loc[pearson.scope_type == "sensor"].head(10)
    nonlinear = spearman.loc[(spearman.scope_type == "sensor") & (spearman.spearman_minus_pearson > .12)].head(8)
    strongest_cross = cross.loc[(cross.lag_minutes != 0) & ~cross.source_feature.eq("aqi") & ~cross.target_feature.eq("aqi")].drop_duplicates(["sensor_id", "source_feature", "target_feature"]).head(10)
    strongest_auto = autocorr.drop_duplicates(["sensor_id", "feature"]).head(12)
    pca_lines = []
    for item in result["pca"]["groups"]:
        if item.get("status") == "AVAILABLE":
            pca_lines.append(f"- {item['group']}: {item['components_for_threshold']['0.9']} components explain at least 90%; features used: {', '.join(item['features'])}.")
    cluster_lines = []
    for item in result["clusters"]["groups"]:
        if item.get("status") == "AVAILABLE":
            cluster_lines.append(f"- {item['group']}: K-Means selected k={item['selected_k']} with silhouette {item['silhouette_score']:.3f}; cluster labels remain neutral.")
    report05 = f"""# AirAware Phase 2 — Advanced EDA Report

Generated: {result['generated_at']}  
Source: Phase 1 cleaned dataset ({result['validation']['rows']:,} rows). No observations were changed or removed.

## Statistical findings

{chr(10).join(f"- {row.feature.upper()}: mean {row.mean:.2f}, median {row.median:.2f}, P95 {row.p95:.2f}, P99 {row.p99:.2f}, skewness {row.skewness:.2f}, kurtosis {row.kurtosis:.2f}." for row in global_stats.itertuples())}

## Temporal patterns and seasonality

{chr(10).join('- ' + item['text'] for item in result['temporal']['findings'][:12])}

{chr(10).join('- ' + item['text'] for item in result['decomposition']['findings'][:10]) or '- No decomposition passed the continuity rules.'}

## Strongest autocorrelation lags

{chr(10).join(f"- Sensor {row.sensor_id} {row.feature.upper()}, {row.lag}: r={row.autocorrelation:.3f}, n={row.n_observations}." for row in strongest_auto.itertuples())}

## Strongest within-sensor correlations

{chr(10).join(f"- Sensor {row.scope_value}: {row.feature_a.upper()} ↔ {row.feature_b.upper()}, Pearson r={row.correlation:.3f}, n={row.n_observations}." for row in top_corr.itertuples())}

## Monotonic/nonlinear signals

{chr(10).join(f"- Sensor {row.scope_value}: {row.feature_a.upper()} ↔ {row.feature_b.upper()}, Spearman={row.correlation:.3f}, Pearson={row.pearson_correlation:.3f}." for row in nonlinear.itertuples()) or '- No large Pearson/Spearman separation above the documented screen.'}

## Lagged cross-feature relationships

{chr(10).join(f"- Sensor {row.sensor_id}: {row.source_feature.upper()}(t) is associated with {row.target_feature.upper()}(t{row.lag}), r={row.correlation:.3f}, n={row.n_observations}." for row in strongest_cross.itertuples())}

## Mutual information

{chr(10).join(f"- {row.scope_type} {row.scope_value}: {row.source_feature.upper()} → {row.target_feature.upper()}, MI={row.mutual_information:.3f}, Pearson={row.pearson:.3f}, Spearman={row.spearman:.3f}." for row in mi.head(10).itertuples())}

## PCA

{chr(10).join(pca_lines)}

## Clustering

{chr(10).join(cluster_lines)}

## Anomaly findings

- Advanced anomaly event rows: {len(anomalies):,}; no point was removed.
- Events supported by at least two methods: {int((anomalies.anomaly_methods_triggered >= 2).sum()) if len(anomalies) else 0:,}.
- Categories: {anomalies.analytical_category.value_counts().to_dict() if len(anomalies) else {}}.

## Change-point findings

- Behavioral regime shifts detected: {len(changes):,}.
{chr(10).join(f"- Sensor {row.sensor_id} {row.feature.upper()} at {row.change_timestamp}: {row.change_type}, magnitude={row.magnitude:.3f}." for row in changes.head(12).itertuples())}

## Existing AQI dependency warning

**Existing AQI methodology not yet verified.** AQI relationships remain investigative and no formula, health threshold, or replacement AQI was inferred.
"""
    good_sensors = [str(row["sensor_id"]) for row in phase1["sampling"] if row["longest_gap_hours"] < 6]
    gap_sensors = [f"Sensor {row['sensor_id']} ({row['longest_gap_hours']:.2f} h longest gap)" for row in phase1["sampling"] if row["gap_count"]]
    top_cross_features = cross.loc[(cross.lag_minutes != 0) & ~cross.source_feature.eq("aqi") & ~cross.target_feature.eq("aqi")].head(12)
    report06 = f"""# AirAware Phase 2 — Modeling Readiness Report

This report prepares decisions only. No forecasting model or final ML feature engineering was performed.

## Which variables appear useful for forecasting?

PM2.5, temperature, humidity, and—only in Tulkarem—NO2/O3 have sufficient volume for temporal evaluation. The existing AQI must remain an investigative field because its methodology is unverified.

## Which lags appear useful?

{chr(10).join(f"- Sensor {row.sensor_id} {row.feature.upper()}: {row.lag} autocorrelation {row.autocorrelation:.3f}." for row in strongest_auto.head(10).itertuples())}

Cross-feature candidates:
{chr(10).join(f"- Sensor {row.sensor_id}: {row.source_feature.upper()}(t) ↔ {row.target_feature.upper()}(t{row.lag}), r={row.correlation:.3f}." for row in top_cross_features.itertuples())}

## Which rolling windows appear useful?

The 1h and 3h windows preserve short-term movement; 6h and 12h summarize medium persistence; 24h is appropriate for daily-baseline context. Selection should later be validated with leakage-safe time splits.

## Which pollutants have strong cross-feature relationships?

{chr(10).join(f"- Sensor {row.scope_value}: {row.feature_a.upper()} ↔ {row.feature_b.upper()}, Pearson={row.correlation:.3f}." for row in top_corr.head(8).itertuples())}

## Which features appear redundant?

PCA indicates potential redundancy only statistically:
{chr(10).join(pca_lines)}

## Which sensors contain enough continuous data for forecasting?

Sensors {', '.join(good_sensors)} have long coverage with no multi-hour outage. Sensor 5 requires segmented treatment because of its multi-day outage.

## Which sensors contain gaps requiring special treatment?

{chr(10).join('- ' + item for item in gap_sensors)}

## Which observations should remain flagged?

Retain Phase 1 quality flags and all Phase 2 anomaly consensus records. During future training, evaluate flagged observations with time-aware sensitivity analyses instead of deleting them automatically. Structural NO2/O3 absence in Nablus must remain unavailable, never zero-filled.
"""
    (REPORT_DIR / "05_advanced_eda_report.md").write_text(_clean_text(report05.strip()) + "\n", encoding="utf-8")
    (REPORT_DIR / "06_modeling_readiness_report.md").write_text(_clean_text(report06.strip()) + "\n", encoding="utf-8")


def run_phase2_pipeline() -> dict[str, Any]:
    ensure_directories()
    data, phase1 = _load()
    original_columns = ["sensor_id", "city_id", "city_name", "location_name", *ENVIRONMENTAL_FEATURES, "timestamp"]
    validation = {
        "cleaned_csv_exists": CLEANED_CSV.exists(), "rows": len(data),
        "quality_flags": [column for column in data.columns if column.endswith("_flag")],
        "sensor_ids": [int(value) for value in sorted(data.sensor_id.unique())],
        "timestamp_parse_failures": int(data.timestamp.isna().sum()),
        "structural_unavailability_preserved": all(
            row[feature] == "STRUCTURALLY_UNAVAILABLE" and data.loc[data.sensor_id == row["sensor_id"], feature].isna().all()
            for row in phase1["availability"] for feature in ["no2", "o3"]
            if row[feature] == "STRUCTURALLY_UNAVAILABLE"
        ),
        "observation_columns_unchanged": all(column in data.columns for column in original_columns),
    }
    stats_frame, stat_interpretations = descriptive_statistics(data, phase1)
    temporal_frame, temporal_findings = temporal_patterns(data, phase1)
    rolling = rolling_summary(data, phase1)
    decomposition_rows, decomposition_findings = decomposition(data, phase1)
    autocorr = autocorrelation_analysis(data, phase1)
    strongest_lags = autocorr.drop_duplicates(["sensor_id", "feature"]).copy()
    pearson, spearman = correlations(data, phase1)
    cross = cross_correlations(data, phase1)
    mi = mutual_information(data, phase1)
    sensor_compare, difference_series = nablus_comparison(data)
    pca_variance, pca_loadings, pca_groups = pca_analysis(data, phase1)
    cluster_profiles, cluster_groups, cluster_assignments = clustering(data)
    anomalies = anomaly_analysis(data, phase1)
    changes = change_point_analysis(data, phase1)
    if len(anomalies) and len(changes):
        change_lookup = changes.groupby(["sensor_id", "feature"])["change_timestamp"].apply(list).to_dict()
        persistent = []
        for row in anomalies.itertuples():
            near = any(abs((row.timestamp - timestamp).total_seconds()) <= 6 * 3600 for timestamp in change_lookup.get((row.sensor_id, row.feature), []))
            persistent.append(near and row.anomaly_methods_triggered >= 2)
        anomalies.loc[persistent, "analytical_category"] = "PERSISTENT_SHIFT"

    frames = {
        "statistics": stats_frame, "temporal": temporal_frame, "autocorrelation": autocorr,
        "strongest_lags": strongest_lags, "pearson": pearson, "spearman": spearman,
        "cross_correlation": cross, "mutual_information": mi, "pca_loadings": pca_loadings,
        "pca_variance": pca_variance, "cluster_profiles": cluster_profiles,
        "anomalies": anomalies, "change_points": changes, "sensor_comparison": sensor_compare,
    }
    for name, frame in frames.items():
        _write_csv(name, frame)

    strongest_positive = _records(pearson.sort_values("correlation", ascending=False).head(25))
    strongest_negative = _records(pearson.sort_values("correlation").head(25))
    result = {
        "phase": 2, "version": PHASE2_VERSION, "generated_at": datetime.now(timezone.utc),
        "validation": validation,
        "statistics": {"interpretations": stat_interpretations},
        "temporal": {"findings": temporal_findings}, "rolling_summary": rolling,
        "decomposition": {"series": decomposition_rows, "findings": decomposition_findings},
        "correlation": {"strongest_positive": strongest_positive, "strongest_negative": strongest_negative,
                        "nonlinear_candidates": _records(spearman.loc[spearman.spearman_minus_pearson > .1].head(30))},
        "cross_correlation": {"strongest": _records(cross.loc[cross.lag_minutes != 0].drop_duplicates(["sensor_id", "source_feature", "target_feature"]).head(50))},
        "mutual_information": {"strongest": _records(mi.head(50))},
        "pca": {"groups": pca_groups},
        "clusters": {"groups": cluster_groups, "assignments": cluster_assignments},
        "anomalies": {"total": len(anomalies), "consensus_count": int((anomalies.anomaly_methods_triggered >= 2).sum()) if len(anomalies) else 0, "category_counts": anomalies.analytical_category.value_counts().to_dict() if len(anomalies) else {}},
        "change_points": {"total": len(changes), "strongest": _records(changes.head(50))},
        "sensor_comparison": {"differences": difference_series},
        "aqi_warning": "Existing AQI methodology not yet verified.",
        "methodology": {
            "time_alignment": "All lags use a 15-minute UTC grid. Missing grid positions remain NaN, preventing gaps from creating false row-shift correlations.",
            "pca": "PCA is fitted separately by city on standardized complete feature sets; structural absence is excluded.",
            "clustering": "K-Means k=2..6 is evaluated by silhouette and inertia. DBSCAN parameter sensitivity is reported; invalid/missing rows are excluded only from fitting.",
            "anomalies": "Robust Z/MAD/IQR/rolling Z plus Isolation Forest and LOF produce retained flags; no observation is removed.",
            "change_points": "PELT with an RBF cost is applied only to sufficiently long continuous segments after interpolation of at most two short missing samples.",
        },
        "artifacts": [{"name": filename, "download_url": f"/api/analysis/download/{filename}"} for filename in ARTIFACTS.values()],
    }
    PHASE2_ANALYSIS_JSON.write_text(json.dumps(result, default=_json_value, allow_nan=False, indent=2), encoding="utf-8")
    parsed = json.loads(PHASE2_ANALYSIS_JSON.read_text(encoding="utf-8"))
    _write_reports(parsed, frames, phase1)
    return parsed


if __name__ == "__main__":
    output = run_phase2_pipeline()
    print(json.dumps({"phase": output["phase"], "validation": output["validation"], "artifacts": len(output["artifacts"]), "anomalies": output["anomalies"], "change_points": output["change_points"]["total"]}, indent=2))
