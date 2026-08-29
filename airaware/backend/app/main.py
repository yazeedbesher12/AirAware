from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import (
    CLEANED_CSV, ENVIRONMENTAL_FEATURES, GAPS_CSV, OUTPUT_DIR,
    PHASE2_ARTIFACTS, PHASE3_ARTIFACTS, REPORT_DIR, SUSPICIOUS_CSV,
)
from app.services.data_store import DataStore, get_store, records
from app.services.phase2_store import Phase2Store, get_phase2_store
from app.services.phase3_store import Phase3Store, get_phase3_store

app = FastAPI(
    title="AirAware Local Analysis API",
    description="Local Phase 1–3 data quality, advanced analysis, WHO-reference, and forecasting-dataset API.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"name": "AirAware Local Analysis API", "status": "ready", "docs": "/docs"}


@app.get("/health", tags=["system"])
def health(store: DataStore = Depends(get_store)) -> dict[str, object]:
    return {"status": "ok", "rows": len(store.data), "pipeline_version": store.analysis["pipeline_version"]}


@app.get("/api/overview")
def overview(store: DataStore = Depends(get_store)) -> dict[str, object]:
    return {**store.analysis["overview"], "methodology": store.analysis["methodology"], "duplicates": store.analysis["duplicates"]}


@app.get("/api/sensors")
def sensors(store: DataStore = Depends(get_store)) -> list[dict[str, object]]:
    return store.analysis["sensor_quality"]


@app.get("/api/availability")
def availability(city: str | None = None, sensor_id: int | None = None, store: DataStore = Depends(get_store)) -> list[dict[str, object]]:
    rows = store.analysis["availability"]
    return [row for row in rows if (not city or row["city"].casefold() == city.casefold()) and (sensor_id is None or row["sensor_id"] == sensor_id)]


@app.get("/api/missing")
def missing(city: str | None = None, sensor_id: int | None = None, feature: str | None = None, store: DataStore = Depends(get_store)) -> dict[str, object]:
    payload = store.analysis["missing"]
    per_sensor = [row for row in payload["per_sensor"] if (not city or row["city"].casefold() == city.casefold()) and (sensor_id is None or row["sensor_id"] == sensor_id) and (not feature or row["feature"] == feature)]
    periods = [row for row in store.analysis["missing_periods"] if (not city or row["city"].casefold() == city.casefold()) and (sensor_id is None or row["sensor_id"] == sensor_id) and (not feature or row["feature"] == feature)]
    return {"per_column": payload["per_column"], "per_city": payload["per_city"], "per_sensor": per_sensor, "periods": periods}


@app.get("/api/gaps")
def gaps(city: str | None = None, sensor_id: int | None = None, severity: str | None = None, store: DataStore = Depends(get_store)) -> dict[str, object]:
    items = [row for row in store.analysis["gaps"] if (not city or row["city"].casefold() == city.casefold()) and (sensor_id is None or row["sensor_id"] == sensor_id) and (not severity or row["severity"].casefold() == severity.casefold())]
    return {"items": items, "sampling": store.analysis["sampling"], "methodology": store.analysis["methodology"]["gap_classification"]}


@app.get("/api/timeseries")
def timeseries(
    feature: str = "pm25", city: str | None = None, sensor_id: int | None = None,
    start_time: str | None = None, end_time: str | None = None, rolling_window: str | None = "none",
    max_points: int = Query(1500, ge=100, le=5000), store: DataStore = Depends(get_store),
) -> dict[str, object]:
    try:
        return store.timeseries(feature, city, sensor_id, start_time, end_time, rolling_window, max_points)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/distribution")
def distribution(feature: str = "pm25", city: str | None = None, sensor_id: int | None = None, store: DataStore = Depends(get_store)) -> dict[str, object]:
    return store.distribution(feature, city, sensor_id)


@app.get("/api/scatter")
def scatter(x_feature: str = "pm25", y_feature: str = "aqi", city: str | None = None, sensor_id: int | None = None, store: DataStore = Depends(get_store)) -> dict[str, object]:
    try:
        return store.scatter(x_feature, y_feature, city, sensor_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sensor-quality")
def sensor_quality(city: str | None = None, sensor_id: int | None = None, store: DataStore = Depends(get_store)) -> list[dict[str, object]]:
    return [row for row in store.analysis["sensor_quality"] if (not city or row["city"].casefold() == city.casefold()) and (sensor_id is None or row["sensor_id"] == sensor_id)]


@app.get("/api/nablus-comparison")
def nablus_comparison(feature: str = "pm25", store: DataStore = Depends(get_store)) -> dict[str, object]:
    if feature not in {"pm25", "temperature", "humidity", "aqi"}:
        raise HTTPException(status_code=400, detail="Comparison feature must be pm25, temperature, humidity, or aqi.")
    return {"feature": feature, "pairs": [row for row in store.analysis["nablus_comparison"] if row["feature"] == feature]}


@app.get("/api/aqi-analysis")
def aqi_analysis(store: DataStore = Depends(get_store)) -> dict[str, object]:
    return store.analysis["aqi_analysis"]


@app.get("/api/suspicious")
def suspicious(
    city: str | None = None, sensor_id: int | None = None, feature: str | None = None,
    flag: str | None = None, start_time: str | None = None, end_time: str | None = None,
    limit: int = Query(200, ge=1, le=2000), offset: int = Query(0, ge=0), store: DataStore = Depends(get_store),
) -> dict[str, object]:
    return store.suspicious_rows(city, sensor_id, feature, flag, start_time, end_time, limit, offset)


@app.get("/api/analysis-events")
def analysis_events(store: DataStore = Depends(get_store)) -> dict[str, object]:
    return {key: store.analysis[key] for key in ("jumps", "flatlines", "drifts", "rolling_summary", "numeric_audit")}


@app.get("/api/reports")
def reports() -> list[dict[str, str]]:
    return [{"name": path.name, "download_url": f"/api/reports/{path.name}"} for path in sorted(REPORT_DIR.glob("*.md"))]


@app.get("/api/reports/{filename}")
def report_download(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = REPORT_DIR / safe_name
    if safe_name != filename or not path.exists() or path.suffix != ".md":
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, media_type="text/markdown", filename=safe_name)


@app.get("/api/download/cleaned")
def cleaned_download() -> FileResponse:
    return FileResponse(CLEANED_CSV, media_type="text/csv", filename=CLEANED_CSV.name)


@app.get("/api/download/suspicious")
def suspicious_download() -> FileResponse:
    return FileResponse(SUSPICIOUS_CSV, media_type="text/csv", filename=SUSPICIOUS_CSV.name)


@app.get("/api/download/gaps")
def gaps_download() -> FileResponse:
    return FileResponse(GAPS_CSV, media_type="text/csv", filename=GAPS_CSV.name)


@app.get("/api/analysis/summary")
def phase2_summary(store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    analysis = store.analysis
    return {
        "phase": analysis["phase"],
        "version": analysis["version"],
        "generated_at": analysis["generated_at"],
        "validation": analysis["validation"],
        "statistics": analysis["statistics"],
        "temporal": analysis["temporal"],
        "correlation": analysis["correlation"],
        "cross_correlation": analysis["cross_correlation"],
        "mutual_information": analysis["mutual_information"],
        "anomalies": analysis["anomalies"],
        "change_points": analysis["change_points"],
        "aqi_warning": analysis["aqi_warning"],
        "artifacts": analysis["artifacts"],
    }


@app.get("/api/analysis/statistics")
def phase2_statistics(city: str | None = None, sensor_id: int | None = None, feature: str | None = None, store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    return store.statistics(city, sensor_id, feature)


@app.get("/api/analysis/distributions")
def phase2_distributions(city: str | None = None, sensor_id: int | None = None, feature: str = "pm25", store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    try:
        return store.distributions(city, sensor_id, feature)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/analysis/temporal")
def phase2_temporal(city: str | None = None, sensor_id: int | None = None, feature: str | None = None, pattern_type: str | None = None, store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    return store.temporal(city, sensor_id, feature, pattern_type)


@app.get("/api/analysis/rolling")
def phase2_rolling(city: str | None = None, sensor_id: int | None = None, feature: str = "pm25", window: str = "3h", start_time: str | None = None, end_time: str | None = None, store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    try:
        return store.rolling(city, sensor_id, feature, window, start_time, end_time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/analysis/decomposition")
def phase2_decomposition(city: str | None = None, sensor_id: int | None = None, feature: str | None = None, store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    items = [row for row in store.analysis["decomposition"]["series"] if (not city or row.get("city", "").casefold() == city.casefold()) and (sensor_id is None or row["sensor_id"] == sensor_id) and (not feature or row["feature"] == feature)]
    findings = [row for row in store.analysis["decomposition"]["findings"] if (sensor_id is None or row["sensor_id"] == sensor_id) and (not feature or row["feature"] == feature)]
    return {"items": items, "findings": findings}


@app.get("/api/analysis/autocorrelation")
def phase2_autocorrelation(city: str | None = None, sensor_id: int | None = None, feature: str | None = None, store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    return {"items": store.filtered_frame("autocorrelation", city, sensor_id, feature, 500), "methodology": store.analysis["methodology"]["time_alignment"]}


@app.get("/api/analysis/correlation")
def phase2_correlation(method: str = "pearson", city: str | None = None, sensor_id: int | None = None, feature: str | None = None, store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    return store.correlations(method, city, sensor_id, feature)


@app.get("/api/analysis/cross-correlation")
def phase2_cross_correlation(city: str | None = None, sensor_id: int | None = None, feature: str | None = None, limit: int = Query(250, ge=1, le=2000), store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    frame = store.frames["cross_correlation"]
    if city:
        frame = frame.loc[frame.city.str.casefold() == city.casefold()]
    if sensor_id is not None:
        frame = frame.loc[frame.sensor_id == sensor_id]
    if feature:
        frame = frame.loc[(frame.source_feature == feature) | (frame.target_feature == feature)]
    return {"items": records(frame.head(limit)), "methodology": store.analysis["methodology"]["time_alignment"]}


@app.get("/api/analysis/mutual-information")
def phase2_mutual_information(sensor_id: int | None = None, feature: str | None = None, limit: int = Query(250, ge=1, le=1000), store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    frame = store.frames["mutual_information"]
    if sensor_id is not None:
        frame = frame.loc[(frame.scope_type == "sensor") & (frame.scope_value.astype(str) == str(sensor_id))]
    if feature:
        frame = frame.loc[(frame.source_feature == feature) | (frame.target_feature == feature)]
    return {"items": records(frame.head(limit))}


@app.get("/api/analysis/pca")
def phase2_pca(city: str | None = None, store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    groups = [row for row in store.analysis["pca"]["groups"] if not city or row["group"].casefold() == city.casefold()]
    loadings = store.frames["pca_loadings"]
    variance = store.frames["pca_variance"]
    if city:
        loadings = loadings.loc[loadings.group.str.casefold() == city.casefold()]
        variance = variance.loc[variance.group.str.casefold() == city.casefold()]
    return {"groups": groups, "loadings": records(loadings), "variance": records(variance), "methodology": store.analysis["methodology"]["pca"]}


@app.get("/api/analysis/clusters")
def phase2_clusters(city: str | None = None, store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    groups = [row for row in store.analysis["clusters"]["groups"] if not city or row["group"].casefold() == city.casefold()]
    assignments = [row for row in store.analysis["clusters"]["assignments"] if not city or row["group"].casefold() == city.casefold()]
    profiles = store.frames["cluster_profiles"]
    if city:
        profiles = profiles.loc[profiles.group.str.casefold() == city.casefold()]
    return {"groups": groups, "assignments": assignments, "profiles": records(profiles), "methodology": store.analysis["methodology"]["clustering"]}


@app.get("/api/analysis/anomalies")
def phase2_anomalies(city: str | None = None, sensor_id: int | None = None, feature: str | None = None, method: str | None = None, start_time: str | None = None, end_time: str | None = None, limit: int = Query(500, ge=1, le=3000), store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    frame = store.frames["anomalies"]
    if city:
        frame = frame.loc[frame.city.str.casefold() == city.casefold()]
    if sensor_id is not None:
        frame = frame.loc[frame.sensor_id == sensor_id]
    if feature:
        frame = frame.loc[frame.feature == feature]
    if method and f"{method}_flag" in frame:
        frame = frame.loc[frame[f"{method}_flag"].astype(str).str.casefold() == "true"]
    if start_time:
        frame = frame.loc[frame.timestamp >= pd.to_datetime(start_time, utc=True)]
    if end_time:
        frame = frame.loc[frame.timestamp <= pd.to_datetime(end_time, utc=True)]
    return {"total": len(frame), "items": records(frame.head(limit)), "summary": store.analysis["anomalies"], "methodology": store.analysis["methodology"]["anomalies"]}


@app.get("/api/analysis/change-points")
def phase2_change_points(city: str | None = None, sensor_id: int | None = None, feature: str | None = None, store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    return {"items": store.filtered_frame("change_points", city, sensor_id, feature, 500), "summary": store.analysis["change_points"], "methodology": store.analysis["methodology"]["change_points"]}


@app.get("/api/analysis/sensor-comparison")
def phase2_sensor_comparison(feature: str = "pm25", store: Phase2Store = Depends(get_phase2_store)) -> dict[str, object]:
    table = store.frames["sensor_comparison"]
    table = table.loc[table.feature == feature]
    differences = [row for row in store.analysis["sensor_comparison"]["differences"] if row["feature"] == feature]
    return {"feature": feature, "pairs": records(table), "difference_series": differences}


@app.get("/api/analysis/download/{filename}")
def phase2_download(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    allowed = set(PHASE2_ARTIFACTS.values())
    path = OUTPUT_DIR / safe_name
    if safe_name != filename or safe_name not in allowed or not path.exists():
        raise HTTPException(status_code=404, detail="Phase 2 artifact not found.")
    return FileResponse(path, media_type="text/csv", filename=safe_name)


@app.get("/api/ml/summary")
def phase3_summary(store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    return store.analysis


@app.get("/api/ml/features")
def ml_features(family: str | None = None, store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    return store.feature_catalog(family)


@app.get("/api/ml/feature-ranking")
def ml_feature_ranking(target: str | None = None, association_method: str | None = None, limit: int = Query(100, ge=1, le=1000), store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    return store.ranking(target, association_method, limit)


@app.get("/api/ml/redundancy")
def ml_redundancy(limit: int = Query(200, ge=1, le=1000), store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    frame = store.frames["feature_redundancy"].head(limit)
    return {"total": len(store.frames["feature_redundancy"]), "items": records(frame)}


@app.get("/api/ml/sample-validity")
def ml_sample_validity(city: str | None = None, sensor_id: int | None = None, horizon: str | None = None, store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    return store.sample_validity(city, sensor_id, horizon)


@app.get("/api/ml/targets")
def ml_targets(city: str | None = None, sensor_id: int | None = None, horizon: str = "1h", start_time: str | None = None, end_time: str | None = None, max_points: int = Query(1500, ge=100, le=5000), store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    try:
        return store.targets(city, sensor_id, horizon, start_time, end_time, max_points)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/who/status")
def who_status(city: str | None = None, sensor_id: int | None = None, store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    return store.who_status(city, sensor_id)


@app.get("/api/who/timeseries")
def who_timeseries(pollutant: str = "pm25", city: str | None = None, sensor_id: int | None = None, start_time: str | None = None, end_time: str | None = None, max_points: int = Query(2000, ge=100, le=5000), store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    frame = store.who_frame(pollutant, city, sensor_id, start_time, end_time)
    config = store.frames["who_guidelines"]
    config = config.loc[config.pollutant == pollutant]
    if frame.empty or frame.status.eq("Not Available").all():
        return {"pollutant": pollutant, "available": False, "config": records(config), "items": [], "message": f"{pollutant.upper()} is not available for the selected sensors.", "warning": store.analysis["who"]["warning"]}
    per_sensor = max(25, max_points // max(1, frame.sensor_id.nunique()))
    sampled = []
    for _, group in frame.groupby("sensor_id", sort=True):
        stride = max(1, len(group) // per_sensor)
        sampled.append(group.iloc[::stride].head(per_sensor))
    output = pd.concat(sampled).sort_values(["sensor_id", "timestamp"]) if sampled else frame.head(0)
    return {"pollutant": pollutant, "available": True, "config": records(config), "items": records(output), "warning": store.analysis["who"]["warning"]}


@app.get("/api/who/coverage")
def who_coverage(pollutant: str = "pm25", city: str | None = None, sensor_id: int | None = None, store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    frame = store.who_frame(pollutant, city, sensor_id)
    grouped = frame.groupby(["sensor_id", "city"], as_index=False).agg(total_windows=("timestamp", "size"), average_coverage_percent=("coverage_percent", "mean"), valid_windows=("status", lambda values: int(values.isin(["Meets WHO Guideline", "Above WHO Guideline", "Unit Not Compatible"]).sum())))
    return {"pollutant": pollutant, "items": records(grouped)}


@app.get("/api/who/exceedances")
def who_exceedances(pollutant: str = "pm25", city: str | None = None, sensor_id: int | None = None, start_time: str | None = None, end_time: str | None = None, limit: int = Query(500, ge=1, le=3000), store: Phase3Store = Depends(get_phase3_store)) -> dict[str, object]:
    frame = store.who_frame(pollutant, city, sensor_id, start_time, end_time)
    compatible = frame.loc[frame.unit_compatible.astype(str).str.casefold() == "true"]
    valid = compatible.loc[compatible.status.isin(["Meets WHO Guideline", "Above WHO Guideline"])]
    exceeded = valid.loc[valid.status == "Above WHO Guideline"]
    return {"pollutant": pollutant, "valid_windows": len(valid), "exceedance_windows": len(exceeded), "exceedance_percentage": (len(exceeded) / len(valid) * 100 if len(valid) else 0), "items": records(exceeded.head(limit)), "interpretation": "Overlapping health-reference windows; not a count of independent WHO compliance days."}


@app.get("/api/ml/download/{filename}")
def phase3_download(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    allowed = set(PHASE3_ARTIFACTS.values())
    path = OUTPUT_DIR / safe_name
    if safe_name != filename or safe_name not in allowed or not path.exists():
        raise HTTPException(status_code=404, detail="Phase 3 artifact not found.")
    return FileResponse(path, media_type="text/csv", filename=safe_name)
