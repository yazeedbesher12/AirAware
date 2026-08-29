from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
RAW_DIR = BACKEND_DIR / "data" / "raw"
CLEANED_DIR = BACKEND_DIR / "data" / "cleaned"
OUTPUT_DIR = BACKEND_DIR / "outputs"
REPORT_DIR = BACKEND_DIR / "reports"
CONFIG_DIR = BACKEND_DIR / "config"

RAW_CSV = RAW_DIR / "Tulkarem+Nablus.csv"
CLEANED_CSV = CLEANED_DIR / "AirAware_Tulkarem_Nablus_cleaned.csv"
ANALYSIS_JSON = OUTPUT_DIR / "analysis.json"
PHASE2_ANALYSIS_JSON = OUTPUT_DIR / "phase2_analysis.json"
PHASE3_ANALYSIS_JSON = OUTPUT_DIR / "phase3_analysis.json"
WHO_2021_CONFIG = CONFIG_DIR / "who_2021_guidelines.json"
ML_FEATURES_CSV = OUTPUT_DIR / "AirAware_ML_Features.csv"
FORECAST_TARGETS_CSV = OUTPUT_DIR / "AirAware_Forecast_Targets.csv"
SUSPICIOUS_CSV = OUTPUT_DIR / "suspicious_observations.csv"
GAPS_CSV = OUTPUT_DIR / "sensor_gaps.csv"
MISSING_PERIODS_CSV = OUTPUT_DIR / "missing_periods.csv"
CHANGE_LOG_CSV = OUTPUT_DIR / "cleaning_change_log.csv"

PHASE2_ARTIFACTS = {
    "statistics": "descriptive_statistics.csv",
    "temporal": "temporal_patterns.csv",
    "autocorrelation": "autocorrelation_results.csv",
    "strongest_lags": "strongest_lags.csv",
    "pearson": "pearson_correlations.csv",
    "spearman": "spearman_correlations.csv",
    "cross_correlation": "cross_correlations.csv",
    "mutual_information": "mutual_information.csv",
    "pca_loadings": "pca_loadings.csv",
    "pca_variance": "pca_variance.csv",
    "cluster_profiles": "cluster_profiles.csv",
    "anomalies": "anomaly_results.csv",
    "change_points": "change_points.csv",
    "sensor_comparison": "nablus_sensor_comparison.csv",
}

PHASE3_ARTIFACTS = {
    "ml_features": "AirAware_ML_Features.csv",
    "forecast_targets": "AirAware_Forecast_Targets.csv",
    "feature_catalog": "feature_catalog.csv",
    "feature_ranking": "candidate_feature_ranking.csv",
    "feature_redundancy": "feature_redundancy.csv",
    "who_guidelines": "who_guideline_config.csv",
    "who_analysis": "who_analysis_results.csv",
    "sample_validity": "forecast_sample_validity.csv",
    "forecast_summary": "forecast_dataset_summary.csv",
}

ENVIRONMENTAL_FEATURES = ["pm25", "temperature", "humidity", "no2", "o3", "aqi"]
FEATURE_UNITS = {
    "pm25": "ug/m3",
    "temperature": "deg C",
    "humidity": "%",
    "no2": "source unit (unverified)",
    "o3": "source unit (unverified)",
    "aqi": "index",
}


def ensure_directories() -> None:
    for directory in (RAW_DIR, CLEANED_DIR, OUTPUT_DIR, REPORT_DIR, CONFIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
