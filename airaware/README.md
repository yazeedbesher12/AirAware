# AirAware - Local Phase 1, Phase 2, and Phase 3 Dashboard

AirAware audits and conservatively cleans the Tulkarem and Nablus sensor dataset, provides advanced exploratory analysis, and prepares leakage-safe ML features, WHO 2021 health-reference windows, and future forecasting targets through a local FastAPI service and React/Vite dashboard. It does not recalculate source AQI, train forecasting models, or call external runtime services.

## Architecture

```text
backend/data/raw/Tulkarem+Nablus.csv
  -> Phase 1 Pandas / NumPy / SciPy quality pipeline
  -> cleaned CSV + quality flags + audit reports
  -> Phase 2 advanced EDA pipeline and cached CSV/JSON outputs
  -> Phase 3 ML features + WHO reference analysis + forecast targets
  -> FastAPI on 127.0.0.1:8000
  -> React/Vite on 127.0.0.1:5173
```

The original workspace CSV remains untouched. Phase 1 verifies its SHA-256 checksum, and Phase 2 uses the cleaned CSV as its observation source without changing or deleting observations.

## Install and rerun analysis

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
python run_phase2_pipeline.py
python run_phase3_pipeline.py
```

Phase 2 precomputes descriptive statistics, temporal profiles, rolling stability, STL decomposition, elapsed-time ACF/PACF, Pearson/Spearman and lagged cross-correlations, mutual information, city-specific PCA, K-Means/DBSCAN evaluation, multi-method anomaly consensus, and PELT change points. Expensive results are cached in `backend/outputs/phase2_analysis.json`.

Phase 3 uses Phase 2 evidence to build exact-time lag features, backward-only rolling features, rate-of-change and quality context, continuous segment IDs, future PM2.5 targets, WHO 2021 averaging-period analysis, and forecast-sample validity. Feature and target files remain separate. WHO values are centralized in `backend/config/who_2021_guidelines.json`; NO2/O3 comparisons remain disabled because their source units are not documented.

Run deterministic leakage, gap, rolling, target, unit, and structural-availability tests with:

```powershell
cd backend
.venv\Scripts\python.exe -m pytest -q
```

## Start locally

Backend terminal:

```powershell
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload
```

Frontend terminal:

```powershell
cd frontend
npm install
npm run dev
```

Or, after dependencies are installed:

```powershell
.\run_local.ps1
```

- Dashboard: <http://localhost:5173>
- API: <http://127.0.0.1:8000>
- API documentation: <http://127.0.0.1:8000/docs>

## Phase 1 methodology

- Structural measurement absence is determined per sensor and never interpolated.
- Expected cadence is the sensor-specific modal interval (15 minutes here).
- A gap is an interval greater than 1.5 times expected cadence; severity is documented in the generated report.
- Sudden changes use sensor-feature robust MAD and IQR criteria.
- Long identical and near-zero-variance runs are possible stuck patterns, never confirmed hardware failures.
- Baseline changes remain exploratory.
- The cleaned CSV preserves every observation and adds compact quality flags.

## Phase 2 machine-readable outputs

- `descriptive_statistics.csv`
- `temporal_patterns.csv`
- `autocorrelation_results.csv`
- `strongest_lags.csv`
- `pearson_correlations.csv`
- `spearman_correlations.csv`
- `cross_correlations.csv`
- `mutual_information.csv`
- `pca_loadings.csv`
- `pca_variance.csv`
- `cluster_profiles.csv`
- `anomaly_results.csv`
- `change_points.csv`
- `nablus_sensor_comparison.csv`

They are stored in `backend/outputs/` and exposed through `/api/analysis/download/{filename}`.

## Reports

- `backend/reports/01_data_audit_report.md`
- `backend/reports/02_cleaning_report.md`
- `backend/reports/03_sensor_quality_report.md`
- `backend/reports/04_phase1_findings.md`
- `backend/reports/05_advanced_eda_report.md`
- `backend/reports/06_modeling_readiness_report.md`
- `backend/reports/07_feature_leakage_report.md`
- `backend/reports/08_who_integration_report.md`
- `backend/reports/09_feature_engineering_report.md`
- `backend/reports/10_forecasting_dataset_readiness.md`

## Phase 3 outputs

- `AirAware_ML_Features.csv`
- `AirAware_Forecast_Targets.csv`
- `feature_catalog.csv`
- `candidate_feature_ranking.csv`
- `feature_redundancy.csv`
- `who_guideline_config.csv`
- `who_analysis_results.csv`
- `forecast_sample_validity.csv`
- `forecast_dataset_summary.csv`

These are stored in `backend/outputs/` and exposed through the `/api/ml/*` and `/api/who/*` routes. The WHO reference configuration cites the official [WHO 2021 global air quality guidelines](https://www.who.int/publications/i/item/9789240034228).

## Phase 4A individual model benchmarking

Train one saved model family without touching the test split during selection:

```powershell
cd backend
.venv\Scripts\python.exe train_models.py --model xgboost
```

Optional filters are `--horizon 1h|3h|6h`, `--sensor SENSOR_ID`, and `--target pm25`. Run every independent benchmark with:

```powershell
.venv\Scripts\python.exe train_models.py --model all
```

Run the no-retraining, reading-by-reading deployment replay after training:

```powershell
.venv\Scripts\python.exe run_realtime_replay.py
```

Models are stored under `backend/models/<model>/`; test and time-safe expanding-origin validation predictions are stored under `backend/outputs/predictions/<model>/`. Central evidence files include `model_results.json`, `model_results_detailed.json`, `model_leaderboard.json`, `training_run.json`, `stacking_oof_predictions.csv`, `test_predictions_all_models.csv`, `realtime_replay_results.json`, and `realtime_replay_predictions.csv`.

Phase 4A reports are `11_model_training_report.md`, `12_model_evaluation_report.md`, `13_ensemble_readiness_report.md`, and `14_realtime_model_readiness_report.md`. No weighted ensemble, stacking meta-model, or combined forecast is trained in this phase.
