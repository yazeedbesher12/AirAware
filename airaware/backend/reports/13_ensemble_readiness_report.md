# 13 Ensemble Readiness Report

## Scope

This report prepares evidence for a later phase. No weighted ensemble, stacking meta-model, or combined prediction was trained.

## Strongest models by horizon

- **1h:** gru has the lowest test RMSE (3.437).
- **3h:** tft has the lowest test RMSE (4.340).
- **6h:** lstm has the lowest test RMSE (4.703).

## Diversity evidence

- lightgbm / xgboost at 3h: residual correlation 0.966 (overlap 1,104).
- lightgbm / xgboost at 1h: residual correlation 0.964 (overlap 1,136).
- lightgbm / xgboost at 6h: residual correlation 0.953 (overlap 1,056).
- lightgbm / random_forest at 1h: residual correlation 0.935 (overlap 1,136).
- gru / lstm at 3h: residual correlation 0.904 (overlap 956).

Lower-error-correlation pairs worth retaining for review:
- persistence / tft at 6h: residual correlation 0.184.
- gru / tft at 6h: residual correlation 0.205.
- prophet / tft at 6h: residual correlation 0.220.
- prophet / tft at 3h: residual correlation 0.248.
- persistence / prophet at 1h: residual correlation 0.299.
- persistence / prophet at 3h: residual correlation 0.349.

## Candidate set for later review

GRU, LSTM, TFT, and LightGBM provide a useful balance of accuracy, horizon coverage, and model-family diversity. XGBoost is a credible LightGBM substitute, but the two are highly redundant. Prophet is diverse but too inaccurate to include solely for diversity.

## Stacking readiness

The time-safe OOF archive contains 4,086 unique horizon/timestamp/sensor rows; the chronological test archive contains 3,296. Missing sequence-model predictions during warm-up remain explicit. These archives were not used to fit a meta-model.

## Limitations

The dataset spans only 17 days. Candidate recommendations are provisional and require review on more unseen temporal coverage.