# 11 Model Training Report

Generated: 2026-08-29T11:04:46.821710+00:00

## Data and validation design

Phase 3 ML features and valid PM2.5 targets were used without rebuilding feature engineering. Rows are ordered chronologically: train is the oldest period, validation is later, and test is the newest unseen period. Two expanding-origin folds generated time-safe OOF predictions; no shuffle or random split was used.

Imputers and neural scalers were fitted on training data only. LSTM, GRU, and TFT sequences remain inside continuous segments. The source AQI was not used as a target. No ensemble or stacking model was trained.

## Training runs

| Model | Horizon | Train | Validation | Test | OOF predictions | Duration (s) | Best epoch | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| persistence | 1h | 2947 | 1362 | 1136 | 1362 | 0.22 | N/A | completed |
| persistence | 3h | 2939 | 1362 | 1104 | 1362 | 0.08 | N/A | completed |
| persistence | 6h | 2932 | 1362 | 1056 | 1362 | 0.08 | N/A | completed |
| random_forest | 1h | 2947 | 1362 | 1136 | 1362 | 8.07 | N/A | completed |
| random_forest | 3h | 2939 | 1362 | 1104 | 1362 | 7.64 | N/A | completed |
| random_forest | 6h | 2932 | 1362 | 1056 | 1362 | 7.55 | N/A | completed |
| xgboost | 1h | 2947 | 1362 | 1136 | 1362 | 6.05 | N/A | completed |
| xgboost | 3h | 2939 | 1362 | 1104 | 1362 | 5.76 | N/A | completed |
| xgboost | 6h | 2932 | 1362 | 1056 | 1362 | 4.76 | N/A | completed |
| lightgbm | 1h | 2947 | 1362 | 1136 | 1362 | 2.69 | N/A | completed |
| lightgbm | 3h | 2939 | 1362 | 1104 | 1362 | 2.75 | N/A | completed |
| lightgbm | 6h | 2932 | 1362 | 1056 | 1362 | 2.87 | N/A | completed |
| prophet | 1h | 2947 | 1362 | 1136 | 1362 | 8.32 | N/A | completed |
| prophet | 3h | 2939 | 1362 | 1104 | 1362 | 7.91 | N/A | completed |
| prophet | 6h | 2932 | 1362 | 1056 | 1362 | 10.06 | N/A | completed |
| lstm | 1h | 2947 | 1362 | 1136 | 1255 | 9.03 | 11 | completed |
| lstm | 3h | 2939 | 1362 | 1104 | 1223 | 8.01 | 7 | completed |
| lstm | 6h | 2932 | 1362 | 1056 | 1223 | 8.01 | 7 | completed |
| gru | 1h | 2947 | 1362 | 1136 | 1255 | 11.99 | 8 | completed |
| gru | 3h | 2939 | 1362 | 1104 | 1223 | 10.12 | 2 | completed |
| gru | 6h | 2932 | 1362 | 1056 | 1223 | 9.88 | 2 | completed |
| tft | 1h | 2947 | 1362 | 1136 | 1350 | 185.89 | 6 | completed |
| tft | 3h | 2939 | 1362 | 1104 | 1350 | 186.85 | 6 | completed |
| tft | 6h | 2932 | 1362 | 1056 | 1350 | 184.26 | 6 | completed |

## Saved hyperparameters

### persistence — 1h

```json
{
  "rule": "prediction(t+h)=latest valid value at t"
}
```

### persistence — 3h

```json
{
  "rule": "prediction(t+h)=latest valid value at t"
}
```

### persistence — 6h

```json
{
  "rule": "prediction(t+h)=latest valid value at t"
}
```

### random_forest — 1h

```json
{
  "n_estimators": 180,
  "max_depth": 14,
  "min_samples_split": 4,
  "min_samples_leaf": 2,
  "max_features": 0.75,
  "n_jobs": 4,
  "random_state": 42
}
```

### random_forest — 3h

```json
{
  "n_estimators": 180,
  "max_depth": 14,
  "min_samples_split": 4,
  "min_samples_leaf": 2,
  "max_features": 0.75,
  "n_jobs": 4,
  "random_state": 42
}
```

### random_forest — 6h

```json
{
  "n_estimators": 180,
  "max_depth": 14,
  "min_samples_split": 4,
  "min_samples_leaf": 2,
  "max_features": 0.75,
  "n_jobs": 4,
  "random_state": 42
}
```

### xgboost — 1h

```json
{
  "n_estimators": 320,
  "learning_rate": 0.04,
  "max_depth": 5,
  "subsample": 0.85,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.05,
  "reg_lambda": 1.0,
  "n_jobs": 4,
  "random_state": 42,
  "objective": "reg:squarederror"
}
```

### xgboost — 3h

```json
{
  "n_estimators": 320,
  "learning_rate": 0.04,
  "max_depth": 5,
  "subsample": 0.85,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.05,
  "reg_lambda": 1.0,
  "n_jobs": 4,
  "random_state": 42,
  "objective": "reg:squarederror"
}
```

### xgboost — 6h

```json
{
  "n_estimators": 320,
  "learning_rate": 0.04,
  "max_depth": 5,
  "subsample": 0.85,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.05,
  "reg_lambda": 1.0,
  "n_jobs": 4,
  "random_state": 42,
  "objective": "reg:squarederror"
}
```

### lightgbm — 1h

```json
{
  "n_estimators": 320,
  "learning_rate": 0.035,
  "num_leaves": 24,
  "max_depth": 8,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.85,
  "bagging_freq": 1,
  "reg_alpha": 0.05,
  "reg_lambda": 0.3,
  "n_jobs": 4,
  "random_state": 42,
  "verbosity": -1
}
```

### lightgbm — 3h

```json
{
  "n_estimators": 320,
  "learning_rate": 0.035,
  "num_leaves": 24,
  "max_depth": 8,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.85,
  "bagging_freq": 1,
  "reg_alpha": 0.05,
  "reg_lambda": 0.3,
  "n_jobs": 4,
  "random_state": 42,
  "verbosity": -1
}
```

### lightgbm — 6h

```json
{
  "n_estimators": 320,
  "learning_rate": 0.035,
  "num_leaves": 24,
  "max_depth": 8,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.85,
  "bagging_freq": 1,
  "reg_alpha": 0.05,
  "reg_lambda": 0.3,
  "n_jobs": 4,
  "random_state": 42,
  "verbosity": -1
}
```

### prophet — 1h

```json
{
  "daily_seasonality": true,
  "weekly_seasonality": false,
  "yearly_seasonality": false,
  "seasonality_mode": "additive",
  "changepoint_prior_scale": 0.05,
  "uncertainty_samples": 0
}
```

### prophet — 3h

```json
{
  "daily_seasonality": true,
  "weekly_seasonality": false,
  "yearly_seasonality": false,
  "seasonality_mode": "additive",
  "changepoint_prior_scale": 0.05,
  "uncertainty_samples": 0
}
```

### prophet — 6h

```json
{
  "daily_seasonality": true,
  "weekly_seasonality": false,
  "yearly_seasonality": false,
  "seasonality_mode": "additive",
  "changepoint_prior_scale": 0.05,
  "uncertainty_samples": 0
}
```

### lstm — 1h

```json
{
  "lookback_samples": 12,
  "lookback_hours": 3,
  "hidden_units": 32,
  "layers": 1,
  "dropout": 0.15,
  "learning_rate": 0.002,
  "batch_size": 64,
  "max_epochs": 12,
  "patience": 3,
  "random_seed": 42
}
```

### lstm — 3h

```json
{
  "lookback_samples": 12,
  "lookback_hours": 3,
  "hidden_units": 32,
  "layers": 1,
  "dropout": 0.15,
  "learning_rate": 0.002,
  "batch_size": 64,
  "max_epochs": 12,
  "patience": 3,
  "random_seed": 42
}
```

### lstm — 6h

```json
{
  "lookback_samples": 12,
  "lookback_hours": 3,
  "hidden_units": 32,
  "layers": 1,
  "dropout": 0.15,
  "learning_rate": 0.002,
  "batch_size": 64,
  "max_epochs": 12,
  "patience": 3,
  "random_seed": 42
}
```

### gru — 1h

```json
{
  "lookback_samples": 12,
  "lookback_hours": 3,
  "hidden_units": 32,
  "layers": 1,
  "dropout": 0.15,
  "learning_rate": 0.002,
  "batch_size": 64,
  "max_epochs": 12,
  "patience": 3,
  "random_seed": 42
}
```

### gru — 3h

```json
{
  "lookback_samples": 12,
  "lookback_hours": 3,
  "hidden_units": 32,
  "layers": 1,
  "dropout": 0.15,
  "learning_rate": 0.002,
  "batch_size": 64,
  "max_epochs": 12,
  "patience": 3,
  "random_seed": 42
}
```

### gru — 6h

```json
{
  "lookback_samples": 12,
  "lookback_hours": 3,
  "hidden_units": 32,
  "layers": 1,
  "dropout": 0.15,
  "learning_rate": 0.002,
  "batch_size": 64,
  "max_epochs": 12,
  "patience": 3,
  "random_seed": 42
}
```

### tft — 1h

```json
{
  "encoder_length": 24,
  "prediction_length": 1,
  "hidden_size": 8,
  "attention_head_size": 1,
  "hidden_continuous_size": 8,
  "dropout": 0.15,
  "learning_rate": 0.01,
  "batch_size": 64,
  "max_epochs": 5,
  "patience": 2,
  "random_seed": 42
}
```

### tft — 3h

```json
{
  "encoder_length": 24,
  "prediction_length": 1,
  "hidden_size": 8,
  "attention_head_size": 1,
  "hidden_continuous_size": 8,
  "dropout": 0.15,
  "learning_rate": 0.01,
  "batch_size": 64,
  "max_epochs": 5,
  "patience": 2,
  "random_seed": 42
}
```

### tft — 6h

```json
{
  "encoder_length": 24,
  "prediction_length": 1,
  "hidden_size": 8,
  "attention_head_size": 1,
  "hidden_continuous_size": 8,
  "dropout": 0.15,
  "learning_rate": 0.01,
  "batch_size": 64,
  "max_epochs": 5,
  "patience": 2,
  "random_seed": 42
}
```

## Warnings and limitations

- prophet 1h: Prophet uses ds/y per sensor and does not consume the multivariate engineered feature set.
- prophet 3h: Prophet uses ds/y per sensor and does not consume the multivariate engineered feature set.
- prophet 6h: Prophet uses ds/y per sensor and does not consume the multivariate engineered feature set.
- lstm 1h: CPU training with a 3-hour sequence; scalers and imputers were fitted on training rows only.
- lstm 3h: CPU training with a 3-hour sequence; scalers and imputers were fitted on training rows only.
- lstm 6h: CPU training with a 3-hour sequence; scalers and imputers were fitted on training rows only.
- gru 1h: CPU training with a 3-hour sequence; scalers and imputers were fitted on training rows only.
- gru 3h: CPU training with a 3-hour sequence; scalers and imputers were fitted on training rows only.
- gru 6h: CPU training with a 3-hour sequence; scalers and imputers were fitted on training rows only.
- tft 1h: TFT used an established PyTorch Forecasting implementation with a compact CPU-safe configuration.
- tft 3h: TFT used an established PyTorch Forecasting implementation with a compact CPU-safe configuration.
- tft 6h: TFT used an established PyTorch Forecasting implementation with a compact CPU-safe configuration.
- The available dataset spans 17 days, so deep-model stability requires more unseen temporal coverage.