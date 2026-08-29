# AirAware Phase 3 - Forecasting Dataset Readiness

## Horizon readiness

- 1h: 5,445 valid of 5,636 timestamps (96.6%); 3,808 positive events among 5,257 valid event labels.
- 3h: 5,405 valid of 5,636 timestamps (95.9%); 3,807 positive events among 5,256 valid event labels.
- 6h: 5,350 valid of 5,636 timestamps (94.9%); 3,811 positive events among 5,259 valid event labels.

The dataset supports prototype evaluation at 1h, 3h, and 6h. Sensors 1, 2, and 4 are the strongest continuity candidates. Sensor 5 must be segmented because of its multi-day outage.

## Forecastable pollutants

PM2.5 is the primary target across all sensors. NO2 and O3 are available only for Tulkarem and require verified units before health-event labeling. Existing source AQI remains excluded as ground truth.

## Modeling structure recommendation

Start with per-sensor PM2.5 baselines, then compare a combined model with explicit sensor/city identifiers. Do not place Tulkarem NO2/O3 into a universal numeric feature matrix without a model-specific structural-missingness design. The 17-day span is too short for robust long-season claims.

## Validation and preprocessing

Use chronological splits with at least complete daily cycles in validation/test and walk-forward evaluation. Fit imputation, scaling, and feature selection on each training fold only. Tree models generally do not require scaling; neural/time-series models typically do.

For an initial benchmark on the observed 2026-08-06 through 2026-08-23 window, use the oldest 10 calendar days for training, the next 4 days for validation, and the final 4 days as untouched test data. This is a date-based plan, not a random percentage split. Walk-forward evaluation should be the primary comparison, and Sensor 5 must remain segmented around its outage.

## Event evaluation

Event labels use the valid future PM2.5 24-hour rolling average ending at each horizon, not an instantaneous reading and not source AQI. Event prevalence must be checked per fold before choosing event metrics.
