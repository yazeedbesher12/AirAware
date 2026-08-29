# AirAware Phase 3 - Feature Engineering Report

## Output

- Candidate engineered features: 58.
- Evidence-driven lag families: 16.
- Selected backward rolling features: 23.
- Highly redundant pairs flagged: 23.

## Evidence-driven design

PM2.5 lags at 15 minutes, 1 hour, 3 hours, and 6 hours reflect Phase 2 persistence. Temperature/humidity and Tulkarem-only NO2/O3 use 15-minute, 1-hour, and 3-hour lags. Rolling features focus on 1-hour, 3-hour, and 6-hour behavior; the 24-hour PM2.5 rolling average is reserved for WHO reference analysis.

Day-of-month, week-of-year, and month were deliberately omitted because the dataset spans only 17 days and these calendar identifiers would not provide stable repeated seasonal evidence. Hour and day-of-week use cyclic encodings.

## Strong candidate associations

- pm25 -> target_pm25_1h: |r|=0.886, n=5445.
- pm25_humidity_interaction -> target_pm25_1h: |r|=0.881, n=5445.
- pm25_mean_1h -> target_pm25_1h: |r|=0.878, n=5445.
- pm25_lag_15m -> target_pm25_1h: |r|=0.867, n=5419.
- pm25_mean_3h -> target_pm25_1h: |r|=0.832, n=5445.
- pm25_min_3h -> target_pm25_1h: |r|=0.827, n=5445.
- pm25_median_3h -> target_pm25_1h: |r|=0.827, n=5445.
- pm25_lag_1h -> target_pm25_1h: |r|=0.816, n=5418.
- pm25 -> target_pm25_3h: |r|=0.742, n=5405.
- pm25_humidity_interaction -> target_pm25_3h: |r|=0.742, n=5405.
- pm25_mean_1h -> target_pm25_3h: |r|=0.734, n=5405.
- pm25_lag_15m -> target_pm25_3h: |r|=0.725, n=5379.
- pm25_mean_3h -> target_pm25_3h: |r|=0.678, n=5405.
- pm25_min_3h -> target_pm25_3h: |r|=0.677, n=5405.
- pm25_median_3h -> target_pm25_3h: |r|=0.675, n=5405.
- pm25_lag_1h -> target_pm25_3h: |r|=0.672, n=5378.

## Missingness and structural absence

No engineered value is zero-filled. Initial-history lag/rolling nulls, temporary missing readings, and structural NO2/O3 absence remain distinguishable through metadata and feature availability.

## Gap and leakage controls

All time operations reset at `continuous_segment_id`. Features use t or earlier; targets use exact future timestamps and remain in a separate file.
