# AirAware Phase 3 - Feature Leakage Audit

Generated: 2026-08-24T15:47:39.121465+00:00

## Result

All 58 candidate feature columns passed the construction-time leakage audit.

## Controls

- Every lag is joined by exact elapsed timestamp inside one `continuous_segment_id`.
- Every rolling feature is backward-looking (`closed=right`) and never centered.
- Rolling and lag state resets after outages longer than 90 minutes.
- Future concentration and event values exist only in the target dataset.
- Current quality features use flags known at or before timestamp t.
- Candidate ranking is explicitly exploratory; final selection and scaling must be fitted on training folds only.
- No full-dataset normalization, imputation, future WHO status, or future anomaly state is present in the ML feature file.

## Rejected designs

- Centered rolling averages.
- Row-shift lags that could cross Sensor 5's multi-day gap.
- Source AQI as a target or ground truth.
- Instantaneous PM2.5 compared to the WHO 24-hour AQG as an official violation.
