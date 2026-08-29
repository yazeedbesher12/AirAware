# AirAware Phase 1 — Conservative Cleaning Report

The source file was never overwritten. Parsing normalized timestamps and numeric representation in the derived file only. No AQI values were replaced, no structurally unavailable readings were fabricated, and no statistical outliers were automatically removed.

- Original rows: 5,636
- Cleaned rows: 5,636
- Rows removed: 0
- Observation values modified: 0
- Suspicious event records retained: 399
- Metadata columns added: `missing_flag`, `sampling_gap_flag`, `suspicious_value_flag`, `flatline_flag`, `jump_flag`, `possible_drift_flag`, `overall_quality_flag`

The machine-readable modification ledger is `outputs/cleaning_change_log.csv`; it contains only its header because no source field was changed.
