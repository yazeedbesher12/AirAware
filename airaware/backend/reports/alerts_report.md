# AirAware Alerts & Early Warning Replay

> Historical development replay only. Current frontend alerts remain clearly labeled demo data until a real API is integrated.

## Architecture

Normalized ForecastEngine/SensorHealthEngine-shaped objects -> AlertEngine rules -> deterministic deduplication -> lifecycle state -> active/resolved outputs.

## Results

- Total lifecycle events: 80
- Active at final available evaluation: 24
- Resolved: 56
- Deduplicated update notifications suppressed: 4475
- Escalations: 61
- Counts by severity: {'critical': 11, 'warning': 69}
- Counts by type: {'sensor_offline': 1, 'possible_drift': 3, 'suspicious_reading': 1, 'high_pm_risk': 61, 'who_exceedance': 14}
- Counts by sensor: {'5': 24, '2': 17, '1': 27, '4': 12}

## Limitations

Forecast replay uses only existing LightGBM TEST point and health-average predictions. Existing AirAware safety, High-PM risk, and WHO utilities normalize those artifacts. Missing forecast timestamps or horizons are not synthesized. Sensor-health replay uses the generated causal histories.
