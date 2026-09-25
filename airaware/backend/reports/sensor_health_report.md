# AirAware Sensor Health Report

> The health score is an operational score, not a probability, accuracy, or confidence estimate.

## Existing quality logic

Reused semantics: modal per-sensor cadence; Phase 1's 1.5x gap tolerance, robust MAD/IQR jump method, eight-sample flatline minimum and broad physical bounds; Phase 3's 90-minute major-gap boundary. Existing row flags are retained for comparison, but causal runtime thresholds use only prior values. Phase 1 missing_flag checks only channels that sensor provides (although it also includes source AQI); sampling_gap_flag marks the returning row; suspicious_value_flag combines broad physical, non-finite, zero-pollutant and data-relative-extreme review events; flatline and possible_drift flags backfill detected event intervals; overall_quality_flag applies precedence. Phase 3 current_quality_review_flag is overall_quality_flag != NORMAL; recent_gap_count_3h and recent_missing_pm25_count_3h are backward rolling counts inside continuous segments; recent_anomaly_count_3h rolls Phase 2 events where at least two anomaly methods agreed. AQI is not treated as a hardware channel in the new engine.

## Sensor availability

| Sensor | Rows | Time range | Largest gap | Availability |
|---:|---:|---|---:|---:|
| 1 | 1,628 | 2026-08-06T00:00:00+00:00 to 2026-08-23T00:30:00+00:00 | 30 min | 99.572% |
| 2 | 1,582 | 2026-08-06T11:45:00+00:00 to 2026-08-23T00:30:00+00:00 | 30 min | 99.622% |
| 4 | 1,597 | 2026-08-06T07:30:00+00:00 to 2026-08-23T00:30:00+00:00 | 45 min | 99.502% |
| 5 | 829 | 2026-08-06T09:15:00+00:00 to 2026-08-23T00:30:00+00:00 | 11475 min | 51.877% |

## Supported channels

- Sensor 1: pm25, temperature, humidity, no2, o3
- Sensor 2: pm25, temperature, humidity
- Sensor 4: pm25, temperature, humidity
- Sensor 5: pm25, temperature, humidity

NO2 and O3 are structurally unavailable for Sensors 2, 4 and 5 and never enter missingness, scoring, or alert logic.

## Detectors

- Missing: null or non-finite supported channels only; absent scheduled timestamps are tracked separately.
- Gap/offline: elapsed time from the last known row, plus resumed-arrival gap evidence.
- Flatline: a continuous trailing run at channel-resolution tolerance and minimum duration.
- Jump: current absolute first difference against prior-only robust MAD and IQR limits.
- Noise: repeated short-window sign reversals plus a robust variability-ratio requirement.
- Possible drift: a 24-hour median shift against at least 48 hours of strictly earlier data; this is not confirmed calibration drift.
- Suspicious value: non-finite values and broad physical impossibilities, without narrow environmental normal ranges.
- Cross-sensor inconsistency: four sustained Nablus PM2.5 residuals versus the other two sensors; supporting evidence only.

## Thresholds and score

- Sampling: expected=15 min; gap tolerance=22.5 min (existing Phase 1 1.5x rule); major=90 min (existing Phase 3); offline=180 min (new operational threshold).
- Flatline: 8 samples and 105 min; 0.01-unit channel-resolution floors (existing method, precision corrected from observed data).
- Jump: at least 24 prior differences; max(median+8xMAD, Q3+3xIQR, resolution), retaining the Phase 1 robust multipliers.
- Noise: 8 differences, 48 reference differences, >4x robust scale and at least 5 sign changes (new conservative calibration).
- Drift: 96 recent samples, 192 earlier baseline samples, >3x robust scale and >=80% persistence (causal extension of Phase 1).
- Cross-sensor: 4 sustained samples after 48 references and >4x robust residual scale (new supporting rule).
- Dimensions start at 100. Penalties: missing 15 availability; minor gap 15 availability; major gap 45 availability; offline 100 availability; suspicious value 45 integrity; jump 10 stability; flatline 30 stability; noise 20 stability; cross-sensor 15 consistency; possible drift 30 drift.
- Weighted score: 35% availability + 25% integrity + 20% stability + 10% consistency + 10% drift, bounded 0-100. Offline forces 0. This is operational, not probabilistic.
- Status: Offline overrides everything; any critical issue forces Critical; otherwise no-issue score >=90 is Healthy, score >=65 is Warning, and lower scores are Critical.

## Historical results

| Sensor | Healthy | Warning | Critical | Offline | Lowest score | Final |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 94.312% | 5.505% | 0.183% | 0.000% | 87 | Healthy (100) |
| 2 | 85.390% | 14.610% | 0.000% | 0.000% | 92 | Healthy (100) |
| 4 | 96.012% | 3.988% | 0.000% | 0.000% | 95 | Healthy (100) |
| 5 | 50.250% | 2.190% | 0.438% | 47.121% | 0 | Healthy (100) |

## Event counts

- Sensor 1: missing_reading=2, missing_timestamp=0, sampling_gap=7, major_sampling_gap=0, offline=0, flatline=1, jump=70, abnormal_noise=15, possible_drift=0, suspicious_value=1, cross_sensor_inconsistency=0
- Sensor 2: missing_reading=0, missing_timestamp=0, sampling_gap=6, major_sampling_gap=0, offline=0, flatline=0, jump=15, abnormal_noise=0, possible_drift=3, suspicious_value=0, cross_sensor_inconsistency=1
- Sensor 4: missing_reading=0, missing_timestamp=1, sampling_gap=7, major_sampling_gap=0, offline=0, flatline=0, jump=31, abnormal_noise=3, possible_drift=0, suspicious_value=0, cross_sensor_inconsistency=1
- Sensor 5: missing_reading=0, missing_timestamp=1, sampling_gap=5, major_sampling_gap=2, offline=1, flatline=0, jump=12, abnormal_noise=1, possible_drift=0, suspicious_value=0, cross_sensor_inconsistency=1

## Representative events

- Sensor 5, 2026-08-09T14:00:00+00:00: offline (critical, nan) — Sensor has not reported for 11460 minutes.
- Sensor 5, 2026-08-09T12:30:00+00:00: major_sampling_gap (critical, nan) — Sensor has a major reporting gap of 165 minutes.
- Sensor 1, 2026-08-16T16:15:00+00:00: suspicious_value (critical, o3) — o3=-2.32 is outside broad physical plausibility bounds.
- Sensor 5, 2026-08-17T10:15:00+00:00: major_sampling_gap (critical, nan) — Reporting resumed after a 11475-minute major gap.
- Sensor 2, 2026-08-13T16:00:00+00:00: possible_drift (warning, pm25) — Possible Drift: pm25 shows a sustained +15.90 offset from its trailing baseline.
- Sensor 2, 2026-08-14T00:30:00+00:00: possible_drift (warning, humidity) — Possible Drift: humidity shows a sustained +32.03 offset from its trailing baseline.
- Sensor 5, 2026-08-09T11:30:00+00:00: missing_timestamp (warning, nan) — No sensor row arrived at the expected timestamp; 5 interval(s) are missing.
- Sensor 4, 2026-08-11T16:00:00+00:00: jump (warning, temperature) — temperature changed by 2.36, above its causal robust threshold 1.99.
- Sensor 2, 2026-08-13T16:00:00+00:00: possible_drift (warning, humidity) — Possible Drift: humidity shows a sustained +16.50 offset from its trailing baseline.
- Sensor 5, 2026-08-20T03:15:00+00:00: jump (warning, pm25) — pm25 changed by 17.31, above its causal robust threshold 8.49.
- Sensor 2, 2026-08-21T04:15:00+00:00: cross_sensor_inconsistency (warning, pm25) — PM2.5 differs persistently from both other Nablus sensors; this is supporting review evidence only.
- Sensor 5, 2026-08-21T04:15:00+00:00: cross_sensor_inconsistency (warning, pm25) — PM2.5 differs persistently from both other Nablus sensors; this is supporting review evidence only.
- Sensor 2, 2026-08-11T16:15:00+00:00: jump (warning, temperature) — temperature changed by 1.16, above its causal robust threshold 1.09.
- Sensor 1, 2026-08-12T16:00:00+00:00: jump (warning, temperature) — temperature changed by 1.42, above its causal robust threshold 1.35.
- Sensor 4, 2026-08-12T19:30:00+00:00: jump (warning, humidity) — humidity changed by 7.77, above its causal robust threshold 5.65.
- Sensor 4, 2026-08-14T05:30:00+00:00: abnormal_noise (warning, pm25) — pm25 is rapidly oscillating above its historical variability.
- Sensor 1, 2026-08-16T14:45:00+00:00: abnormal_noise (warning, no2) — no2 is rapidly oscillating above its historical variability.
- Sensor 5, 2026-08-20T00:45:00+00:00: jump (warning, pm25) — pm25 changed by 7.55, above its causal robust threshold 3.89.
- Sensor 4, 2026-08-21T04:30:00+00:00: cross_sensor_inconsistency (warning, pm25) — PM2.5 differs persistently from both other Nablus sensors; this is supporting review evidence only.
- Sensor 5, 2026-08-06T21:30:00+00:00: jump (warning, pm25) — pm25 changed by 12.41, above its causal robust threshold 5.69.

## Existing-flag comparison

- jump_flag: existing=276, engine=158, both=124, all-row agreement=96.700%.
- flatline_flag: existing=8, engine=1, both=1, all-row agreement=99.876%.
- possible_drift_flag: existing=43, engine=204, both=0, all-row agreement=95.617%.
- sampling_gap_flag: existing=26, engine=26, both=26, all-row agreement=100.000%.
- suspicious_value_flag: existing=7, engine=3, both=3, all-row agreement=99.929%.

Differences are intentional: the engine cannot reproduce Phase 1's full-dataset threshold leakage. Early jumps lack the required prior calibration; flatlines alert only after the eighth sample rather than backfilling the whole run; the old 43-row drift interval was backfilled on Sensor 1 while the causal engine found three later Sensor 2 review events; old suspicious flags also include data-relative extremes, whereas runtime critical flags are limited to physical/non-finite failures.

## Causal and false-positive review

All evaluation filters history at the requested timestamp. Jump calibration excludes the current transition; drift compares a trailing recent window only with earlier baseline values; gap state comes from the last known reading; cross-sensor evidence uses only peer values available by the same timestamp. Noise requires both a fourfold robust-scale increase and repeated sign reversals. Cross-sensor inconsistency is supporting evidence and is never a hard failure by itself.

Event concentration by sensor/channel:

- jump: Sensor 1, pm25 = 26 merged event(s).
- jump: Sensor 1, temperature = 19 merged event(s).
- jump: Sensor 4, pm25 = 17 merged event(s).
- jump: Sensor 5, pm25 = 12 merged event(s).
- jump: Sensor 2, pm25 = 10 merged event(s).
- jump: Sensor 1, o3 = 10 merged event(s).
- abnormal_noise: Sensor 1, no2 = 9 merged event(s).
- jump: Sensor 1, no2 = 8 merged event(s).
- sampling_gap: Sensor 1, all channels = 7 merged event(s).
- jump: Sensor 4, temperature = 7 merged event(s).
- jump: Sensor 4, humidity = 7 merged event(s).
- jump: Sensor 1, humidity = 7 merged event(s).
- sampling_gap: Sensor 4, all channels = 7 merged event(s).
- sampling_gap: Sensor 2, all channels = 6 merged event(s).
- sampling_gap: Sensor 5, all channels = 5 merged event(s).
- jump: Sensor 2, temperature = 3 merged event(s).
- abnormal_noise: Sensor 4, pm25 = 3 merged event(s).
- possible_drift: Sensor 2, humidity = 2 merged event(s).
- abnormal_noise: Sensor 1, o3 = 2 merged event(s).
- abnormal_noise: Sensor 1, pm25 = 2 merged event(s).

The initial replay exposed an over-sensitive NO2 resolution assumption; after correction, flatline output fell to one merged event. Jumps are most concentrated on Sensor 1 PM2.5 (26 events), reflecting its five available channels and variable source stream. Noise remains sparse (19 merged events overall). The three long Sensor 2 drift events deserve domain review but are labeled only as possible drift.

## Future ML recommendation

Do not add Isolation Forest or an autoencoder yet. The short dataset lacks labeled hardware faults and calibration truth; the conservative rule/statistical layer is more interpretable and covers the actionable availability, integrity, stability, consistency and possible-drift cases.
