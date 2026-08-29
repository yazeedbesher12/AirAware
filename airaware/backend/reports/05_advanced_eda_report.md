# AirAware Phase 2 — Advanced EDA Report

Generated: 2026-08-24T14:12:09.619875+00:00  
Source: Phase 1 cleaned dataset (5,636 rows). No observations were changed or removed.

## Statistical findings

- PM25: mean 19.31, median 17.63, P95 35.06, P99 41.51, skewness 1.14, kurtosis 2.58.
- TEMPERATURE: mean 31.45, median 30.25, P95 43.18, P99 48.47, skewness 1.09, kurtosis 1.16.
- HUMIDITY: mean 52.14, median 54.56, P95 70.59, P99 74.78, skewness -0.43, kurtosis -0.79.
- NO2: mean 12.07, median 12.08, P95 12.14, P99 12.24, skewness 0.36, kurtosis 3.29.
- O3: mean 55.21, median 56.38, P95 77.64, P99 84.83, skewness -0.51, kurtosis 0.88.
- AQI: mean 102.12, median 98.00, P95 171.00, P99 191.00, skewness 1.15, kurtosis 8.86.

## Temporal patterns and seasonality

- Sensor 1 PM25 has its highest hourly mean at 04:00 UTC (21.56) and lowest at 15:00 UTC (13.24).
- Sensor 1 TEMPERATURE has its highest hourly mean at 15:00 UTC (30.94) and lowest at 03:00 UTC (23.63).
- Sensor 1 HUMIDITY has its highest hourly mean at 02:00 UTC (61.52) and lowest at 11:00 UTC (42.42).
- Sensor 1 NO2 has its highest hourly mean at 16:00 UTC (12.17) and lowest at 15:00 UTC (12.01).
- Sensor 1 O3 has its highest hourly mean at 08:00 UTC (76.68) and lowest at 03:00 UTC (42.77).
- Sensor 1 AQI has its highest hourly mean at 03:00 UTC (116.25) and lowest at 15:00 UTC (73.52).
- Sensor 2 PM25 has its highest hourly mean at 23:00 UTC (24.36) and lowest at 14:00 UTC (14.90).
- Sensor 2 TEMPERATURE has its highest hourly mean at 14:00 UTC (36.07) and lowest at 01:00 UTC (29.08).
- Sensor 2 HUMIDITY has its highest hourly mean at 18:00 UTC (64.93) and lowest at 09:00 UTC (42.32).
- Sensor 2 AQI has its highest hourly mean at 19:00 UTC (122.01) and lowest at 14:00 UTC (82.74).
- Sensor 4 PM25 has its highest hourly mean at 23:00 UTC (22.93) and lowest at 14:00 UTC (12.48).
- Sensor 4 TEMPERATURE has its highest hourly mean at 14:00 UTC (45.42) and lowest at 03:00 UTC (29.64).

- Daily STL for Sensor 1 PM25 has seasonal strength 0.39 and trend strength 0.57; residual spikes remain for anomaly review.
- Daily STL for Sensor 1 NO2 has seasonal strength 0.57 and trend strength 0.06; residual spikes remain for anomaly review.
- Daily STL for Sensor 1 O3 has seasonal strength 0.59 and trend strength 0.31; residual spikes remain for anomaly review.
- Daily STL for Sensor 1 AQI has seasonal strength 0.60 and trend strength 0.64; residual spikes remain for anomaly review.
- Daily STL for Sensor 2 PM25 has seasonal strength 0.44 and trend strength 0.65; residual spikes remain for anomaly review.
- Daily STL for Sensor 2 AQI has seasonal strength 0.25 and trend strength 0.54; residual spikes remain for anomaly review.
- Daily STL for Sensor 4 PM25 has seasonal strength 0.46 and trend strength 0.59; residual spikes remain for anomaly review.
- Daily STL for Sensor 4 AQI has seasonal strength 0.42 and trend strength 0.59; residual spikes remain for anomaly review.

## Strongest autocorrelation lags

- Sensor 4 HUMIDITY, 15 min: r=0.996, n=1589.
- Sensor 5 HUMIDITY, 15 min: r=0.996, n=822.
- Sensor 5 TEMPERATURE, 15 min: r=0.995, n=822.
- Sensor 4 TEMPERATURE, 15 min: r=0.995, n=1589.
- Sensor 2 HUMIDITY, 15 min: r=0.994, n=1575.
- Sensor 1 HUMIDITY, 15 min: r=0.992, n=1618.
- Sensor 2 TEMPERATURE, 15 min: r=0.991, n=1575.
- Sensor 1 TEMPERATURE, 15 min: r=0.988, n=1620.
- Sensor 2 PM25, 15 min: r=0.988, n=1575.
- Sensor 4 PM25, 15 min: r=0.975, n=1589.
- Sensor 1 O3, 15 min: r=0.949, n=1620.
- Sensor 2 AQI, 15 min: r=0.936, n=1575.

## Strongest within-sensor correlations

- Sensor 5: TEMPERATURE ↔ HUMIDITY, Pearson r=-0.957, n=829.
- Sensor 1: TEMPERATURE ↔ HUMIDITY, Pearson r=-0.901, n=1627.
- Sensor 1: PM25 ↔ AQI, Pearson r=0.867, n=1627.
- Sensor 4: PM25 ↔ AQI, Pearson r=0.852, n=1597.
- Sensor 2: PM25 ↔ AQI, Pearson r=0.786, n=1582.
- Sensor 4: TEMPERATURE ↔ HUMIDITY, Pearson r=-0.785, n=1597.
- Sensor 2: TEMPERATURE ↔ HUMIDITY, Pearson r=-0.776, n=1582.
- Sensor 2: PM25 ↔ HUMIDITY, Pearson r=0.738, n=1582.
- Sensor 5: PM25 ↔ AQI, Pearson r=0.735, n=829.
- Sensor 4: PM25 ↔ HUMIDITY, Pearson r=0.734, n=1597.

## Monotonic/nonlinear signals

- Sensor 1: PM25 ↔ AQI, Spearman=0.995, Pearson=0.867.
- Sensor 2: PM25 ↔ AQI, Spearman=0.926, Pearson=0.786.
- Sensor 5: PM25 ↔ AQI, Spearman=0.893, Pearson=0.735.

## Lagged cross-feature relationships

- Sensor 5: TEMPERATURE(t) is associated with HUMIDITY(t-15 min), r=-0.957, n=822.
- Sensor 5: HUMIDITY(t) is associated with TEMPERATURE(t+15 min), r=-0.957, n=822.
- Sensor 1: HUMIDITY(t) is associated with TEMPERATURE(t+15 min), r=-0.901, n=1619.
- Sensor 1: TEMPERATURE(t) is associated with HUMIDITY(t-15 min), r=-0.901, n=1619.
- Sensor 4: HUMIDITY(t) is associated with TEMPERATURE(t+15 min), r=-0.787, n=1589.
- Sensor 4: TEMPERATURE(t) is associated with HUMIDITY(t-15 min), r=-0.787, n=1589.
- Sensor 2: HUMIDITY(t) is associated with TEMPERATURE(t+15 min), r=-0.777, n=1575.
- Sensor 2: TEMPERATURE(t) is associated with HUMIDITY(t-15 min), r=-0.777, n=1575.
- Sensor 2: PM25(t) is associated with HUMIDITY(t-15 min), r=0.736, n=1575.
- Sensor 2: HUMIDITY(t) is associated with PM25(t+15 min), r=0.736, n=1575.

## Mutual information

- global All sensors: AQI → PM25, MI=4.756, Pearson=0.800, Spearman=0.955.
- global All sensors: PM25 → AQI, MI=4.751, Pearson=0.800, Spearman=0.955.
- sensor 4: PM25 → AQI, MI=4.653, Pearson=0.852, Spearman=0.962.
- sensor 4: AQI → PM25, MI=4.648, Pearson=0.852, Spearman=0.962.
- sensor 2: PM25 → AQI, MI=4.640, Pearson=0.786, Spearman=0.926.
- sensor 2: AQI → PM25, MI=4.638, Pearson=0.786, Spearman=0.926.
- sensor 1: PM25 → AQI, MI=4.562, Pearson=0.867, Spearman=0.995.
- sensor 1: AQI → PM25, MI=4.562, Pearson=0.867, Spearman=0.995.
- sensor 5: AQI → PM25, MI=4.354, Pearson=0.735, Spearman=0.893.
- sensor 5: PM25 → AQI, MI=4.352, Pearson=0.735, Spearman=0.893.

## PCA

- Tulkarem: 3 components explain at least 90%; features used: pm25, temperature, humidity, no2, o3, aqi.
- Nablus: 2 components explain at least 90%; features used: pm25, temperature, humidity, aqi.

## Clustering

- Tulkarem: K-Means selected k=2 with silhouette 0.416; cluster labels remain neutral.
- Nablus: K-Means selected k=2 with silhouette 0.451; cluster labels remain neutral.

## Anomaly findings

- Advanced anomaly event rows: 962; no point was removed.
- Events supported by at least two methods: 125.
- Categories: {'MULTIVARIATE_ANOMALY': 645, 'LIKELY_ENVIRONMENTAL_EVENT': 214, 'UNCERTAIN': 90, 'PERSISTENT_SHIFT': 7, 'ISOLATED_SPIKE': 6}.

## Change-point findings

- Behavioral regime shifts detected: 19.
- Sensor 4 PM25 at 2026-08-09 11:30:00+00:00: MEAN_SHIFT, magnitude=-15.520.
- Sensor 4 HUMIDITY at 2026-08-09 09:30:00+00:00: MEAN_SHIFT, magnitude=-29.008.
- Sensor 4 AQI at 2026-08-09 11:30:00+00:00: MEAN_SHIFT, magnitude=-65.979.
- Sensor 2 PM25 at 2026-08-09 11:45:00+00:00: MEAN_SHIFT, magnitude=-14.541.
- Sensor 2 AQI at 2026-08-09 11:45:00+00:00: MEAN_SHIFT, magnitude=-61.661.
- Sensor 2 AQI at 2026-08-13 01:45:00+00:00: MEAN_SHIFT, magnitude=58.333.
- Sensor 2 HUMIDITY at 2026-08-09 21:45:00+00:00: MEAN_SHIFT, magnitude=-22.496.
- Sensor 1 PM25 at 2026-08-16 12:00:00+00:00: MEAN_SHIFT, magnitude=-8.881.
- Sensor 2 HUMIDITY at 2026-08-13 01:45:00+00:00: MEAN_SHIFT, magnitude=20.761.
- Sensor 1 PM25 at 2026-08-12 16:00:00+00:00: MEAN_SHIFT, magnitude=8.457.
- Sensor 4 PM25 at 2026-08-16 23:30:00+00:00: MEAN_SHIFT, magnitude=-11.085.
- Sensor 4 PM25 at 2026-08-20 17:30:00+00:00: MEAN_SHIFT, magnitude=10.797.

## Existing AQI dependency warning

**Existing AQI methodology not yet verified.** AQI relationships remain investigative and no formula, health threshold, or replacement AQI was inferred.
