# AirAware Phase 2 — Modeling Readiness Report

This report prepares decisions only. No forecasting model or final ML feature engineering was performed.

## Which variables appear useful for forecasting?

PM2.5, temperature, humidity, and—only in Tulkarem—NO2/O3 have sufficient volume for temporal evaluation. The existing AQI must remain an investigative field because its methodology is unverified.

## Which lags appear useful?

- Sensor 4 HUMIDITY: 15 min autocorrelation 0.996.
- Sensor 5 HUMIDITY: 15 min autocorrelation 0.996.
- Sensor 5 TEMPERATURE: 15 min autocorrelation 0.995.
- Sensor 4 TEMPERATURE: 15 min autocorrelation 0.995.
- Sensor 2 HUMIDITY: 15 min autocorrelation 0.994.
- Sensor 1 HUMIDITY: 15 min autocorrelation 0.992.
- Sensor 2 TEMPERATURE: 15 min autocorrelation 0.991.
- Sensor 1 TEMPERATURE: 15 min autocorrelation 0.988.
- Sensor 2 PM25: 15 min autocorrelation 0.988.
- Sensor 4 PM25: 15 min autocorrelation 0.975.

Cross-feature candidates:
- Sensor 5: TEMPERATURE(t) ↔ HUMIDITY(t-15 min), r=-0.957.
- Sensor 5: HUMIDITY(t) ↔ TEMPERATURE(t+15 min), r=-0.957.
- Sensor 5: HUMIDITY(t) ↔ TEMPERATURE(t+30 min), r=-0.950.
- Sensor 5: TEMPERATURE(t) ↔ HUMIDITY(t-30 min), r=-0.950.
- Sensor 5: HUMIDITY(t) ↔ TEMPERATURE(t-15 min), r=-0.949.
- Sensor 5: TEMPERATURE(t) ↔ HUMIDITY(t+15 min), r=-0.949.
- Sensor 5: HUMIDITY(t) ↔ TEMPERATURE(t-30 min), r=-0.935.
- Sensor 5: TEMPERATURE(t) ↔ HUMIDITY(t+30 min), r=-0.935.
- Sensor 5: TEMPERATURE(t) ↔ HUMIDITY(t-1 h), r=-0.920.
- Sensor 5: HUMIDITY(t) ↔ TEMPERATURE(t+1 h), r=-0.920.
- Sensor 1: HUMIDITY(t) ↔ TEMPERATURE(t+15 min), r=-0.901.
- Sensor 1: TEMPERATURE(t) ↔ HUMIDITY(t-15 min), r=-0.901.

## Which rolling windows appear useful?

The 1h and 3h windows preserve short-term movement; 6h and 12h summarize medium persistence; 24h is appropriate for daily-baseline context. Selection should later be validated with leakage-safe time splits.

## Which pollutants have strong cross-feature relationships?

- Sensor 5: TEMPERATURE ↔ HUMIDITY, Pearson=-0.957.
- Sensor 1: TEMPERATURE ↔ HUMIDITY, Pearson=-0.901.
- Sensor 1: PM25 ↔ AQI, Pearson=0.867.
- Sensor 4: PM25 ↔ AQI, Pearson=0.852.
- Sensor 2: PM25 ↔ AQI, Pearson=0.786.
- Sensor 4: TEMPERATURE ↔ HUMIDITY, Pearson=-0.785.
- Sensor 2: TEMPERATURE ↔ HUMIDITY, Pearson=-0.776.
- Sensor 2: PM25 ↔ HUMIDITY, Pearson=0.738.

## Which features appear redundant?

PCA indicates potential redundancy only statistically:
- Tulkarem: 3 components explain at least 90%; features used: pm25, temperature, humidity, no2, o3, aqi.
- Nablus: 2 components explain at least 90%; features used: pm25, temperature, humidity, aqi.

## Which sensors contain enough continuous data for forecasting?

Sensors 1, 2, 4 have long coverage with no multi-hour outage. Sensor 5 requires segmented treatment because of its multi-day outage.

## Which sensors contain gaps requiring special treatment?

- Sensor 1 (0.50 h longest gap)
- Sensor 2 (0.50 h longest gap)
- Sensor 4 (0.75 h longest gap)
- Sensor 5 (191.25 h longest gap)

## Which observations should remain flagged?

Retain Phase 1 quality flags and all Phase 2 anomaly consensus records. During future training, evaluate flagged observations with time-aware sensitivity analyses instead of deleting them automatically. Structural NO2/O3 absence in Nablus must remain unavailable, never zero-filled.
