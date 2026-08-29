# AirAware Phase 1 — Data Audit

Generated: 2026-08-24T13:30:46.672627+00:00  
Source SHA-256: `879b04ffda8dd3ea846b0ca3276c1ffd2a6ef07a5ae3ab0eb818cdbf89eb7856`

## Structure

- Rows: 5,636
- Columns: 11
- Cities: Nablus, Tulkarem
- Sensors: 4
- Time range: 2026-08-06T00:00:00+00:00 to 2026-08-23T00:30:00+00:00
- Memory use: 1,096,789 bytes
- Timestamp parse failures: 0
- Coordinate fields found: none

## Schema

| Column | Parsed type |
|---|---|
| sensor_id | Int64 |
| city_id | Int64 |
| city_name | str |
| location_name | str |
| pm25 | float64 |
| temperature | float64 |
| humidity | float64 |
| no2 | float64 |
| o3 | float64 |
| aqi | int64 |
| timestamp | datetime64[us, UTC] |

## Missingness by column

| Column | Missing | Missing % |
|---|---:|---:|
| sensor_id | 0 | 0.000% |
| city_id | 0 | 0.000% |
| city_name | 0 | 0.000% |
| location_name | 0 | 0.000% |
| pm25 | 1 | 0.018% |
| temperature | 0 | 0.000% |
| humidity | 1 | 0.018% |
| no2 | 4,008 | 71.114% |
| o3 | 4,008 | 71.114% |
| aqi | 0 | 0.000% |
| timestamp | 0 | 0.000% |

## Numeric ranges

| Feature | Count | Min | Max | Mean | Median | Negative | Infinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| pm25 | 5,635 | 5.02 | 96.88 | 19.315 | 17.630 | 0 | 0 |
| temperature | 5,636 | 22.48 | 51.41 | 31.449 | 30.250 | 0 | 0 |
| humidity | 5,635 | 19.61 | 78.32 | 52.139 | 54.560 | 0 | 0 |
| no2 | 1,628 | 11.84 | 12.38 | 12.066 | 12.080 | 0 | 0 |
| o3 | 1,628 | -2.32 | 94.02 | 55.208 | 56.375 | 3 | 0 |
| aqi | 5,636 | 11 | 637 | 102.119 | 98.000 | 0 | 0 |

## Sensor measurement availability

| Sensor | City | Location | PM2.5 | Temperature | Humidity | NO2 | O3 | AQI |
|---:|---|---|---|---|---|---|---|---|
| 1 | Tulkarem | Tulkarem Munaplicity | PARTIALLY_MISSING | AVAILABLE | PARTIALLY_MISSING | AVAILABLE | AVAILABLE | AVAILABLE |
| 2 | Nablus | ANNU New Campus | AVAILABLE | AVAILABLE | AVAILABLE | STRUCTURALLY_UNAVAILABLE | STRUCTURALLY_UNAVAILABLE | AVAILABLE |
| 4 | Nablus | ANNU old campus | AVAILABLE | AVAILABLE | AVAILABLE | STRUCTURALLY_UNAVAILABLE | STRUCTURALLY_UNAVAILABLE | AVAILABLE |
| 5 | Nablus | ANNU HIJJAWI | AVAILABLE | AVAILABLE | AVAILABLE | STRUCTURALLY_UNAVAILABLE | STRUCTURALLY_UNAVAILABLE | AVAILABLE |

## Sensor-specific sampling

| Sensor | City | Expected min | Min interval min | Max interval min | Gaps | Missing expected |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Tulkarem | 15.0 | 15.0 | 30.0 | 7 | 7 |
| 2 | Nablus | 15.0 | 15.0 | 30.0 | 6 | 6 |
| 4 | Nablus | 15.0 | 15.0 | 45.0 | 7 | 8 |
| 5 | Nablus | 15.0 | 15.0 | 11475.0 | 6 | 769 |

## Duplicate interpretation

Only sensor_id + timestamp collisions are treated as timestamp duplicates; simultaneous readings from different sensors are legitimate.

Full duplicate rows: 0. Sensor/timestamp duplicate rows: 0.
