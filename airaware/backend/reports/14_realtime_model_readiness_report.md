# 14 Real-Time Model Readiness Report

Historical test rows were replayed chronologically, one reading at a time, with an independent gap-aware buffer per sensor. No model retrained during inference.

| Model | Accuracy (1h RMSE) | Avg inference ms | P95 ms | Model size MB | Minimum history | CPU ready | GPU required | Real-time ready |
|---|---:|---:|---:|---:|---|---|---|---|
| persistence | 5.21470740573858 | 0.17827760922742325 | 0.26242500007356284 | 0.005710601806640625 | 1 valid reading | True | False | True |
| random_forest | 4.36933141710931 | 21.916104550970964 | 25.26157499960391 | 84.30990409851074 | 6 hours | True | False | True |
| xgboost | 4.363273403625348 | 3.2033740594623326 | 3.8446000000931235 | 4.721982002258301 | 6 hours | True | False | True |
| lightgbm | 4.118352410694059 | 3.550752032763351 | 4.4057749996682105 | 4.018791198730469 | 6 hours | True | False | True |
| prophet | 14.36951459171583 | 29.33811350120815 | 34.641725000255974 | 3.9213504791259766 | current timestamp and sensor identity | True | False | True |
| lstm | 4.376383961430188 | 3.093574089814907 | 3.920825000250261 | 0.1943502426147461 | approximately 9 hours | True | False | True |
| gru | 4.249871018707299 | 3.296119538824537 | 4.243224999981976 | 0.15395259857177734 | approximately 9 hours | True | False | True |
| tft | N/A | N/A | N/A | 2.817 | 30 hours full-feature history (6h encoder + 24h feature lookback) | True | False | False |

## Operational interpretation
Tree models and persistence are the easiest CPU deployment candidates. Prophet is reloadable but is sensor-specific and does not share the multivariate feature set. LSTM/GRU require sequence state and approximately nine hours of warm history. TFT is computationally heavier and is not marked deployment-ready until a dedicated incremental adapter passes the same replay.

## Missing readings
A delayed interval remains missing. A gap above 90 minutes resets continuity; the service must return prediction_available=false until the model-specific history requirement is restored.

No final model is selected here. Accuracy and later ensemble/stacking evidence must be reviewed together.

## TFT checkpoint deployment audit

All three TFT checkpoints reload on CPU in a fresh Python process and reproduce their stored offline predictions within 1e-5. The learned weights were not changed and no retraining occurred.

The checkpoint encoder is 24 readings, which is exactly 6 hours at the 15-minute cadence. The selected inputs also contain backward-looking PM2.5 WHO 24-hour features. Full input parity therefore requires a 30-hour continuous raw buffer (6-hour encoder plus 24-hour maximum feature lookback). The earlier approximately 12-hour statement was incorrect.

Real-time readiness remains **false**. The fitted `MultiNormalizer` contains `EncoderNormalizer` instances that calculate target center/scale from encoder target columns. Those target columns are Phase 3 future-shifted targets and are not fully known at forecast time. Supplying them would leak future observations; inventing replacements would create training-serving skew.

Causal replay was blocked before prediction by the fail-closed adapter. Real-time average/p95 latency is therefore N/A. Offline single-sample model-forward benchmarks are:

| Horizon | Offline avg forward ms | Offline p95 forward ms | Reload difference |
|---|---:|---:|---:|
| 1h | 25.378 | 27.314 | 0.00000191 |
| 3h | 24.670 | 27.290 | 0.00000191 |
| 6h | 25.145 | 27.285 | 0.00000191 |

A causal target-normalization design requires retraining. Per instruction, retraining was not started.
