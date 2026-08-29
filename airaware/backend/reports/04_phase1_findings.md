# AirAware Phase 1 — Findings Before Advanced EDA

## Evidence summary

1. Best completeness: sensor 2 (ANNU New Campus) at 0.00% missing among measurements it provides.
2. Worst completeness: sensor 1 (Tulkarem Munaplicity) at 0.02% missing among measurements it provides.
3. Largest outage: sensor 5, 191.25 hours (2026-08-09T11:00:00+00:00 to 2026-08-17T10:15:00+00:00).
4. Major/critical gap count: 1.
5. Suspicious value/jump event records retained: 399.
6. Possible stuck-pattern periods: 1; these are patterns, not confirmed hardware failures.
7. Exploratory behavior-change periods: 1; these do not confirm calibration drift.
8. Strongest Nablus PM2.5 agreement: sensors 2 ↔ 4 (Pearson 0.962, n=1580).
9. Largest Nablus PM2.5 divergence: sensors 4 ↔ 5 (MAE 2.94 µg/m³, n=829).
10. Existing AQI range: 11–637; values above 500: 1. Existing AQI methodology is not verified.

## Major concerns before Advanced EDA

Structural NO2/O3 unavailability is kept distinct from random missingness. Multi-day outages can bias comparisons. Robust jump, flat-line, and baseline-shift flags require contextual review; no pollution cause or hardware failure is inferred here. The existing AQI must remain exploratory until its source methodology is established.
