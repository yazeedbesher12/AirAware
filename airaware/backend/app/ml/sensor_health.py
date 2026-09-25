"""Causal, interpretable sensor-health evaluation for AirAware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.ml.sensor_health_config import (
    ALERT_RECOMMENDED,
    CHANNEL_RESOLUTION,
    CROSS_SENSOR_MIN_REFERENCE_SAMPLES,
    CROSS_SENSOR_RECENT_SAMPLES,
    CROSS_SENSOR_SCALE_MULTIPLIER,
    DIMENSION_WEIGHTS,
    DRIFT_MIN_BASELINE_SAMPLES,
    DRIFT_RECENT_SAMPLES,
    DRIFT_SCALE_MULTIPLIER,
    EXPECTED_INTERVAL_MINUTES,
    FLATLINE_MIN_DURATION_MINUTES,
    FLATLINE_MIN_SAMPLES,
    HEALTHY_SCORE_MIN,
    ISSUE_PENALTIES,
    JUMP_IQR_MULTIPLIER,
    JUMP_MAD_MULTIPLIER,
    JUMP_MIN_REFERENCE_DIFFS,
    MAJOR_GAP_MINUTES,
    MINOR_GAP_MINUTES,
    NOISE_MIN_REFERENCE_DIFFS,
    NOISE_MIN_SIGN_CHANGES,
    NOISE_SCALE_MULTIPLIER,
    NOISE_WINDOW_SAMPLES,
    OFFLINE_MINUTES,
    PHYSICAL_BOUNDS,
    SENSOR_CHANNELS,
    SENSOR_METADATA,
    WARNING_SCORE_MIN,
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _scaled_mad(values: pd.Series | np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return 0.0
    median = float(np.median(data))
    return float(1.4826 * np.median(np.abs(data - median)))


def _issue(
    issue_type: str,
    severity: str,
    timestamp: pd.Timestamp,
    message: str,
    evidence: dict[str, Any],
    channel: str | None = None,
) -> dict[str, Any]:
    return {
        "type": issue_type,
        "severity": severity,
        "channel": channel,
        "timestamp": timestamp.isoformat(),
        "evidence": evidence,
        "message": message,
        "alert_recommended": ALERT_RECOMMENDED[severity],
    }


@dataclass(frozen=True)
class SensorHealthEngine:
    """Evaluate one sensor from a caller-owned backward-only history buffer.

    ``sensor_history`` is not loaded internally and rows later than ``as_of`` are
    ignored.  ``peer_history`` is optional supporting evidence for Nablus only.
    """

    expected_interval_minutes: int = EXPECTED_INTERVAL_MINUTES

    def evaluate(
        self,
        sensor_history: pd.DataFrame,
        as_of_timestamp: Any,
        peer_history: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        if sensor_history.empty:
            raise ValueError("sensor_history must contain at least one sensor row")
        as_of = pd.Timestamp(as_of_timestamp)
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        else:
            as_of = as_of.tz_convert("UTC")

        data = sensor_history.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        sensor_ids = data["sensor_id"].dropna().astype(int).unique()
        if len(sensor_ids) != 1:
            raise ValueError("sensor_history must contain exactly one sensor_id")
        sensor_id = int(sensor_ids[0])
        if sensor_id not in SENSOR_CHANNELS:
            raise ValueError(f"unsupported sensor_id: {sensor_id}")
        data = data.loc[data.timestamp.le(as_of)].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if data.empty:
            raise ValueError("sensor_history contains no rows at or before as_of_timestamp")

        channels = list(SENSOR_CHANNELS[sensor_id])
        metadata = SENSOR_METADATA[sensor_id]
        issues: list[dict[str, Any]] = []
        current_rows = data.loc[data.timestamp.eq(as_of)]
        current = current_rows.iloc[-1] if not current_rows.empty else None
        last = data.iloc[-1]
        last_at = pd.Timestamp(last.timestamp)
        minutes_since_last = max(0.0, (as_of - last_at).total_seconds() / 60.0)

        if current is None:
            self._detect_absence(issues, as_of, minutes_since_last)
        else:
            self._detect_arrival_gap(issues, data, as_of)
            self._detect_missing_and_suspicious(issues, current, channels, as_of)
            for channel in channels:
                self._detect_flatline(issues, data, channel, as_of)
                self._detect_jump(issues, data, channel, as_of)
                self._detect_noise(issues, data, channel, as_of)
                self._detect_drift(issues, data, channel, as_of)
            if peer_history is not None and sensor_id in {2, 4, 5}:
                self._detect_cross_sensor(issues, data, peer_history, sensor_id, as_of)

        dimensions = {name: 100 for name in DIMENSION_WEIGHTS}
        for item in issues:
            dimension, penalty = ISSUE_PENALTIES[item["type"]]
            dimensions[dimension] = max(0, dimensions[dimension] - penalty)
        score = int(round(sum(dimensions[name] * weight for name, weight in DIMENSION_WEIGHTS.items())))
        score = min(100, max(0, score))
        types = {item["type"] for item in issues}
        if "offline" in types:
            status, score = "Offline", 0
        elif any(item["severity"] == "critical" for item in issues):
            status = "Critical"
        elif score >= HEALTHY_SCORE_MIN and not issues:
            status = "Healthy"
        elif score >= WARNING_SCORE_MIN:
            status = "Warning"
        else:
            status = "Critical"

        summary = {
            "Healthy": "Sensor is operating normally.",
            "Warning": "Sensor is operating, but one or more readings require review.",
            "Critical": "Sensor data has a critical operational integrity or availability issue.",
            "Offline": "Sensor has not reported within the offline threshold.",
        }[status]
        return {
            "sensor_id": sensor_id,
            "site": metadata["site"],
            "city": metadata["city"],
            "timestamp": as_of.isoformat(),
            "status": status,
            "health_score": score,
            "score_interpretation": "AirAware operational sensor-health score; not a probability or confidence estimate.",
            "supported_channels": channels,
            "issues": issues,
            "dimensions": dimensions,
            "last_reading_at": last_at.isoformat(),
            "minutes_since_last_reading": round(minutes_since_last, 3),
            "summary": summary,
        }

    def _detect_absence(self, issues: list[dict[str, Any]], now: pd.Timestamp, delay: float) -> None:
        if delay <= MINOR_GAP_MINUTES:
            return
        missed = max(1, int(np.floor(delay / self.expected_interval_minutes)))
        if delay >= OFFLINE_MINUTES:
            issues.append(_issue("offline", "critical", now, f"Sensor has not reported for {delay:.0f} minutes.", {"gap_minutes": delay, "missed_intervals": missed, "offline_threshold_minutes": OFFLINE_MINUTES}))
        elif delay >= MAJOR_GAP_MINUTES:
            issues.append(_issue("major_sampling_gap", "critical", now, f"Sensor has a major reporting gap of {delay:.0f} minutes.", {"gap_minutes": delay, "missed_intervals": missed, "major_gap_threshold_minutes": MAJOR_GAP_MINUTES}))
        else:
            issues.append(_issue("missing_timestamp", "warning", now, f"No sensor row arrived at the expected timestamp; {missed} interval(s) are missing.", {"gap_minutes": delay, "missed_intervals": missed}))

    def _detect_arrival_gap(self, issues: list[dict[str, Any]], data: pd.DataFrame, now: pd.Timestamp) -> None:
        if len(data) < 2:
            return
        gap = (data.timestamp.iloc[-1] - data.timestamp.iloc[-2]).total_seconds() / 60.0
        if gap <= MINOR_GAP_MINUTES:
            return
        missed = max(1, int(round(gap / self.expected_interval_minutes)) - 1)
        if gap >= MAJOR_GAP_MINUTES:
            issues.append(_issue("major_sampling_gap", "critical", now, f"Reporting resumed after a {gap:.0f}-minute major gap.", {"gap_minutes": gap, "missed_intervals": missed}))
        else:
            issues.append(_issue("sampling_gap", "warning", now, f"Sensor missed {missed} expected reporting interval(s).", {"gap_minutes": gap, "missed_intervals": missed}))

    def _detect_missing_and_suspicious(self, issues: list[dict[str, Any]], row: pd.Series, channels: list[str], now: pd.Timestamp) -> None:
        for channel in channels:
            value = row.get(channel)
            numeric = _number(value)
            if numeric is None:
                issue_type = "missing_reading" if pd.isna(value) else "suspicious_value"
                severity = "warning" if issue_type == "missing_reading" else "critical"
                message = f"Supported channel {channel} is missing." if issue_type == "missing_reading" else f"Supported channel {channel} is non-finite."
                issues.append(_issue(issue_type, severity, now, message, {"value": None if pd.isna(value) else str(value)}, channel))
                continue
            low, high = PHYSICAL_BOUNDS[channel]
            if numeric < low or (high is not None and numeric > high):
                issues.append(_issue("suspicious_value", "critical", now, f"{channel}={numeric:g} is outside broad physical plausibility bounds.", {"value": numeric, "lower_bound": low, "upper_bound": high}, channel))

    def _continuous_values(self, data: pd.DataFrame, channel: str) -> pd.DataFrame:
        values = data[["timestamp", channel]].copy()
        values[channel] = pd.to_numeric(values[channel], errors="coerce")
        values[channel] = values[channel].replace([np.inf, -np.inf], np.nan)
        gaps = values.timestamp.diff().dt.total_seconds().div(60).gt(MINOR_GAP_MINUTES)
        segment = gaps.cumsum()
        return values.loc[segment.eq(segment.iloc[-1])].dropna(subset=[channel])

    def _detect_flatline(self, issues: list[dict[str, Any]], data: pd.DataFrame, channel: str, now: pd.Timestamp) -> None:
        segment = self._continuous_values(data, channel)
        if len(segment) < FLATLINE_MIN_SAMPLES:
            return
        adaptive = max(CHANNEL_RESOLUTION[channel] * 0.25, float(segment[channel].iloc[:-1].quantile(.75) - segment[channel].iloc[:-1].quantile(.25)) * .001)
        last_value = float(segment[channel].iloc[-1])
        start = len(segment) - 1
        while start > 0 and abs(float(segment[channel].iloc[start - 1]) - last_value) <= adaptive:
            start -= 1
        run = segment.iloc[start:]
        duration = (run.timestamp.iloc[-1] - run.timestamp.iloc[0]).total_seconds() / 60.0
        variation = float(run[channel].max() - run[channel].min())
        if len(run) >= FLATLINE_MIN_SAMPLES and duration >= FLATLINE_MIN_DURATION_MINUTES:
            issues.append(_issue("flatline", "warning", now, f"{channel} has remained effectively unchanged for {duration:.0f} minutes.", {"start_timestamp": run.timestamp.iloc[0].isoformat(), "duration_minutes": duration, "samples": len(run), "observed_variation": variation, "tolerance": adaptive}, channel))

    def _detect_jump(self, issues: list[dict[str, Any]], data: pd.DataFrame, channel: str, now: pd.Timestamp) -> None:
        values = self._continuous_values(data, channel)
        diffs = values[channel].diff().abs().dropna()
        if len(diffs) < JUMP_MIN_REFERENCE_DIFFS + 1:
            return
        current = float(diffs.iloc[-1])
        reference = diffs.iloc[:-1]
        median = float(reference.median())
        mad = _scaled_mad(reference)
        q1, q3 = reference.quantile([.25, .75])
        threshold = max(median + JUMP_MAD_MULTIPLIER * mad, float(q3 + JUMP_IQR_MULTIPLIER * (q3 - q1)), CHANNEL_RESOLUTION[channel])
        if current >= threshold and threshold > 0:
            issues.append(_issue("jump", "warning", now, f"{channel} changed by {current:g}, above its causal robust threshold {threshold:.2f}.", {"absolute_change": current, "threshold": threshold, "reference_differences": len(reference), "method": "max(median+8*MAD, Q3+3*IQR, resolution)"}, channel))

    def _detect_noise(self, issues: list[dict[str, Any]], data: pd.DataFrame, channel: str, now: pd.Timestamp) -> None:
        values = self._continuous_values(data, channel)[channel]
        diffs = values.diff().dropna()
        if len(diffs) < NOISE_MIN_REFERENCE_DIFFS + NOISE_WINDOW_SAMPLES:
            return
        recent = diffs.iloc[-NOISE_WINDOW_SAMPLES:]
        reference = diffs.iloc[:-NOISE_WINDOW_SAMPLES]
        recent_scale = _scaled_mad(recent)
        reference_scale = max(_scaled_mad(reference), CHANNEL_RESOLUTION[channel])
        signs = np.sign(recent.to_numpy())
        sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0))
        if recent_scale > NOISE_SCALE_MULTIPLIER * reference_scale and sign_changes >= NOISE_MIN_SIGN_CHANGES:
            issues.append(_issue("abnormal_noise", "warning", now, f"{channel} is rapidly oscillating above its historical variability.", {"recent_difference_mad": recent_scale, "reference_difference_mad": reference_scale, "variability_ratio": recent_scale / reference_scale, "sign_changes": sign_changes, "window_samples": NOISE_WINDOW_SAMPLES}, channel))

    def _detect_drift(self, issues: list[dict[str, Any]], data: pd.DataFrame, channel: str, now: pd.Timestamp) -> None:
        values = self._continuous_values(data, channel)[channel]
        required = DRIFT_RECENT_SAMPLES + DRIFT_MIN_BASELINE_SAMPLES
        if len(values) < required:
            return
        recent = values.iloc[-DRIFT_RECENT_SAMPLES:]
        baseline = values.iloc[:-DRIFT_RECENT_SAMPLES]
        baseline_median = float(baseline.median())
        scale = max(_scaled_mad(baseline), CHANNEL_RESOLUTION[channel])
        shift = float(recent.median() - baseline_median)
        same_side = float(((recent - baseline_median) * np.sign(shift) > 0).mean()) if shift else 0.0
        if abs(shift) > DRIFT_SCALE_MULTIPLIER * scale and same_side >= .8:
            issues.append(_issue("possible_drift", "warning", now, f"Possible Drift: {channel} shows a sustained {shift:+.2f} offset from its trailing baseline.", {"recent_samples": len(recent), "baseline_samples": len(baseline), "median_shift": shift, "robust_scale": scale, "same_side_fraction": same_side}, channel))

    def _detect_cross_sensor(self, issues: list[dict[str, Any]], data: pd.DataFrame, peers: pd.DataFrame, sensor_id: int, now: pd.Timestamp) -> None:
        peer_data = peers.copy()
        peer_data["timestamp"] = pd.to_datetime(peer_data["timestamp"], utc=True)
        peer_data = peer_data.loc[peer_data.timestamp.le(now) & peer_data.sensor_id.astype(int).isin({2, 4, 5})]
        pivot = peer_data.pivot_table(index="timestamp", columns="sensor_id", values="pm25", aggfunc="last").sort_index()
        if sensor_id not in pivot or len(pivot.columns) < 3:
            return
        others = [column for column in pivot.columns if int(column) != sensor_id]
        residual = pivot[sensor_id] - pivot[others].median(axis=1)
        residual = residual.dropna()
        if len(residual) < CROSS_SENSOR_MIN_REFERENCE_SAMPLES + CROSS_SENSOR_RECENT_SAMPLES:
            return
        recent = residual.iloc[-CROSS_SENSOR_RECENT_SAMPLES:]
        if residual.index[-1] != now:
            return
        reference = residual.iloc[:-CROSS_SENSOR_RECENT_SAMPLES]
        center = float(reference.median())
        scale = max(_scaled_mad(reference), CHANNEL_RESOLUTION["pm25"])
        deviations = recent - center
        direction = np.sign(float(deviations.median()))
        if direction and (deviations * direction > CROSS_SENSOR_SCALE_MULTIPLIER * scale).all():
            issues.append(_issue("cross_sensor_inconsistency", "warning", now, "PM2.5 differs persistently from both other Nablus sensors; this is supporting review evidence only.", {"recent_samples": len(recent), "median_residual": float(recent.median()), "reference_center": center, "robust_scale": scale, "peer_sensor_ids": [int(value) for value in others]}, "pm25"))
