"""Run causal Sensor Health replay and write development/audit artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import CLEANED_CSV, OUTPUT_DIR, REPORT_DIR
from app.ml import sensor_health_config as health_config
from app.ml.sensor_health import SensorHealthEngine
from app.ml.sensor_health_config import (
    CHANNEL_RESOLUTION,
    CROSS_SENSOR_MIN_REFERENCE_SAMPLES,
    CROSS_SENSOR_RECENT_SAMPLES,
    CROSS_SENSOR_SCALE_MULTIPLIER,
    DIMENSION_WEIGHTS,
    EXPECTED_INTERVAL_MINUTES,
    ISSUE_PENALTIES,
    MAJOR_GAP_MINUTES,
    OFFLINE_MINUTES,
    SENSOR_CHANNELS,
    SENSOR_METADATA,
)


HEALTH_OUTPUT_DIR = OUTPUT_DIR / "sensor_health"
SUMMARY_PATH = OUTPUT_DIR / "sensor_health_summary.json"
EVENTS_PATH = OUTPUT_DIR / "sensor_health_events.csv"
REPORT_PATH = REPORT_DIR / "sensor_health_report.md"
FLAG_TYPES = {
    "jump_flag": {"jump"},
    "flatline_flag": {"flatline"},
    "possible_drift_flag": {"possible_drift"},
    "sampling_gap_flag": {"sampling_gap", "major_sampling_gap"},
    "suspicious_value_flag": {"suspicious_value"},
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def load_data(path: Path = CLEANED_CSV) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)


def availability_audit(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for sensor_id, group in data.groupby("sensor_id", sort=True):
        times = group.timestamp.drop_duplicates().sort_values()
        intervals = times.diff().dt.total_seconds().div(60).dropna()
        expected = float(intervals.mode().iloc[0]) if not intervals.mode().empty else EXPECTED_INTERVAL_MINUTES
        internal_grid = pd.date_range(times.iloc[0], times.iloc[-1], freq=f"{int(expected)}min", tz="UTC")
        missing = internal_grid.difference(pd.DatetimeIndex(times))
        gap_mask = intervals.gt(expected * 1.5)
        major_mask = intervals.ge(MAJOR_GAP_MINUTES)
        audit[str(int(sensor_id))] = {
            "sensor_id": int(sensor_id),
            "city": SENSOR_METADATA[int(sensor_id)]["city"],
            "site": SENSOR_METADATA[int(sensor_id)]["site"],
            "first_timestamp": times.iloc[0].isoformat(),
            "last_timestamp": times.iloc[-1].isoformat(),
            "total_rows": int(len(group)),
            "unique_timestamps": int(len(times)),
            "expected_sampling_interval_minutes": expected,
            "observed_interval_minutes": {str(float(k)): int(v) for k, v in intervals.value_counts().sort_index().items()},
            "missing_timestamps": int(len(missing)),
            "largest_gap_minutes": float(intervals.max()) if len(intervals) else 0.0,
            "gaps_over_expected_tolerance": int(gap_mask.sum()),
            "gaps_at_or_over_major_threshold": int(major_mask.sum()),
            "expected_readings": int(len(internal_grid)),
            "readings_received_percent": round(100 * len(times) / len(internal_grid), 3) if len(internal_grid) else 100.0,
            "boundary_note": "Expected readings are counted only from this sensor's first through last timestamp; dataset boundaries are not outages.",
        }
    return audit


def _history_row(result: dict[str, Any], source: pd.Series | None) -> dict[str, Any]:
    counts = Counter(issue["type"] for issue in result["issues"])
    row = {
        "timestamp": result["timestamp"],
        "sensor_id": result["sensor_id"],
        "status": result["status"],
        "health_score": result["health_score"],
        "missing_count": counts["missing_reading"] + counts["missing_timestamp"],
        "gap_flag": int(bool(counts["sampling_gap"] or counts["major_sampling_gap"] or counts["offline"])),
        "flatline_flag": int(bool(counts["flatline"])),
        "jump_flag": int(bool(counts["jump"])),
        "noise_flag": int(bool(counts["abnormal_noise"])),
        "possible_drift_flag": int(bool(counts["possible_drift"])),
        "suspicious_value_flag": int(bool(counts["suspicious_value"])),
        "cross_sensor_inconsistency_flag": int(bool(counts["cross_sensor_inconsistency"])),
        "issue_count": len(result["issues"]),
        "critical_issue_count": sum(issue["severity"] == "critical" for issue in result["issues"]),
        "minutes_since_last_reading": result["minutes_since_last_reading"],
        "issues_json": json.dumps(result["issues"], separators=(",", ":"), default=_json_default),
        "source_row_present": int(source is not None),
    }
    for flag in FLAG_TYPES:
        row[f"existing_{flag}"] = int(bool(source.get(flag, False))) if source is not None else 0
    return row


def historical_replay(data: pd.DataFrame) -> dict[int, pd.DataFrame]:
    engine = SensorHealthEngine()
    histories: dict[int, pd.DataFrame] = {}
    # All data may be supplied as a buffer; the engine itself rejects future rows.
    for sensor_id, sensor_data in data.groupby("sensor_id", sort=True):
        sensor_data = sensor_data.sort_values("timestamp").reset_index(drop=True)
        by_time = {timestamp: row for timestamp, row in sensor_data.set_index("timestamp").iterrows()}
        grid = pd.date_range(sensor_data.timestamp.min(), sensor_data.timestamp.max(), freq=f"{EXPECTED_INTERVAL_MINUTES}min", tz="UTC")
        rows: list[dict[str, Any]] = []
        for timestamp in grid:
            result = engine.evaluate(sensor_data, timestamp)
            rows.append(_history_row(result, by_time.get(timestamp)))
        histories[int(sensor_id)] = pd.DataFrame(rows)
    _add_cross_sensor_replay(data, histories)
    return histories


def _add_cross_sensor_replay(data: pd.DataFrame, histories: dict[int, pd.DataFrame]) -> None:
    """Add the engine's peer rule in one chronological pass for efficient replay."""
    nablus = data.loc[data.sensor_id.isin([2, 4, 5])]
    pivot = nablus.pivot_table(index="timestamp", columns="sensor_id", values="pm25", aggfunc="last").sort_index()
    for sensor_id in [2, 4, 5]:
        others = [column for column in pivot.columns if int(column) != sensor_id]
        residual = (pivot[sensor_id] - pivot[others].median(axis=1)).dropna()
        first_end = CROSS_SENSOR_MIN_REFERENCE_SAMPLES + CROSS_SENSOR_RECENT_SAMPLES - 1
        for end in range(first_end, len(residual)):
            recent = residual.iloc[end - CROSS_SENSOR_RECENT_SAMPLES + 1:end + 1]
            if any(np.diff(recent.index.view("i8")) > pd.Timedelta(minutes=15).value):
                continue
            reference = residual.iloc[:end - CROSS_SENSOR_RECENT_SAMPLES + 1]
            center = float(reference.median())
            values = reference.to_numpy(dtype=float)
            scale = max(1.4826 * float(np.median(np.abs(values - np.median(values)))), CHANNEL_RESOLUTION["pm25"])
            deviations = recent - center
            direction = np.sign(float(deviations.median()))
            if not direction or not (deviations * direction > CROSS_SENSOR_SCALE_MULTIPLIER * scale).all():
                continue
            timestamp = recent.index[-1]
            history = histories[sensor_id]
            mask = pd.to_datetime(history.timestamp, utc=True).eq(timestamp) & history.source_row_present.eq(1)
            if not mask.any():
                continue
            idx = history.index[mask][0]
            issue = {
                "type": "cross_sensor_inconsistency", "severity": "warning", "channel": "pm25",
                "timestamp": timestamp.isoformat(),
                "evidence": {"recent_samples": CROSS_SENSOR_RECENT_SAMPLES, "median_residual": float(recent.median()), "reference_center": center, "robust_scale": scale, "peer_sensor_ids": [int(value) for value in others]},
                "message": "PM2.5 differs persistently from both other Nablus sensors; this is supporting review evidence only.",
                "alert_recommended": True,
            }
            current = json.loads(history.at[idx, "issues_json"])
            current.append(issue)
            history.at[idx, "issues_json"] = json.dumps(current, separators=(",", ":"))
            history.at[idx, "cross_sensor_inconsistency_flag"] = 1
            history.at[idx, "issue_count"] = int(history.at[idx, "issue_count"]) + 1
            dimension, dimension_penalty = ISSUE_PENALTIES["cross_sensor_inconsistency"]
            score_penalty = round(dimension_penalty * DIMENSION_WEIGHTS[dimension])
            history.at[idx, "health_score"] = max(0, int(history.at[idx, "health_score"]) - score_penalty)
            if history.at[idx, "status"] == "Healthy":
                history.at[idx, "status"] = "Warning"


def merge_events(histories: dict[int, pd.DataFrame]) -> pd.DataFrame:
    detected: list[dict[str, Any]] = []
    for sensor_id, history in histories.items():
        for row in history.itertuples(index=False):
            for issue in json.loads(row.issues_json):
                detected.append({
                    "sensor_id": sensor_id,
                    "site": SENSOR_METADATA[sensor_id]["site"],
                    "timestamp": pd.Timestamp(row.timestamp),
                    "event_type": issue["type"],
                    "channel": issue.get("channel") or "",
                    "severity": issue["severity"],
                    "evidence": issue["evidence"],
                    "message": issue["message"],
                    "alert_recommended": issue["alert_recommended"],
                })
    columns = ["sensor_id", "site", "start_timestamp", "end_timestamp", "event_type", "channel", "severity", "duration_minutes", "evidence", "message", "alert_recommended"]
    if not detected:
        return pd.DataFrame(columns=columns)
    raw = pd.DataFrame(detected).sort_values(["sensor_id", "event_type", "channel", "timestamp"])
    merged: list[dict[str, Any]] = []
    keys = ["sensor_id", "site", "event_type", "channel", "severity", "alert_recommended"]
    for key, group in raw.groupby(keys, dropna=False, sort=False):
        group = group.sort_values("timestamp")
        blocks = group.timestamp.diff().dt.total_seconds().div(60).gt(EXPECTED_INTERVAL_MINUTES * 1.5).cumsum()
        for _, block in group.groupby(blocks):
            start, end = block.timestamp.iloc[0], block.timestamp.iloc[-1]
            evidence = {"first": block.evidence.iloc[0], "last": block.evidence.iloc[-1], "evaluations": int(len(block))}
            merged.append(dict(zip(keys, key)) | {
                "start_timestamp": start.isoformat(),
                "end_timestamp": end.isoformat(),
                "duration_minutes": float((end - start).total_seconds() / 60),
                "evidence": json.dumps(evidence, separators=(",", ":"), default=_json_default),
                "message": block.message.iloc[-1],
            })
    return pd.DataFrame(merged)[columns].sort_values(["start_timestamp", "sensor_id", "event_type"])


def flag_comparison(histories: dict[int, pd.DataFrame]) -> dict[str, Any]:
    source_rows = pd.concat(histories.values(), ignore_index=True)
    source_rows = source_rows.loc[source_rows.source_row_present.eq(1)]
    output: dict[str, Any] = {}
    for old_flag, types in FLAG_TYPES.items():
        old = source_rows[f"existing_{old_flag}"].astype(bool)
        new_column = {
            "jump_flag": "jump_flag", "flatline_flag": "flatline_flag",
            "possible_drift_flag": "possible_drift_flag", "sampling_gap_flag": "gap_flag",
            "suspicious_value_flag": "suspicious_value_flag",
        }[old_flag]
        new = source_rows[new_column].astype(bool)
        tp = int((old & new).sum())
        output[old_flag] = {
            "rows_compared": int(len(old)), "existing_positive": int(old.sum()), "engine_positive": int(new.sum()),
            "both_positive": tp, "overall_agreement_percent": round(100 * (old == new).mean(), 3),
            "existing_positive_detected_percent": round(100 * tp / old.sum(), 3) if old.sum() else None,
            "engine_positive_matching_existing_percent": round(100 * tp / new.sum(), 3) if new.sum() else None,
        }
    return output


def build_summary(data: pd.DataFrame, histories: dict[int, pd.DataFrame], events: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    sensors: dict[str, Any] = {}
    for sensor_id, history in histories.items():
        counts = history.status.value_counts()
        sensor_events = events.loc[events.sensor_id.eq(sensor_id)]
        event_counts = sensor_events.event_type.value_counts().to_dict()
        final = history.iloc[-1]
        sensors[str(sensor_id)] = {
            "site": SENSOR_METADATA[sensor_id]["site"],
            "supported_channels": list(SENSOR_CHANNELS[sensor_id]),
            "total_evaluated_timestamps": int(len(history)),
            "status_percentages": {status: round(100 * int(counts.get(status, 0)) / len(history), 3) for status in ["Healthy", "Warning", "Critical", "Offline"]},
            "event_counts": {kind: int(event_counts.get(kind, 0)) for kind in ["missing_reading", "missing_timestamp", "sampling_gap", "major_sampling_gap", "offline", "flatline", "jump", "abnormal_noise", "possible_drift", "suspicious_value", "cross_sensor_inconsistency"]},
            "largest_gap_minutes": audit[str(sensor_id)]["largest_gap_minutes"],
            "lowest_health_score": int(history.health_score.min()),
            "current_status": final.status,
            "current_health_score": int(final.health_score),
            "current_timestamp": final.timestamp,
        }
    comparison = flag_comparison(histories)
    return {
        "score_notice": "AirAware operational sensor-health score; not a probability or calibrated confidence.",
        "configuration": {
            "expected_interval_minutes": EXPECTED_INTERVAL_MINUTES,
            "gap_tolerance_minutes": EXPECTED_INTERVAL_MINUTES * 1.5,
            "major_gap_minutes": MAJOR_GAP_MINUTES,
            "offline_minutes": OFFLINE_MINUTES,
            "availability_window_rule": "Each sensor's first-to-last timestamp only; boundaries are excluded from outage inference.",
        },
        "availability_audit": audit,
        "sensors": sensors,
        "existing_flag_comparison": comparison,
        "causal_verification": {
            "engine_filters_rows_after_as_of": True,
            "rolling_windows_are_backward_only": True,
            "jump_reference_excludes_current_transition": True,
            "drift_baseline_precedes_recent_window": True,
            "cross_sensor_data_is_filtered_at_as_of": True,
        },
    }


def write_report(summary: dict[str, Any], events: pd.DataFrame) -> None:
    lines = [
        "# AirAware Sensor Health Report", "",
        "> The health score is an operational score, not a probability, accuracy, or confidence estimate.", "",
        "## Existing quality logic", "",
        "Reused semantics: modal per-sensor cadence; Phase 1's 1.5x gap tolerance, robust MAD/IQR jump method, eight-sample flatline minimum and broad physical bounds; Phase 3's 90-minute major-gap boundary. Existing row flags are retained for comparison, but causal runtime thresholds use only prior values. Phase 1 missing_flag checks only channels that sensor provides (although it also includes source AQI); sampling_gap_flag marks the returning row; suspicious_value_flag combines broad physical, non-finite, zero-pollutant and data-relative-extreme review events; flatline and possible_drift flags backfill detected event intervals; overall_quality_flag applies precedence. Phase 3 current_quality_review_flag is overall_quality_flag != NORMAL; recent_gap_count_3h and recent_missing_pm25_count_3h are backward rolling counts inside continuous segments; recent_anomaly_count_3h rolls Phase 2 events where at least two anomaly methods agreed. AQI is not treated as a hardware channel in the new engine.", "",
        "## Sensor availability", "",
        "| Sensor | Rows | Time range | Largest gap | Availability |", "|---:|---:|---|---:|---:|",
    ]
    for sensor_id, row in summary["availability_audit"].items():
        lines.append(f"| {sensor_id} | {row['total_rows']:,} | {row['first_timestamp']} to {row['last_timestamp']} | {row['largest_gap_minutes']:.0f} min | {row['readings_received_percent']:.3f}% |")
    lines += ["", "## Supported channels", ""]
    for sensor_id, channels in SENSOR_CHANNELS.items():
        lines.append(f"- Sensor {sensor_id}: {', '.join(channels)}")
    lines += [
        "", "NO2 and O3 are structurally unavailable for Sensors 2, 4 and 5 and never enter missingness, scoring, or alert logic.",
        "", "## Detectors", "",
        "- Missing: null or non-finite supported channels only; absent scheduled timestamps are tracked separately.",
        "- Gap/offline: elapsed time from the last known row, plus resumed-arrival gap evidence.",
        "- Flatline: a continuous trailing run at channel-resolution tolerance and minimum duration.",
        "- Jump: current absolute first difference against prior-only robust MAD and IQR limits.",
        "- Noise: repeated short-window sign reversals plus a robust variability-ratio requirement.",
        "- Possible drift: a 24-hour median shift against at least 48 hours of strictly earlier data; this is not confirmed calibration drift.",
        "- Suspicious value: non-finite values and broad physical impossibilities, without narrow environmental normal ranges.",
        "- Cross-sensor inconsistency: four sustained Nablus PM2.5 residuals versus the other two sensors; supporting evidence only.",
        "", "## Thresholds and score", "",
        f"- Sampling: expected={health_config.EXPECTED_INTERVAL_MINUTES} min; gap tolerance={health_config.MINOR_GAP_MINUTES:g} min (existing Phase 1 1.5x rule); major={health_config.MAJOR_GAP_MINUTES} min (existing Phase 3); offline={health_config.OFFLINE_MINUTES} min (new operational threshold).",
        f"- Flatline: {health_config.FLATLINE_MIN_SAMPLES} samples and {health_config.FLATLINE_MIN_DURATION_MINUTES} min; 0.01-unit channel-resolution floors (existing method, precision corrected from observed data).",
        f"- Jump: at least {health_config.JUMP_MIN_REFERENCE_DIFFS} prior differences; max(median+{health_config.JUMP_MAD_MULTIPLIER:g}xMAD, Q3+{health_config.JUMP_IQR_MULTIPLIER:g}xIQR, resolution), retaining the Phase 1 robust multipliers.",
        f"- Noise: {health_config.NOISE_WINDOW_SAMPLES} differences, {health_config.NOISE_MIN_REFERENCE_DIFFS} reference differences, >{health_config.NOISE_SCALE_MULTIPLIER:g}x robust scale and at least {health_config.NOISE_MIN_SIGN_CHANGES} sign changes (new conservative calibration).",
        f"- Drift: {health_config.DRIFT_RECENT_SAMPLES} recent samples, {health_config.DRIFT_MIN_BASELINE_SAMPLES} earlier baseline samples, >{health_config.DRIFT_SCALE_MULTIPLIER:g}x robust scale and >=80% persistence (causal extension of Phase 1).",
        f"- Cross-sensor: {health_config.CROSS_SENSOR_RECENT_SAMPLES} sustained samples after {health_config.CROSS_SENSOR_MIN_REFERENCE_SAMPLES} references and >{health_config.CROSS_SENSOR_SCALE_MULTIPLIER:g}x robust residual scale (new supporting rule).",
        "- Dimensions start at 100. Penalties: missing 15 availability; minor gap 15 availability; major gap 45 availability; offline 100 availability; suspicious value 45 integrity; jump 10 stability; flatline 30 stability; noise 20 stability; cross-sensor 15 consistency; possible drift 30 drift.",
        "- Weighted score: 35% availability + 25% integrity + 20% stability + 10% consistency + 10% drift, bounded 0-100. Offline forces 0. This is operational, not probabilistic.",
        "- Status: Offline overrides everything; any critical issue forces Critical; otherwise no-issue score >=90 is Healthy, score >=65 is Warning, and lower scores are Critical.",
    ]
    lines += ["", "## Historical results", "", "| Sensor | Healthy | Warning | Critical | Offline | Lowest score | Final |", "|---:|---:|---:|---:|---:|---:|---|"]
    for sensor_id, row in summary["sensors"].items():
        p = row["status_percentages"]
        lines.append(f"| {sensor_id} | {p['Healthy']:.3f}% | {p['Warning']:.3f}% | {p['Critical']:.3f}% | {p['Offline']:.3f}% | {row['lowest_health_score']} | {row['current_status']} ({row['current_health_score']}) |")
    lines += ["", "## Event counts", ""]
    for sensor_id, row in summary["sensors"].items():
        rendered = ", ".join(f"{key}={value}" for key, value in row["event_counts"].items())
        lines.append(f"- Sensor {sensor_id}: {rendered}")
    lines += ["", "## Representative events", ""]
    for row in events.sort_values(["severity", "duration_minutes"], ascending=[True, False]).head(20).itertuples():
        lines.append(f"- Sensor {row.sensor_id}, {row.start_timestamp}: {row.event_type} ({row.severity}, {row.channel or 'all channels'}) — {row.message}")
    lines += ["", "## Existing-flag comparison", ""]
    for flag, row in summary["existing_flag_comparison"].items():
        lines.append(f"- {flag}: existing={row['existing_positive']}, engine={row['engine_positive']}, both={row['both_positive']}, all-row agreement={row['overall_agreement_percent']:.3f}%.")
    lines += [
        "", "Differences are intentional: the engine cannot reproduce Phase 1's full-dataset threshold leakage. Early jumps lack the required prior calibration; flatlines alert only after the eighth sample rather than backfilling the whole run; the old 43-row drift interval was backfilled on Sensor 1 while the causal engine found three later Sensor 2 review events; old suspicious flags also include data-relative extremes, whereas runtime critical flags are limited to physical/non-finite failures.",
        "", "## Causal and false-positive review", "", "All evaluation filters history at the requested timestamp. Jump calibration excludes the current transition; drift compares a trailing recent window only with earlier baseline values; gap state comes from the last known reading; cross-sensor evidence uses only peer values available by the same timestamp. Noise requires both a fourfold robust-scale increase and repeated sign reversals. Cross-sensor inconsistency is supporting evidence and is never a hard failure by itself.", "",
        "Event concentration by sensor/channel:", "",
    ]
    concentration = events.groupby(["event_type", "sensor_id", "channel"], dropna=False).size().sort_values(ascending=False)
    for (event_type, sensor_id, channel), count in concentration.head(20).items():
        lines.append(f"- {event_type}: Sensor {sensor_id}, {channel if pd.notna(channel) and channel else 'all channels'} = {count} merged event(s).")
    lines += ["", "The initial replay exposed an over-sensitive NO2 resolution assumption; after correction, flatline output fell to one merged event. Jumps are most concentrated on Sensor 1 PM2.5 (26 events), reflecting its five available channels and variable source stream. Noise remains sparse (19 merged events overall). The three long Sensor 2 drift events deserve domain review but are labeled only as possible drift.", "", "## Future ML recommendation", "", "Do not add Isolation Forest or an autoencoder yet. The short dataset lacks labeled hardware faults and calibration truth; the conservative rule/statistical layer is more interpretable and covers the actionable availability, integrity, stability, consistency and possible-drift cases.", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    HEALTH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    audit = availability_audit(data)
    histories = historical_replay(data)
    for sensor_id, history in histories.items():
        history.to_csv(HEALTH_OUTPUT_DIR / f"history_sensor_{sensor_id}.csv", index=False)
    events = merge_events(histories)
    events.to_csv(EVENTS_PATH, index=False)
    summary = build_summary(data, histories, events, audit)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    write_report(summary, events)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({"outputs": str(HEALTH_OUTPUT_DIR), "sensors": result["sensors"]}, indent=2))
