"""Stateful lifecycle engine for normalized AirAware forecast/health results."""

from __future__ import annotations

import copy
import hashlib
from datetime import timedelta
from typing import Any

import pandas as pd

from app.ml.alert_config import (
    CATEGORY_BY_TYPE,
    DEFAULT_COOLDOWN_MINUTES,
    HEALTH_SEVERITY_MAP,
    HIGH_PM_POLICY,
    PRIORITY_RANK,
    SEVERITY_RANK,
    SOURCE_BY_TYPE,
    STATIC_POLICY,
)
from app.ml.sensor_health_config import SENSOR_METADATA


AlertState = dict[str, Any]


def empty_alert_state() -> AlertState:
    return {"alerts": [], "deduplicated_repeats_suppressed": 0, "escalations": 0}


def _time(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _identifier(dedup_key: str, created_at: str) -> str:
    digest = hashlib.sha256(f"{dedup_key}|{created_at}".encode("utf-8")).hexdigest()[:16]
    return f"alert_{digest}"


def _dedup(sensor_id: int, alert_type: str, horizon: str | None = None, channel: str | None = None) -> str:
    values = [str(sensor_id), alert_type]
    if horizon:
        values.append(horizon)
    if channel:
        values.append(channel)
    return ":".join(values)


class AlertEngine:
    """Create, update, escalate, and resolve deterministic alert events.

    The engine consumes already-normalized results. It never loads a model,
    dataset, API, or persistence layer.
    """

    def __init__(self, cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES):
        self.cooldown = timedelta(minutes=cooldown_minutes)

    def evaluate(
        self,
        sensor_id: int,
        timestamp: Any,
        forecasts: dict[str, dict[str, Any]] | None = None,
        sensor_health: dict[str, Any] | None = None,
        previous_alert_state: AlertState | None = None,
    ) -> dict[str, Any]:
        if sensor_id not in SENSOR_METADATA:
            raise ValueError(f"Unsupported sensor_id: {sensor_id}")
        now = _time(timestamp)
        now_text = now.isoformat()
        state = copy.deepcopy(previous_alert_state or empty_alert_state())
        state.setdefault("alerts", [])
        state.setdefault("deduplicated_repeats_suppressed", 0)
        state.setdefault("escalations", 0)
        conditions = self._conditions(sensor_id, now, forecasts, sensor_health)
        condition_by_key = {item["dedup_key"]: item for item in conditions}
        transitions: list[dict[str, Any]] = []

        supplied_keys: set[str] = set()
        if forecasts is not None:
            for horizon in forecasts:
                supplied_keys.add(_dedup(sensor_id, "high_pm_risk", horizon=horizon, channel="pm25"))
                supplied_keys.add(_dedup(sensor_id, "who_exceedance", horizon=horizon, channel="pm25"))
        if sensor_health is not None:
            supplied_keys.add(_dedup(sensor_id, "sensor_offline"))
            supplied_keys.update(
                alert["dedup_key"] for alert in state["alerts"]
                if alert["sensor_id"] == sensor_id and alert["status"] == "active"
                and alert["type"] in {"possible_drift", "suspicious_reading"}
            )
            supplied_keys.update(
                item["dedup_key"] for item in conditions
                if item["type"] in {"possible_drift", "suspicious_reading"}
            )

        for key in supplied_keys - condition_by_key.keys():
            active = self._active_for(state, key)
            if active is None:
                continue
            active["status"] = "resolved"
            active["updated_at"] = now_text
            active["resolved_at"] = now_text
            active["notification_recommended"] = False
            active["lifecycle_transition"] = "resolved"
            transitions.append(copy.deepcopy(active))

        for condition in conditions:
            active = self._active_for(state, condition["dedup_key"])
            if active is None:
                alert = self._new_alert(condition, now_text)
                state["alerts"].append(alert)
                transitions.append(copy.deepcopy(alert))
                continue

            severity_increased = SEVERITY_RANK[condition["severity"]] > SEVERITY_RANK[active["severity"]]
            priority_increased = PRIORITY_RANK[condition["priority"]] > PRIORITY_RANK[active["priority"]]
            last_notified = _time(active["last_notified_at"])
            cooldown_elapsed = now - last_notified >= self.cooldown
            notify = severity_increased or priority_increased or cooldown_elapsed
            active.update({key: value for key, value in condition.items() if key not in {"created_at", "id"}})
            active["updated_at"] = now_text
            active["notification_recommended"] = notify
            active["lifecycle_transition"] = "escalated" if severity_increased or priority_increased else "updated"
            if notify:
                active["last_notified_at"] = now_text
            else:
                state["deduplicated_repeats_suppressed"] += 1
            if severity_increased or priority_increased:
                state["escalations"] += 1
                transitions.append(copy.deepcopy(active))

        return {
            "state": state,
            "transitions": transitions,
            "active_alerts": [copy.deepcopy(alert) for alert in state["alerts"] if alert["status"] == "active"],
            "resolved_alerts": [copy.deepcopy(alert) for alert in state["alerts"] if alert["status"] == "resolved"],
        }

    @staticmethod
    def _active_for(state: AlertState, dedup_key: str) -> dict[str, Any] | None:
        return next((alert for alert in reversed(state["alerts"]) if alert["dedup_key"] == dedup_key and alert["status"] == "active"), None)

    @staticmethod
    def _new_alert(condition: dict[str, Any], now_text: str) -> dict[str, Any]:
        alert = {
            **condition,
            "id": _identifier(condition["dedup_key"], now_text),
            "status": "active",
            "created_at": now_text,
            "updated_at": now_text,
            "resolved_at": None,
            "last_notified_at": now_text,
            "notification_recommended": True,
            "alert_recommended": True,
            "lifecycle_transition": "new",
        }
        return alert

    def _conditions(
        self,
        sensor_id: int,
        now: pd.Timestamp,
        forecasts: dict[str, dict[str, Any]] | None,
        health: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        metadata = SENSOR_METADATA[sensor_id]
        base = {"sensor_id": sensor_id, "site": metadata["site"], "city": metadata["city"]}
        conditions: list[dict[str, Any]] = []
        if forecasts is not None:
            for horizon, forecast in forecasts.items():
                risk = forecast.get("high_pm_risk", {}).get("level")
                if risk in HIGH_PM_POLICY and horizon in HIGH_PM_POLICY[risk]:
                    severity, priority = HIGH_PM_POLICY[risk][horizon]
                    conditions.append(self._condition(base, "high_pm_risk", severity, priority, horizon, "pm25", f"{risk} PM2.5 risk expected", f"{risk} PM2.5 risk is forecast in {horizon}.", {
                        "forecast_pm25": forecast.get("pm25", {}).get("forecast"),
                        "safety_upper_estimate": forecast.get("pm25", {}).get("safety_upper_estimate"),
                        "safety_margin": forecast.get("pm25", {}).get("safety_margin"),
                        "risk_level": risk, "horizon": horizon,
                    }, risk))
                who = forecast.get("who", {})
                if who.get("status") == "Above WHO Guideline":
                    severity, priority = STATIC_POLICY["who_exceedance"]
                    conditions.append(self._condition(base, "who_exceedance", severity, priority, horizon, "pm25", "WHO PM2.5 24h guideline exceedance", "Predicted PM2.5 24-hour average is above the WHO guideline.", {
                        "predicted_24h_average": forecast.get("future_24h_average", {}).get("forecast"),
                        "guideline": who.get("guideline"),
                        "ratio_to_guideline": who.get("ratio_to_guideline"),
                        "horizon": horizon,
                    }, who.get("status")))

        if health is not None:
            if health.get("status") == "Offline":
                severity, priority = STATIC_POLICY["sensor_offline"]
                conditions.append(self._condition(base, "sensor_offline", severity, priority, None, None, "Sensor Offline", f"Sensor {sensor_id} is offline.", {
                    "last_reading_at": health.get("last_reading_at"),
                    "minutes_since_last_reading": health.get("minutes_since_last_reading"),
                    "site": metadata["site"],
                }, "Offline"))
            for issue in health.get("issues", []):
                issue_type = issue.get("type")
                channel = issue.get("channel")
                if issue_type == "possible_drift":
                    severity, priority = STATIC_POLICY["possible_drift"]
                    conditions.append(self._condition(base, "possible_drift", severity, priority, None, channel, "Possible Sensor Drift", issue.get("message") or f"Possible Drift detected for {channel or 'sensor data'}.", issue.get("evidence", {}), "possible_drift"))
                elif issue_type == "suspicious_value":
                    severity, priority = HEALTH_SEVERITY_MAP.get(issue.get("severity", "warning"), HEALTH_SEVERITY_MAP["warning"])
                    label = channel.upper() if channel else "sensor"
                    conditions.append(self._condition(base, "suspicious_reading", severity, priority, None, channel, "Suspicious Sensor Reading", f"Suspicious {label} reading detected.", {**issue.get("evidence", {}), "reason": issue.get("message")}, issue.get("severity")))
        return conditions

    @staticmethod
    def _condition(
        base: dict[str, Any], alert_type: str, severity: str, priority: str,
        horizon: str | None, channel: str | None, title: str, message: str,
        evidence: dict[str, Any], condition_level: Any,
    ) -> dict[str, Any]:
        return {
            **base,
            "dedup_key": _dedup(base["sensor_id"], alert_type, horizon, channel),
            "type": alert_type,
            "category": CATEGORY_BY_TYPE[alert_type],
            "severity": severity,
            "priority": priority,
            "title": title,
            "message": message,
            "horizon": horizon,
            "channel": channel,
            "source": SOURCE_BY_TYPE[alert_type],
            "evidence": evidence,
            "condition_level": condition_level,
            "is_early_warning": CATEGORY_BY_TYPE[alert_type] in {"forecast", "health_guideline"},
        }
