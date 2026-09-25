"""Central configuration for AirAware alerts and notification policy."""

from __future__ import annotations


ALERT_TYPES = (
    "high_pm_risk",
    "who_exceedance",
    "sensor_offline",
    "possible_drift",
    "suspicious_reading",
)

DEFAULT_COOLDOWN_MINUTES = 60

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}
PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

HIGH_PM_POLICY = {
    "Elevated": {
        "1h": ("warning", "high"),
        "3h": ("warning", "medium"),
        "6h": ("warning", "medium"),
    },
    "High": {
        "1h": ("critical", "urgent"),
        "3h": ("warning", "high"),
        "6h": ("warning", "high"),
    },
}

STATIC_POLICY = {
    "who_exceedance": ("warning", "medium"),
    "sensor_offline": ("critical", "urgent"),
    "possible_drift": ("warning", "high"),
}

HEALTH_SEVERITY_MAP = {
    "info": ("info", "low"),
    "warning": ("warning", "high"),
    "critical": ("critical", "urgent"),
}

SOURCE_BY_TYPE = {
    "high_pm_risk": "forecast_engine",
    "who_exceedance": "forecast_engine",
    "sensor_offline": "sensor_health_engine",
    "possible_drift": "sensor_health_engine",
    "suspicious_reading": "sensor_health_engine",
}

CATEGORY_BY_TYPE = {
    "high_pm_risk": "forecast",
    "who_exceedance": "health_guideline",
    "sensor_offline": "sensor_health",
    "possible_drift": "sensor_health",
    "suspicious_reading": "sensor_health",
}
