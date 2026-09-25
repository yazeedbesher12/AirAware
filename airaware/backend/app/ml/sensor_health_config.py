"""Central, operational configuration for AirAware sensor health.

The score produced from these settings is an explainable operational score.  It
is not a probability, accuracy estimate, or calibrated confidence measure.
"""

from __future__ import annotations


EXPECTED_INTERVAL_MINUTES = 15
GAP_TOLERANCE_MULTIPLIER = 1.5  # Reuses the Phase 1 cadence rule.
MINOR_GAP_MINUTES = EXPECTED_INTERVAL_MINUTES * GAP_TOLERANCE_MULTIPLIER
MAJOR_GAP_MINUTES = 90  # Reuses the Phase 3 continuous-segment boundary.
OFFLINE_MINUTES = 180  # New: twelve missed 15-minute reports.

SENSOR_CHANNELS: dict[int, tuple[str, ...]] = {
    1: ("pm25", "temperature", "humidity", "no2", "o3"),
    2: ("pm25", "temperature", "humidity"),
    4: ("pm25", "temperature", "humidity"),
    5: ("pm25", "temperature", "humidity"),
}

SENSOR_METADATA = {
    1: {"city": "Tulkarem", "site": "Tulkarem Municipality"},
    2: {"city": "Nablus", "site": "ANNU New Campus"},
    4: {"city": "Nablus", "site": "ANNU Old Campus"},
    5: {"city": "Nablus", "site": "Hisham Hijjawi College of Technology"},
}

# Broad physical plausibility only.  Environmentally unusual but possible high
# pollutant values are deliberately not rejected.
PHYSICAL_BOUNDS = {
    "pm25": (0.0, None),
    "temperature": (-50.0, 70.0),
    "humidity": (0.0, 100.0),
    "no2": (0.0, None),
    "o3": (0.0, None),
}

# Resolution-aware flatline floors; the adaptive tolerance can be larger.
CHANNEL_RESOLUTION = {
    "pm25": 0.01,
    "temperature": 0.01,
    "humidity": 0.01,
    "no2": 0.01,
    "o3": 0.01,
}

FLATLINE_MIN_SAMPLES = 8  # Reuses Phase 1's minimum at the 15-minute cadence.
FLATLINE_MIN_DURATION_MINUTES = 105
JUMP_MIN_REFERENCE_DIFFS = 24
JUMP_MAD_MULTIPLIER = 8.0  # Reuses Phase 1.
JUMP_IQR_MULTIPLIER = 3.0  # Reuses Phase 1.
NOISE_WINDOW_SAMPLES = 8
NOISE_MIN_REFERENCE_DIFFS = 48
NOISE_SCALE_MULTIPLIER = 4.0
NOISE_MIN_SIGN_CHANGES = 5
DRIFT_RECENT_SAMPLES = 96  # 24 hours at nominal cadence.
DRIFT_MIN_BASELINE_SAMPLES = 192
DRIFT_SCALE_MULTIPLIER = 3.0
CROSS_SENSOR_RECENT_SAMPLES = 4
CROSS_SENSOR_MIN_REFERENCE_SAMPLES = 48
CROSS_SENSOR_SCALE_MULTIPLIER = 4.0

DIMENSION_WEIGHTS = {
    "availability": 0.35,
    "integrity": 0.25,
    "stability": 0.20,
    "consistency": 0.10,
    "drift": 0.10,
}

# Penalties are applied to the named dimension and capped at zero.
ISSUE_PENALTIES = {
    "missing_reading": ("availability", 15),
    "missing_timestamp": ("availability", 15),
    "sampling_gap": ("availability", 15),
    "major_sampling_gap": ("availability", 45),
    "offline": ("availability", 100),
    "suspicious_value": ("integrity", 45),
    "jump": ("stability", 10),
    "flatline": ("stability", 30),
    "abnormal_noise": ("stability", 20),
    "cross_sensor_inconsistency": ("consistency", 15),
    "possible_drift": ("drift", 30),
}

HEALTHY_SCORE_MIN = 90
WARNING_SCORE_MIN = 65

ALERT_RECOMMENDED = {
    "info": False,
    "warning": True,
    "critical": True,
}
