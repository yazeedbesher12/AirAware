from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.dataset import load_phase4_dataset
from app.ml.realtime_features import MultiSensorHistoryBuffer, generate_realtime_features
from app.ml.realtime_inference import SavedModelPredictor


BACKEND = Path(__file__).resolve().parents[1]


def test_chronological_splits_and_no_target_features():
    dataset = load_phase4_dataset(BACKEND / "outputs")
    train, validation, test = dataset.splits("1h")
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert not any(name.startswith(("target_", "pollution_event_", "forecast_sample_valid_")) for name in dataset.features)


def test_realtime_buffer_is_sensor_independent_and_resets_major_gap():
    buffer = MultiSensorHistoryBuffer()
    first, reset = buffer.update({"timestamp": "2026-08-01T00:00:00Z", "sensor_id": 1, "pm25": 10})
    assert reset and len(first) == 1
    other, other_reset = buffer.update({"timestamp": "2026-08-01T00:00:00Z", "sensor_id": 2, "pm25": 20})
    assert other_reset and len(other) == 1 and buffer.reading_count(1) == 1
    after_gap, reset = buffer.update({"timestamp": "2026-08-01T03:00:00Z", "sensor_id": 1, "pm25": 12})
    assert reset and len(after_gap) == 1


def test_realtime_feature_generator_matches_phase3_training_features():
    cleaned = pd.read_csv(BACKEND / "data/cleaned/AirAware_Tulkarem_Nablus_cleaned.csv")
    reference = pd.read_csv(BACKEND / "outputs/AirAware_ML_Features.csv")
    cleaned["timestamp"] = pd.to_datetime(cleaned.timestamp, utc=True); reference["timestamp"] = pd.to_datetime(reference.timestamp, utc=True)
    moment = pd.Timestamp("2026-08-20T12:00:00Z")
    history = cleaned[(cleaned.sensor_id == 1) & (cleaned.timestamp <= moment) & (cleaned.timestamp >= moment - pd.Timedelta(hours=30))]
    generated = generate_realtime_features(history).iloc[-1]
    expected = reference[(reference.sensor_id == 1) & (reference.timestamp == moment)].iloc[0]
    for feature in ("pm25_lag_1h", "pm25_mean_3h", "pm25_std_6h", "humidity_mean_3h", "pm25_who_24h_average"):
        assert np.isclose(float(generated[feature]), float(expected[feature]), equal_nan=True, atol=1e-8)


def test_saved_random_forest_reloads_without_retraining():
    dataset = load_phase4_dataset(BACKEND / "outputs")
    _, _, test = dataset.splits("1h")
    predictor = SavedModelPredictor(BACKEND, "random_forest", "1h")
    concentration, health, latency = predictor.predict(test.iloc[[0]])
    assert np.isfinite(concentration) and np.isfinite(health) and latency >= 0
