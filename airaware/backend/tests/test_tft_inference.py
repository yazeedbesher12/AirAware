from pathlib import Path

import pandas as pd
import pytest

from app.ml.tft_inference import TFTInferenceAdapter, UnsafeTFTCheckpointError


BACKEND = Path(__file__).resolve().parents[1]


def test_tft_metadata_reconstructs_saved_contract_without_loading_checkpoint():
    adapter = TFTInferenceAdapter(BACKEND, "1h", load_checkpoint=False)
    assert adapter.metadata.max_encoder_length == 24
    assert adapter.metadata.min_encoder_length == 12
    assert adapter.metadata.encoder_history_hours == 6
    assert adapter.metadata.maximum_feature_lookback_hours == 24
    assert adapter.metadata.full_feature_history_hours == 30
    assert adapter.metadata.full_feature_history_samples == 121
    assert adapter.metadata.target_normalizer == "MultiNormalizer"
    assert adapter.metadata.real_time_ready is False


def test_tft_adapter_never_mixes_sensor_histories():
    adapter = TFTInferenceAdapter(BACKEND, "1h", load_checkpoint=False)
    timestamps = pd.date_range("2026-08-01", periods=121, freq="15min", tz="UTC")
    history = pd.DataFrame({"timestamp": timestamps, "sensor_id": [1] * 120 + [2]})
    with pytest.raises(ValueError, match="exactly one sensor"):
        adapter.predict(history)


def test_tft_adapter_fails_closed_on_future_target_normalizer():
    adapter = TFTInferenceAdapter(BACKEND, "6h", load_checkpoint=False)
    history = pd.DataFrame({"timestamp": pd.date_range("2026-08-01", periods=121, freq="15min", tz="UTC"), "sensor_id": 1})
    with pytest.raises(UnsafeTFTCheckpointError, match="EncoderNormalizer"):
        adapter.predict(history)
