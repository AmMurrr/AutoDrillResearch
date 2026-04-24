from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neural_approach.wav2vec_extractor import extract_wav2vec_embeddings, statistical_pooling


pytestmark = pytest.mark.unit


def test_extract_wav2vec_embeddings_input_validation() -> None:
    with pytest.raises(ValueError):
        extract_wav2vec_embeddings(np.ones(100, dtype=np.float32), sample_rate=8000)

    with pytest.raises(ValueError):
        extract_wav2vec_embeddings(np.array([], dtype=np.float32), sample_rate=16000)


def test_extract_wav2vec_embeddings_with_mocked_model(monkeypatch) -> None:
    class FakeFeatureExtractor:
        def __call__(self, speech, sampling_rate, return_tensors, return_attention_mask):
            del return_tensors
            del return_attention_mask
            assert sampling_rate == 16000
            input_values = torch.tensor(speech, dtype=torch.float32).unsqueeze(0)
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
            return {"input_values": input_values, "attention_mask": attention_mask}

    class FakeModel:
        def __call__(self, input_values, attention_mask=None):
            del attention_mask
            hidden = input_values.unsqueeze(-1).repeat(1, 1, 4)
            return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(
        "neural_approach.wav2vec_extractor._resolve_device",
        lambda _device=None: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "neural_approach.wav2vec_extractor._load_model_bundle",
        lambda _model_name, _device_str, _hf_token: (FakeFeatureExtractor(), FakeModel()),
    )

    samples = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    result = extract_wav2vec_embeddings(samples, sample_rate=16000, model_name="fake/model")

    assert result.frame_embeddings.shape == (4, 4)
    assert result.pooled_embedding.shape == (8,)
    assert np.allclose(result.pooled_embedding, statistical_pooling(result.frame_embeddings))
    assert result.sample_rate == 16000
    assert result.model_name == "fake/model"
    assert result.device == "cpu"


def test_statistical_pooling_concatenates_mean_and_std() -> None:
    frames = np.array(
        [
            [1.0, 2.0],
            [3.0, 6.0],
            [5.0, 10.0],
        ],
        dtype=np.float32,
    )

    pooled = statistical_pooling(frames)

    assert pooled.shape == (4,)
    assert np.allclose(pooled[:2], [3.0, 6.0])
    assert np.allclose(pooled[2:], np.std(frames, axis=0))
