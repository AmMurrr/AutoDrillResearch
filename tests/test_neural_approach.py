from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neural_approach.embedding_comparator import compare_embeddings
from neural_approach.pipeline import analyze
from neural_approach.scorer import compute_scoring_result
from neural_approach.wav2vec_extractor import extract_wav2vec_embeddings


def test_compare_embeddings_cosine_identical_sequences() -> None:
    frames = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = compare_embeddings(frames, frames, metric="cosine")

    assert result.metric == "cosine"
    assert np.isclose(result.similarity, 1.0, atol=1e-6)
    assert np.isclose(result.temporal_distance, 0.0, atol=1e-6)


def test_compare_embeddings_invalid_metric() -> None:
    frames = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compare_embeddings(frames, frames, metric="manhattan")


def test_compute_scoring_result_verdict_thresholds() -> None:
    good = compute_scoring_result(1.0, 0.0, metric="cosine", model_name="m")
    acceptable = compute_scoring_result(0.4, 0.1, metric="cosine", model_name="m")
    weak = compute_scoring_result(-0.9, 0.8, metric="cosine", model_name="m")

    assert good.verdict == "хорошо"
    assert acceptable.verdict == "удовлетворительно"
    assert weak.verdict == "неудовлетворительно"


def test_neural_pipeline_analyze_happy_path(monkeypatch) -> None:
    audio = SimpleNamespace(samples=np.ones(1600, dtype=np.float32), sample_rate=16000)

    user_frames = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]],
        dtype=np.float32,
    )
    ref_frames = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]],
        dtype=np.float32,
    )

    state = {"call_idx": 0}

    def fake_preprocess(_path: str):
        return audio

    def fake_extract(_samples, _sr, model_name="facebook/wav2vec2-base", device=None, hf_token=None):
        _ = (device, hf_token)
        state["call_idx"] += 1
        frames = user_frames if state["call_idx"] == 1 else ref_frames
        pooled = np.mean(frames, axis=0, dtype=np.float32)
        return SimpleNamespace(
            frame_embeddings=frames,
            pooled_embedding=pooled,
            sample_rate=16000,
            model_name=model_name,
            device="cpu",
        )

    monkeypatch.setattr("neural_approach.pipeline.preprocess_audio", fake_preprocess)
    monkeypatch.setattr("neural_approach.pipeline.extract_wav2vec_embeddings", fake_extract)

    result = analyze("user.wav", "ref.wav", transcript="hello", similarity="cosine")

    assert result.pronunciation_score > 95.0
    assert result.verdict == "хорошо"
    assert result.metric == "cosine"


def test_extract_wav2vec_embeddings_input_validation() -> None:
    with pytest.raises(ValueError):
        extract_wav2vec_embeddings(np.ones(100, dtype=np.float32), sample_rate=8000)

    with pytest.raises(ValueError):
        extract_wav2vec_embeddings(np.array([], dtype=np.float32), sample_rate=16000)


def test_extract_wav2vec_embeddings_with_mocked_model(monkeypatch) -> None:
    class FakeFeatureExtractor:
        def __call__(self, speech, sampling_rate, return_tensors, return_attention_mask):
            _ = (return_tensors, return_attention_mask)
            assert sampling_rate == 16000
            input_values = torch.tensor(speech, dtype=torch.float32).unsqueeze(0)
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
            return {"input_values": input_values, "attention_mask": attention_mask}

    class FakeModel:
        def __call__(self, input_values, attention_mask=None):
            _ = attention_mask
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
    assert result.pooled_embedding.shape == (4,)
    assert result.sample_rate == 16000
    assert result.model_name == "fake/model"
    assert result.device == "cpu"
