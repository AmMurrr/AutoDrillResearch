from __future__ import annotations

import os

import numpy as np
import pytest

from neural_approach.wav2vec_extractor import (
    DEFAULT_MODEL_NAME,
    extract_wav2vec_embeddings,
    resolve_hf_token,
    resolve_model_name_or_path,
    statistical_pooling,
)


pytestmark = pytest.mark.unit


def test_extract_wav2vec_embeddings_input_validation() -> None:
    with pytest.raises(ValueError):
        extract_wav2vec_embeddings(np.ones(100, dtype=np.float32), sample_rate=8000)

    with pytest.raises(ValueError):
        extract_wav2vec_embeddings(np.array([], dtype=np.float32), sample_rate=16000)


def test_extract_wav2vec_embeddings_with_real_model() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

    samples = np.sin(2 * np.pi * 220 * np.arange(16000, dtype=np.float32) / 16000).astype(
        np.float32
    )
    try:
        result = extract_wav2vec_embeddings(
            samples,
            sample_rate=16000,
            model_name=DEFAULT_MODEL_NAME,
            device="cpu",
            hf_token=resolve_hf_token(None),
        )
    except OSError as exc:
        pytest.skip(
            f"Real wav2vec2 extractor test requires locally cached {DEFAULT_MODEL_NAME}: {exc}"
        )

    assert result.frame_embeddings.ndim == 2
    assert result.frame_embeddings.shape[0] > 0
    assert result.frame_embeddings.shape[1] > 0
    assert result.pooled_embedding.shape == (result.frame_embeddings.shape[1] * 2,)
    assert np.allclose(result.pooled_embedding, statistical_pooling(result.frame_embeddings))
    assert result.sample_rate == 16000
    assert result.model_name == DEFAULT_MODEL_NAME
    assert result.device == "cpu"
    assert result.embedding_layer == 6


def test_resolve_model_name_prefers_downloaded_project_model(monkeypatch, tmp_path) -> None:
    local_model_dir = tmp_path / "facebook-wav2vec2-base"
    local_model_dir.mkdir()
    (local_model_dir / "config.json").write_text("{}", encoding="utf-8")
    (local_model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (local_model_dir / "pytorch_model.bin").write_bytes(b"weights")

    monkeypatch.setattr("neural_approach.wav2vec_extractor.LOCAL_MODELS_DIR", tmp_path)

    assert resolve_model_name_or_path(DEFAULT_MODEL_NAME) == str(local_model_dir)


def test_resolve_model_name_falls_back_to_hub_name_for_incomplete_local_model(
    monkeypatch, tmp_path
) -> None:
    local_model_dir = tmp_path / "facebook-wav2vec2-base"
    local_model_dir.mkdir()
    (local_model_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("neural_approach.wav2vec_extractor.LOCAL_MODELS_DIR", tmp_path)

    assert resolve_model_name_or_path(DEFAULT_MODEL_NAME) == DEFAULT_MODEL_NAME


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
