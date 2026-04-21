from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from neural_approach.embedding_comparator import compare_embeddings
from neural_approach.pipeline import analyze
from neural_approach.scorer import compute_scoring_result
from neural_approach.wav2vec_extractor import (
    DEFAULT_MODEL_NAME,
    HF_TOKEN_ENV_VAR,
    extract_wav2vec_embeddings,
    resolve_hf_token,
)


TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "test"
HAPPY_PERFECT_AUDIO = TEST_DATA_DIR / "happy_perfect.wav"
HAPPY_NORMAL_AUDIO = TEST_DATA_DIR / "happy_normal.wav"
HAPPY_PROBLEM_AUDIO = TEST_DATA_DIR / "happy_problem.wav"
HAPPY_WRONG_WORD_AUDIO = TEST_DATA_DIR / "happy_wrong_word.mp3"
HAPPY_EMPTY_AUDIO = TEST_DATA_DIR / "happy_empty.wav"

_ANALYZE_CACHE: dict[str, object] = {}


@pytest.fixture(autouse=True)
def _mock_vosk_word_gate(monkeypatch):
    monkeypatch.setattr(
        "neural_approach.pipeline.check_expected_text_for_preprocessed_audio",
        lambda samples, sample_rate, expected_text: SimpleNamespace(
            is_match=True,
            expected_text=(expected_text or "").strip().lower(),
            recognized_text=(expected_text or "").strip().lower(),
        ),
    )


def _require_hf_token() -> str:
    token = resolve_hf_token(None)
    if token:
        return token

    pytest.skip(
        "Neural integration tests require HF token in env. "
        f"Set {HF_TOKEN_ENV_VAR}"
    )


def _analyze_test_audio(audio_path: Path):
    cache_key = str(audio_path.resolve())
    cached = _ANALYZE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        result = analyze(
            user_audio_path=str(audio_path),
            transcript="happy",
            similarity="cosine",
            model_name=DEFAULT_MODEL_NAME,
            hf_token=_require_hf_token(),
            max_anchors_per_class=2,
        )
    except OSError as exc:
        pytest.skip(
            "wav2vec2-base is not available in local cache for offline test run: "
            f"{exc}"
        )

    _ANALYZE_CACHE[cache_key] = result
    return result


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


def test_compare_embeddings_passes_sakoe_chiba_radius(monkeypatch) -> None:
    frames = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    captured: dict[str, int | None] = {}

    def fake_distance(seq_a, seq_b, window=None):
        _ = (seq_a, seq_b)
        captured["window"] = window
        return 0.0

    monkeypatch.setattr("neural_approach.embedding_comparator.dtw_ndim.distance", fake_distance)

    result = compare_embeddings(
        frames,
        frames,
        metric="cosine",
        sakoe_chiba_radius=9,
    )

    assert result.metric == "cosine"
    assert captured["window"] == 9


def test_compare_embeddings_disables_band_for_non_positive_radius(monkeypatch) -> None:
    frames = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    captured: dict[str, int | None] = {}

    def fake_distance(seq_a, seq_b, window=None):
        _ = (seq_a, seq_b)
        captured["window"] = window
        return 0.0

    monkeypatch.setattr("neural_approach.embedding_comparator.dtw_ndim.distance", fake_distance)

    compare_embeddings(
        frames,
        frames,
        metric="cosine",
        sakoe_chiba_radius=0,
    )

    assert captured["window"] is None


def test_compare_embeddings_invalid_metric() -> None:
    frames = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compare_embeddings(frames, frames, metric="manhattan")


def test_neural_analyze_euclidean_not_supported() -> None:
    with pytest.raises(ValueError):
        analyze(
            user_audio_path="unused.wav",
            transcript="happy",
            similarity="euclidean",
            model_name=DEFAULT_MODEL_NAME,
            hf_token=None,
        )


def test_compute_scoring_result_verdict_thresholds() -> None:
    good = compute_scoring_result(1.0, 0.0, metric="cosine", model_name="m")
    acceptable = compute_scoring_result(0.4, 0.1, metric="cosine", model_name="m")
    weak = compute_scoring_result(-0.9, 0.8, metric="cosine", model_name="m")

    assert good.verdict == "хорошо"
    assert weak.verdict == "неудовлетворительно"
    assert good.pronunciation_score >= acceptable.pronunciation_score
    assert acceptable.pronunciation_score > weak.pronunciation_score


def test_neural_analyze_uses_expected_test_data_files() -> None:
    assert TEST_DATA_DIR.exists()
    assert HAPPY_PERFECT_AUDIO.exists()
    assert HAPPY_NORMAL_AUDIO.exists()
    assert HAPPY_PROBLEM_AUDIO.exists()
    assert HAPPY_WRONG_WORD_AUDIO.exists()
    assert HAPPY_EMPTY_AUDIO.exists()


def test_neural_analyze_happy_perfect_runs_with_real_wav2vec2() -> None:
    result = _analyze_test_audio(HAPPY_PERFECT_AUDIO)

    assert result.status == "ok"
    assert result.model_name == DEFAULT_MODEL_NAME
    assert 0.0 <= result.pronunciation_score <= 100.0


def test_neural_analyze_happy_problem_runs_with_real_wav2vec2() -> None:
    result = _analyze_test_audio(HAPPY_PROBLEM_AUDIO)

    assert result.status == "ok"
    assert result.model_name == DEFAULT_MODEL_NAME
    assert 0.0 <= result.pronunciation_score <= 100.0


def test_neural_analyze_happy_wrong_word_returns_zero_with_vosk(monkeypatch) -> None:
    monkeypatch.setattr(
        "neural_approach.pipeline.check_expected_text_for_preprocessed_audio",
        lambda samples, sample_rate, expected_text: SimpleNamespace(
            is_match=False,
            expected_text="happy",
            recognized_text="world",
        ),
    )

    result = analyze(
        user_audio_path=str(HAPPY_WRONG_WORD_AUDIO),
        transcript="happy",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=_require_hf_token(),
        max_anchors_per_class=2,
    )

    assert result.pronunciation_score == 0.0
    assert result.status == "wrong_word"
    assert "recognized:world" in result.reason


def test_neural_analyze_happy_empty_file_is_marked_empty_audio() -> None:
    result = analyze(
        user_audio_path=str(HAPPY_EMPTY_AUDIO),
        transcript="happy",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=_require_hf_token(),
        max_anchors_per_class=2,
    )

    assert result.pronunciation_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_neural_analyze_empty_audio_returns_empty_audio_status() -> None:
    silent = np.zeros(16000, dtype=np.float32)

    with NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, silent, samplerate=16000)
        result = analyze(
            user_audio_path=tmp.name,
            transcript="happy",
            similarity="cosine",
            model_name=DEFAULT_MODEL_NAME,
            hf_token=_require_hf_token(),
            max_anchors_per_class=2,
        )

    assert result.pronunciation_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


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
