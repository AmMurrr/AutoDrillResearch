from __future__ import annotations

import os

import pytest

from neural_approach.pipeline import analyze
from neural_approach.wav2vec_extractor import DEFAULT_MODEL_NAME, resolve_hf_token
from tests._paths import (
    HAPPY_EMPTY_AUDIO,
    HAPPY_NORMAL_AUDIO,
    HAPPY_PERFECT_AUDIO,
    HAPPY_PROBLEM_AUDIO,
    HAPPY_WRONG_WORD_AUDIO,
    TEST_DATA_DIR,
)


pytestmark = pytest.mark.integration


_ANALYZE_CACHE: dict[str, object] = {}


def _analyze_with_real_model(audio_path: str):
    cached = _ANALYZE_CACHE.get(audio_path)
    if cached is not None:
        return cached

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

    try:
        result = analyze(
            user_audio_path=audio_path,
            transcript="happy",
            similarity="cosine",
            model_name=DEFAULT_MODEL_NAME,
            hf_token=resolve_hf_token(None),
            sakoe_chiba_radius=12,
            use_vosk=False,
            raw_distance_alpha=0.65,
            max_anchors_per_class=12,
        )
    except OSError as exc:
        pytest.skip(
            f"Real wav2vec2 integration test requires locally cached {DEFAULT_MODEL_NAME}: {exc}"
        )

    _ANALYZE_CACHE[audio_path] = result
    return result


def test_neural_integration_uses_expected_test_data_files() -> None:
    assert TEST_DATA_DIR.exists()
    assert HAPPY_PERFECT_AUDIO.exists()
    assert HAPPY_NORMAL_AUDIO.exists()
    assert HAPPY_PROBLEM_AUDIO.exists()
    assert HAPPY_WRONG_WORD_AUDIO.exists()
    assert HAPPY_EMPTY_AUDIO.exists()


def test_neural_pipeline_happy_perfect_with_real_audio_and_model() -> None:
    result = _analyze_with_real_model(str(HAPPY_PERFECT_AUDIO))

    assert result.status == "ok"
    assert result.model_name == DEFAULT_MODEL_NAME
    assert result.pronunciation_score > 90.0


def test_neural_pipeline_happy_normal_with_real_audio_and_model() -> None:
    result = _analyze_with_real_model(str(HAPPY_NORMAL_AUDIO))

    assert result.status == "ok"
    assert result.model_name == DEFAULT_MODEL_NAME
    assert 30.0 <= result.pronunciation_score <= 90.0


def test_neural_pipeline_happy_problem_with_real_audio_and_model() -> None:
    result = _analyze_with_real_model(str(HAPPY_PROBLEM_AUDIO))

    assert result.status == "ok"
    assert result.model_name == DEFAULT_MODEL_NAME
    assert result.pronunciation_score < 30.0


def test_neural_pipeline_happy_empty_file_with_real_audio() -> None:
    result = analyze(
        user_audio_path=str(HAPPY_EMPTY_AUDIO),
        transcript="happy",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=None,
        sakoe_chiba_radius=12,
        use_vosk=False,
        raw_distance_alpha=0.65,
        max_anchors_per_class=12,
    )

    assert result.pronunciation_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_neural_pipeline_wrong_word_is_rejected_by_vosk() -> None:
    result = analyze(
        user_audio_path=str(HAPPY_WRONG_WORD_AUDIO),
        transcript="happy",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=None,
        sakoe_chiba_radius=12,
        use_vosk=True,
        raw_distance_alpha=0.65,
        max_anchors_per_class=12,
    )

    assert result.pronunciation_score == 0.0
    assert result.status == "wrong_word"
    assert "expected:happy" in result.reason
    assert "recognized:happy" not in result.reason
