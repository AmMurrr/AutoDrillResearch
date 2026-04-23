from __future__ import annotations

import os
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import soundfile as sf

from neural_approach.pipeline import analyze
from neural_approach.wav2vec_extractor import DEFAULT_MODEL_NAME, resolve_hf_token
from tests._paths import (
    HAPPY_EMPTY_AUDIO,
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
            use_vosk=False,
            max_anchors_per_class=2,
        )
    except OSError as exc:
        pytest.skip(
            "Real wav2vec2 integration test requires locally cached "
            f"{DEFAULT_MODEL_NAME}: {exc}"
        )

    _ANALYZE_CACHE[audio_path] = result
    return result


def test_neural_integration_uses_expected_test_data_files() -> None:
    assert TEST_DATA_DIR.exists()
    assert HAPPY_PERFECT_AUDIO.exists()
    assert HAPPY_PROBLEM_AUDIO.exists()
    assert HAPPY_WRONG_WORD_AUDIO.exists()
    assert HAPPY_EMPTY_AUDIO.exists()


def test_neural_pipeline_happy_perfect_with_real_audio_and_model() -> None:
    result = _analyze_with_real_model(str(HAPPY_PERFECT_AUDIO))

    assert result.status == "ok"
    assert result.model_name == DEFAULT_MODEL_NAME
    assert 0.0 <= result.pronunciation_score <= 100.0


def test_neural_pipeline_happy_problem_with_real_audio_and_model() -> None:
    result = _analyze_with_real_model(str(HAPPY_PROBLEM_AUDIO))

    assert result.status == "ok"
    assert result.model_name == DEFAULT_MODEL_NAME
    assert 0.0 <= result.pronunciation_score <= 100.0


def test_neural_pipeline_happy_empty_file_with_real_audio() -> None:
    result = analyze(
        user_audio_path=str(HAPPY_EMPTY_AUDIO),
        transcript="happy",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=None,
        use_vosk=False,
        max_anchors_per_class=2,
    )

    assert result.pronunciation_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_neural_pipeline_silent_generated_audio_returns_empty_audio() -> None:
    silent = np.zeros(16000, dtype=np.float32)

    with NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, silent, samplerate=16000)
        result = analyze(
            user_audio_path=tmp.name,
            transcript="happy",
            similarity="cosine",
            model_name=DEFAULT_MODEL_NAME,
            hf_token=None,
            use_vosk=False,
            max_anchors_per_class=2,
        )

    assert result.pronunciation_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"
