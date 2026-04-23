from __future__ import annotations

from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import soundfile as sf

from classic_approach.pipeline import analyze
from tests._paths import (
    HAPPY_EMPTY_AUDIO,
    HAPPY_PERFECT_AUDIO,
    HAPPY_PROBLEM_AUDIO,
    HAPPY_WRONG_WORD_AUDIO,
    TEST_DATA_DIR,
)


pytestmark = pytest.mark.integration


def test_classic_integration_uses_expected_test_data_files() -> None:
    assert TEST_DATA_DIR.exists()
    assert HAPPY_PERFECT_AUDIO.exists()
    assert HAPPY_PROBLEM_AUDIO.exists()
    assert HAPPY_WRONG_WORD_AUDIO.exists()
    assert HAPPY_EMPTY_AUDIO.exists()


def test_classic_pipeline_happy_perfect_with_real_audio() -> None:
    result = analyze(
        user_audio_path=str(HAPPY_PERFECT_AUDIO),
        transcript="happy",
        use_vosk=False,
        max_anchors_per_class=3,
    )

    assert result.status == "ok"
    assert 0.0 <= result.dtw_score <= 100.0


def test_classic_pipeline_happy_problem_with_real_audio() -> None:
    result = analyze(
        user_audio_path=str(HAPPY_PROBLEM_AUDIO),
        transcript="happy",
        use_vosk=False,
        max_anchors_per_class=3,
    )

    assert result.status == "ok"
    assert 0.0 <= result.dtw_score <= 100.0


def test_classic_pipeline_happy_empty_file_with_real_audio() -> None:
    result = analyze(
        user_audio_path=str(HAPPY_EMPTY_AUDIO),
        transcript="happy",
        use_vosk=False,
        max_anchors_per_class=3,
    )

    assert result.dtw_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_classic_pipeline_silent_generated_audio_returns_empty_audio() -> None:
    silent = np.zeros(16000, dtype=np.float32)

    with NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, silent, samplerate=16000)
        result = analyze(
            user_audio_path=tmp.name,
            transcript="happy",
            use_vosk=False,
            max_anchors_per_class=3,
        )

    assert result.dtw_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"
