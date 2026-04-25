from __future__ import annotations

import pytest

from classic_approach.pipeline import analyze
from tests._paths import (
    HAPPY_EMPTY_AUDIO,
    HAPPY_NORMAL_AUDIO,
    HAPPY_PERFECT_AUDIO,
    HAPPY_PROBLEM_AUDIO,
    HAPPY_WRONG_WORD_AUDIO,
    TEST_DATA_DIR,
)


pytestmark = pytest.mark.integration


def _analyze_happy_without_vosk(audio_path: str):
    return analyze(
        user_audio_path=audio_path,
        transcript="happy",
        use_vosk=False,
        sakoe_chiba_radius=12,
        max_anchors_per_class=12,
    )


def test_classic_integration_uses_expected_test_data_files() -> None:
    assert TEST_DATA_DIR.exists()
    assert HAPPY_PERFECT_AUDIO.exists()
    assert HAPPY_NORMAL_AUDIO.exists()
    assert HAPPY_PROBLEM_AUDIO.exists()
    assert HAPPY_WRONG_WORD_AUDIO.exists()
    assert HAPPY_EMPTY_AUDIO.exists()


def test_classic_pipeline_happy_perfect_with_real_audio() -> None:
    result = _analyze_happy_without_vosk(str(HAPPY_PERFECT_AUDIO))

    assert result.status == "ok"
    assert result.dtw_score > 90.0


def test_classic_pipeline_happy_normal_with_real_audio() -> None:
    result = _analyze_happy_without_vosk(str(HAPPY_NORMAL_AUDIO))

    assert result.status == "ok"
    assert 30.0 <= result.dtw_score <= 90.0


def test_classic_pipeline_happy_problem_with_real_audio() -> None:
    result = _analyze_happy_without_vosk(str(HAPPY_PROBLEM_AUDIO))

    assert result.status == "ok"
    assert result.dtw_score < 30.0


def test_classic_pipeline_happy_empty_file_with_real_audio() -> None:
    result = _analyze_happy_without_vosk(str(HAPPY_EMPTY_AUDIO))

    assert result.dtw_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_classic_pipeline_wrong_word_is_rejected_by_vosk() -> None:
    result = analyze(
        user_audio_path=str(HAPPY_WRONG_WORD_AUDIO),
        transcript="happy",
        use_vosk=True,
        sakoe_chiba_radius=12,
        max_anchors_per_class=12,
    )

    assert result.dtw_score == 0.0
    assert result.status == "wrong_word"
    assert "expected:happy" in result.reason
    assert "recognized:happy" not in result.reason
