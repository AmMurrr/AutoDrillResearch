from __future__ import annotations

from types import SimpleNamespace

import pytest

from asr.vosk import VoskError
from classic_approach.pipeline import analyze
from tests._paths import HAPPY_PERFECT_AUDIO, HAPPY_WRONG_WORD_AUDIO


pytestmark = pytest.mark.unit


def test_classic_pipeline_rejects_empty_transcript() -> None:
    result = analyze(
        user_audio_path=str(HAPPY_PERFECT_AUDIO),
        transcript="   ",
        use_vosk=False,
        max_anchors_per_class=3,
    )

    assert result.dtw_score == 0.0
    assert result.status == "invalid_reference"
    assert result.reason == "empty_transcript"


def test_classic_pipeline_returns_wrong_word_when_vosk_mismatches(monkeypatch) -> None:
    monkeypatch.setattr(
        "classic_approach.pipeline.check_expected_text_for_preprocessed_audio",
        lambda samples, sample_rate, expected_text: SimpleNamespace(
            is_match=False,
            expected_text="happy",
            recognized_text="world",
        ),
    )

    result = analyze(
        user_audio_path=str(HAPPY_WRONG_WORD_AUDIO),
        transcript="happy",
        max_anchors_per_class=3,
    )

    assert result.dtw_score == 0.0
    assert result.status == "wrong_word"
    assert "recognized:world" in result.reason
    assert result.verdict == "неудовлетворительно"


def test_classic_pipeline_returns_asr_error_when_vosk_fails(monkeypatch) -> None:
    def _raise_vosk_error(samples, sample_rate, expected_text):
        del samples
        del sample_rate
        del expected_text
        raise VoskError("model_unavailable")

    monkeypatch.setattr(
        "classic_approach.pipeline.check_expected_text_for_preprocessed_audio",
        _raise_vosk_error,
    )

    result = analyze(
        user_audio_path=str(HAPPY_PERFECT_AUDIO),
        transcript="happy",
        max_anchors_per_class=3,
    )

    assert result.dtw_score == 0.0
    assert result.status == "asr_error"
    assert "vosk_failure:model_unavailable" == result.reason
