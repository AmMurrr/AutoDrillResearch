from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from classic_approach.dtw import dtw_distance
from classic_approach.mfcc_extractor import apply_cmvn
from classic_approach.pipeline import analyze


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
        "classic_approach.pipeline.check_expected_text_for_preprocessed_audio",
        lambda samples, sample_rate, expected_text: SimpleNamespace(
            is_match=True,
            expected_text=(expected_text or "").strip().lower(),
            recognized_text=(expected_text or "").strip().lower(),
        ),
    )


def _analyze_test_audio(audio_path: Path):
    cache_key = str(audio_path.resolve())
    cached = _ANALYZE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    result = analyze(
        user_audio_path=str(audio_path),
        transcript="happy",
        max_anchors_per_class=3,
    )
    _ANALYZE_CACHE[cache_key] = result
    return result


def test_dtw_distance_identical_features_is_zero() -> None:
    features = np.array(
        [
            [0.0, 0.5, 1.0, 0.3],
            [1.0, 1.5, 0.8, 0.2],
        ],
        dtype=np.float32,
    )

    distance = dtw_distance(features, features)
    assert distance == 0.0


def test_dtw_distance_passes_sakoe_chiba_radius(monkeypatch) -> None:
    features = np.array(
        [
            [0.0, 0.2, 0.4],
            [1.0, 0.9, 0.8],
        ],
        dtype=np.float32,
    )
    captured: dict[str, int | None] = {}

    def fake_distance(seq_a, seq_b, window=None):
        _ = (seq_a, seq_b)
        captured["window"] = window
        return 0.0

    monkeypatch.setattr("classic_approach.dtw.dtw_ndim.distance", fake_distance)

    distance = dtw_distance(features, features, sakoe_chiba_radius=12)

    assert distance == 0.0
    assert captured["window"] == 12


def test_dtw_distance_disables_band_for_non_positive_radius(monkeypatch) -> None:
    features = np.array(
        [
            [0.0, 0.2, 0.4],
            [1.0, 0.9, 0.8],
        ],
        dtype=np.float32,
    )
    captured: dict[str, int | None] = {}

    def fake_distance(seq_a, seq_b, window=None):
        _ = (seq_a, seq_b)
        captured["window"] = window
        return 0.0

    monkeypatch.setattr("classic_approach.dtw.dtw_ndim.distance", fake_distance)

    distance = dtw_distance(features, features, sakoe_chiba_radius=0)

    assert distance == 0.0
    assert captured["window"] is None


def test_apply_cmvn_rowwise_mean_and_std() -> None:
    mfcc = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 11.0, 12.0, 13.0],
        ],
        dtype=np.float32,
    )

    normalized = apply_cmvn(mfcc)
    row_means = np.mean(normalized, axis=1)
    row_stds = np.std(normalized, axis=1)

    assert np.allclose(row_means, 0.0, atol=1e-6)
    assert np.allclose(row_stds, 1.0, atol=1e-6)


def test_classic_analyze_uses_expected_test_data_files() -> None:
    assert TEST_DATA_DIR.exists()
    assert HAPPY_PERFECT_AUDIO.exists()
    assert HAPPY_NORMAL_AUDIO.exists()
    assert HAPPY_PROBLEM_AUDIO.exists()
    assert HAPPY_WRONG_WORD_AUDIO.exists()
    assert HAPPY_EMPTY_AUDIO.exists()


def test_classic_analyze_happy_perfect_is_regular_word_case() -> None:
    result = _analyze_test_audio(HAPPY_PERFECT_AUDIO)

    assert result.status == "ok"
    assert 0.0 <= result.dtw_score <= 100.0


def test_classic_analyze_happy_problem_is_regular_word_case() -> None:
    result = _analyze_test_audio(HAPPY_PROBLEM_AUDIO)

    assert result.status == "ok"
    assert 0.0 <= result.dtw_score <= 100.0


def test_classic_analyze_happy_wrong_word_is_zero_with_vosk(monkeypatch) -> None:
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


def test_classic_analyze_happy_empty_file_is_marked_empty_audio() -> None:
    result = _analyze_test_audio(HAPPY_EMPTY_AUDIO)

    assert result.dtw_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_classic_analyze_empty_audio_returns_empty_audio_status() -> None:
    silent = np.zeros(16000, dtype=np.float32)

    with NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, silent, samplerate=16000)
        result = analyze(
            user_audio_path=tmp.name,
            transcript="happy",
            max_anchors_per_class=3,
        )

    assert result.dtw_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"
