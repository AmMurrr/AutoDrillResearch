from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf

from classic_approach.dtw import dtw_distance
from classic_approach.forced_aligner import pseudo_localize_errors
from classic_approach.mfcc_extractor import apply_cmvn
from classic_approach.pipeline import _distance_to_score, analyze


TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "test"
REFERENCE_AUDIO = TEST_DATA_DIR / "pronunciation_en_hello.wav"
HELLO_PERFECT_AUDIO = TEST_DATA_DIR / "hello_perfect.mp3"
HELLO_NORMAL_AUDIO = TEST_DATA_DIR / "hello_normal.wav"
HELLO_PROBLEM_AUDIO = TEST_DATA_DIR / "hello_problem.wav"
HELLO_WRONG_WORD_AUDIO = TEST_DATA_DIR / "hello_wrong_word.mp3"
HELLO_EMPTY_AUDIO = TEST_DATA_DIR / "hello_empty.wav"


def _analyze_test_audio(audio_path: Path):
    return analyze(
        user_audio_path=str(audio_path),
        reference_audio_path=str(REFERENCE_AUDIO),
        transcript="hello",
    )


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


def test_distance_to_score_monotonicity() -> None:
    good = _distance_to_score(distance=0.05, user_frames=100, reference_frames=100)
    bad = _distance_to_score(distance=0.6, user_frames=100, reference_frames=100)
    assert good > bad


def test_pseudo_localize_errors_marks_end_of_word() -> None:
    reference = np.zeros((2, 12), dtype=np.float32)
    user = reference.copy()
    user[:, 8:] = 4.0

    diagnostics = pseudo_localize_errors(user, reference, transcript="hello")

    assert len(diagnostics) == 1
    assert diagnostics[0]["word"] == "hello"
    assert diagnostics[0]["problem_zone"] == "конец"
    assert diagnostics[0]["is_problematic"] is True


def test_classic_analyze_uses_expected_test_data_files() -> None:
    assert TEST_DATA_DIR.exists()
    assert REFERENCE_AUDIO.exists()
    assert HELLO_PERFECT_AUDIO.exists()
    assert HELLO_NORMAL_AUDIO.exists()
    assert HELLO_PROBLEM_AUDIO.exists()
    assert HELLO_WRONG_WORD_AUDIO.exists()
    assert HELLO_EMPTY_AUDIO.exists()


def test_classic_analyze_hello_perfect_is_near_reference() -> None:
    result = _analyze_test_audio(HELLO_PERFECT_AUDIO)

    assert result.dtw_score > 90.0
    assert result.verdict == "хорошо"
    assert result.problematic_phonemes == []


def test_classic_analyze_hello_normal_is_mid_quality() -> None:
    result = _analyze_test_audio(HELLO_NORMAL_AUDIO)

    assert 40.0 <= result.dtw_score < 90.0
    assert result.verdict == "удовлетворительно"


def test_classic_analyze_hello_problem_is_low_quality() -> None:
    result = _analyze_test_audio(HELLO_PROBLEM_AUDIO)

    assert result.dtw_score < 45.0
    assert result.verdict == "неудовлетворительно"


def test_classic_analyze_hello_wrong_word_is_low() -> None:
    result = _analyze_test_audio(HELLO_WRONG_WORD_AUDIO)

    assert result.dtw_score < 25.0
    assert result.verdict == "неудовлетворительно"


def test_classic_analyze_real_audio_ordering() -> None:
    perfect = _analyze_test_audio(HELLO_PERFECT_AUDIO)
    normal = _analyze_test_audio(HELLO_NORMAL_AUDIO)
    wrong_word = _analyze_test_audio(HELLO_WRONG_WORD_AUDIO)
    problem = _analyze_test_audio(HELLO_PROBLEM_AUDIO)

    assert perfect.dtw_score > normal.dtw_score > problem.dtw_score
    assert problem.dtw_score > wrong_word.dtw_score


def test_classic_analyze_hello_empty_file_is_marked_empty_audio() -> None:
    result = _analyze_test_audio(HELLO_EMPTY_AUDIO)

    assert result.dtw_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_classic_analyze_multiple_references_aggregates_results() -> None:
    single = analyze(
        user_audio_path=str(HELLO_NORMAL_AUDIO),
        reference_audio_path=str(REFERENCE_AUDIO),
        transcript="hello",
    )
    multi = analyze(
        user_audio_path=str(HELLO_NORMAL_AUDIO),
        reference_audio_path=[str(REFERENCE_AUDIO), str(REFERENCE_AUDIO)],
        transcript="hello",
    )

    assert np.isclose(multi.dtw_score, single.dtw_score, atol=1e-6)
    assert multi.verdict == single.verdict


def test_classic_analyze_empty_audio_returns_empty_audio_status() -> None:
    silent = np.zeros(16000, dtype=np.float32)

    with NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, silent, samplerate=16000)
        result = analyze(
            user_audio_path=tmp.name,
            reference_audio_path=str(REFERENCE_AUDIO),
            transcript="hello",
        )

    assert result.dtw_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_classic_analyze_nonword_tone_returns_empty_audio_status() -> None:
    sample_rate = 16000
    duration_sec = 1.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)
    tone = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    with NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, tone, samplerate=sample_rate)
        result = analyze(
            user_audio_path=tmp.name,
            reference_audio_path=str(REFERENCE_AUDIO),
            transcript="hello",
        )

    assert result.dtw_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"
