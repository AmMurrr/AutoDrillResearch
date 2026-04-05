from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from classic_approach.dtw import dtw_distance
from classic_approach.forced_aligner import pseudo_localize_errors
from classic_approach.mfcc_extractor import apply_cmvn
from classic_approach.pipeline import _distance_to_score, analyze


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


def test_classic_analyze_happy_path(monkeypatch) -> None:
    user_audio = SimpleNamespace(samples=np.ones(1600, dtype=np.float32), sample_rate=16000)
    ref_audio = SimpleNamespace(samples=np.ones(1600, dtype=np.float32), sample_rate=16000)

    user_mfcc = np.ones((20, 12), dtype=np.float32)
    ref_mfcc = np.ones((20, 12), dtype=np.float32)

    state = {"call_idx": 0}

    def fake_preprocess(_path: str):
        return user_audio if _path == "user.wav" else ref_audio

    def fake_extract(*_args, **_kwargs):
        state["call_idx"] += 1
        return user_mfcc if state["call_idx"] == 1 else ref_mfcc

    monkeypatch.setattr("classic_approach.pipeline.preprocess_audio", fake_preprocess)
    monkeypatch.setattr("classic_approach.pipeline.extract_mfcc", fake_extract)
    monkeypatch.setattr("classic_approach.pipeline.dtw_distance", lambda _a, _b: 0.0)

    result = analyze("user.wav", "ref.wav", transcript="hello")

    assert result.dtw_score == 100.0
    assert result.verdict == "хорошо"
    assert result.distance == 0.0
    assert len(result.error_localization) == 1
    assert result.error_localization[0]["word"] == "hello"


def test_classic_analyze_empty_features_returns_zero(monkeypatch) -> None:
    audio = SimpleNamespace(samples=np.ones(1600, dtype=np.float32), sample_rate=16000)

    monkeypatch.setattr("classic_approach.pipeline.preprocess_audio", lambda _p: audio)
    monkeypatch.setattr(
        "classic_approach.pipeline.extract_mfcc",
        lambda *_args, **_kwargs: np.zeros((20, 0), dtype=np.float32),
    )

    result = analyze("user.wav", "ref.wav", transcript="hello")

    assert result.dtw_score == 0.0
    assert result.verdict == "неудовлетворительно"
    assert np.isinf(result.distance)
    assert result.error_localization == []
