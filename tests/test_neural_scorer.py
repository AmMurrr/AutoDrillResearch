from __future__ import annotations

from neural_approach.scorer import compute_calibrated_scoring_result, compute_raw_distance
from scoring.anchor_calibration import SigmoidCalibrationParams


def _params() -> SigmoidCalibrationParams:
    return SigmoidCalibrationParams(
        d100=0.1,
        d0=0.5,
        a=19.459101,
        b=0.3,
        epsilon=0.02,
    )


def test_compute_raw_distance_for_best_case() -> None:
    raw = compute_raw_distance(
        similarity=1.0,
        temporal_distance=0.0,
        metric="cosine",
        alpha=0.65,
    )

    assert raw == 0.0


def test_compute_raw_distance_for_worst_case() -> None:
    raw = compute_raw_distance(
        similarity=-1.0,
        temporal_distance=1.0,
        metric="cosine",
        alpha=0.5,
    )

    assert raw == 1.0


def test_neural_calibrated_score_is_bounded_and_high_for_good_distance() -> None:
    result = compute_calibrated_scoring_result(
        similarity=0.95,
        temporal_distance=0.1,
        metric="cosine",
        model_name="test-model",
        raw_distance=0.1,
        calibration_params=_params(),
    )

    assert result.pronunciation_score > 95.0
    assert 0.0 <= result.pronunciation_score <= 100.0


def test_neural_calibrated_score_is_zero_for_force_zero() -> None:
    result = compute_calibrated_scoring_result(
        similarity=0.2,
        temporal_distance=0.8,
        metric="cosine",
        model_name="test-model",
        raw_distance=0.4,
        calibration_params=_params(),
        force_zero=True,
        reason="known_zero_anchor",
    )

    assert result.pronunciation_score == 0.0
    assert result.reason == "known_zero_anchor"
