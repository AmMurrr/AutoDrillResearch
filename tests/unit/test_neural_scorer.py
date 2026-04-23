from __future__ import annotations

import pytest

from neural_approach.scorer import (
    compute_anchor_profile_calibrated_scoring_result,
    compute_calibrated_scoring_result,
    compute_raw_distance,
    compute_scoring_result,
)
from scoring.anchor_calibration import (
    AnchorDistanceProfile,
    SigmoidCalibrationParams,
    fit_sigmoid_from_anchor_profiles,
)


pytestmark = pytest.mark.unit


def _sigmoid_params() -> SigmoidCalibrationParams:
    return SigmoidCalibrationParams(
        d100=0.1,
        d0=0.5,
        a=19.459101,
        b=0.3,
        epsilon=0.02,
    )


def _multi_anchor_params():
    perfect_profiles = [
        AnchorDistanceProfile(0.10, 0.18, 0.26),
        AnchorDistanceProfile(0.11, 0.17, 0.28),
        AnchorDistanceProfile(0.09, 0.16, 0.24),
    ]
    moderate_profiles = [
        AnchorDistanceProfile(0.16, 0.12, 0.20),
        AnchorDistanceProfile(0.18, 0.13, 0.22),
    ]
    fail_profiles = [
        AnchorDistanceProfile(0.27, 0.20, 0.11),
        AnchorDistanceProfile(0.29, 0.21, 0.12),
        AnchorDistanceProfile(0.25, 0.19, 0.10),
    ]
    params = fit_sigmoid_from_anchor_profiles(
        perfect_profiles=perfect_profiles,
        moderate_profiles=moderate_profiles,
        fail_profiles=fail_profiles,
    )
    return params, perfect_profiles, moderate_profiles, fail_profiles


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


def test_neural_compute_scoring_result_respects_verdict_thresholds() -> None:
    good = compute_scoring_result(1.0, 0.0, metric="cosine", model_name="m")
    acceptable = compute_scoring_result(0.4, 0.1, metric="cosine", model_name="m")
    weak = compute_scoring_result(-0.9, 0.8, metric="cosine", model_name="m")

    assert good.verdict == "хорошо"
    assert weak.verdict == "неудовлетворительно"
    assert good.pronunciation_score >= acceptable.pronunciation_score
    assert acceptable.pronunciation_score > weak.pronunciation_score


def test_neural_calibrated_score_is_bounded_and_high_for_good_distance() -> None:
    result = compute_calibrated_scoring_result(
        similarity=0.95,
        temporal_distance=0.1,
        metric="cosine",
        model_name="test-model",
        raw_distance=0.1,
        calibration_params=_sigmoid_params(),
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
        calibration_params=_sigmoid_params(),
        force_zero=True,
        reason="known_zero_anchor",
    )

    assert result.pronunciation_score == 0.0
    assert result.reason == "known_zero_anchor"


def test_neural_profile_calibrated_score_uses_anchor_profile() -> None:
    params, perfect_profiles, _moderate_profiles, _fail_profiles = _multi_anchor_params()

    result = compute_anchor_profile_calibrated_scoring_result(
        similarity=0.95,
        temporal_distance=0.1,
        metric="cosine",
        model_name="test-model",
        profile=perfect_profiles[0],
        calibration_params=params,
    )

    assert result.status == "ok"
    assert result.pronunciation_score > 80.0
    assert result.raw_distance == perfect_profiles[0].perfect_distance
    assert result.moderate_raw_distance == perfect_profiles[0].moderate_distance
    assert result.fail_raw_distance == perfect_profiles[0].fail_distance


def test_neural_profile_calibrated_score_rejects_invalid_profile() -> None:
    params, _perfect_profiles, _moderate_profiles, _fail_profiles = _multi_anchor_params()
    invalid_profile = AnchorDistanceProfile(float("inf"), 0.2, 0.1)

    result = compute_anchor_profile_calibrated_scoring_result(
        similarity=0.2,
        temporal_distance=0.8,
        metric="cosine",
        model_name="test-model",
        profile=invalid_profile,
        calibration_params=params,
    )

    assert result.pronunciation_score == 0.0
    assert result.status == "invalid_reference"
    assert result.reason == "invalid_anchor_profile"
