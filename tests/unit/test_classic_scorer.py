from __future__ import annotations

import pytest

from classic_approach.scorer import (
    compute_calibrated_scoring_result,
    compute_profile_calibrated_scoring_result,
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


def test_classic_calibrated_score_high_for_good_distance() -> None:
    result = compute_calibrated_scoring_result(
        distance=0.1,
        calibration_params=_sigmoid_params(),
    )

    assert result.status == "ok"
    assert result.dtw_score > 95.0


def test_classic_calibrated_score_low_for_bad_distance() -> None:
    result = compute_calibrated_scoring_result(
        distance=0.5,
        calibration_params=_sigmoid_params(),
    )

    assert result.dtw_score < 5.0


def test_classic_status_forces_zero_score() -> None:
    result = compute_calibrated_scoring_result(
        distance=0.2,
        calibration_params=_sigmoid_params(),
        status="wrong_word",
        reason="mismatch",
    )

    assert result.dtw_score == 0.0
    assert result.status == "wrong_word"
    assert result.reason == "mismatch"


def test_classic_force_zero_flag_sets_zero_score() -> None:
    result = compute_calibrated_scoring_result(
        distance=0.1,
        calibration_params=_sigmoid_params(),
        force_zero=True,
        reason="closer_to_zero_anchors",
    )

    assert result.dtw_score == 0.0
    assert result.reason == "closer_to_zero_anchors"


def test_classic_profile_calibrated_score_uses_anchor_profile() -> None:
    params, perfect_profiles, _moderate_profiles, _fail_profiles = _multi_anchor_params()

    result = compute_profile_calibrated_scoring_result(
        profile=perfect_profiles[0],
        calibration_params=params,
    )

    assert result.status == "ok"
    assert result.dtw_score > 80.0
    assert result.distance == perfect_profiles[0].perfect_distance
    assert result.moderate_distance == perfect_profiles[0].moderate_distance
    assert result.fail_distance == perfect_profiles[0].fail_distance


def test_classic_profile_calibrated_score_rejects_invalid_profile() -> None:
    params, _perfect_profiles, _moderate_profiles, _fail_profiles = _multi_anchor_params()
    invalid_profile = AnchorDistanceProfile(float("inf"), 0.2, 0.1)

    result = compute_profile_calibrated_scoring_result(
        profile=invalid_profile,
        calibration_params=params,
    )

    assert result.dtw_score == 0.0
    assert result.status == "invalid_reference"
    assert result.reason == "invalid_anchor_profile"
