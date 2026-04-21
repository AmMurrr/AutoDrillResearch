from __future__ import annotations

from classic_approach.scorer import compute_calibrated_scoring_result
from scoring.anchor_calibration import SigmoidCalibrationParams


def _params() -> SigmoidCalibrationParams:
    return SigmoidCalibrationParams(
        d100=0.1,
        d0=0.5,
        a=19.459101,
        b=0.3,
        epsilon=0.02,
    )


def test_classic_calibrated_score_high_for_good_distance() -> None:
    result = compute_calibrated_scoring_result(
        distance=0.1,
        calibration_params=_params(),
    )

    assert result.status == "ok"
    assert result.dtw_score > 95.0


def test_classic_calibrated_score_low_for_bad_distance() -> None:
    result = compute_calibrated_scoring_result(
        distance=0.5,
        calibration_params=_params(),
    )

    assert result.dtw_score < 5.0


def test_classic_status_forces_zero_score() -> None:
    result = compute_calibrated_scoring_result(
        distance=0.2,
        calibration_params=_params(),
        status="wrong_word",
        reason="mismatch",
    )

    assert result.dtw_score == 0.0
    assert result.status == "wrong_word"
    assert result.reason == "mismatch"


def test_classic_force_zero_flag_sets_zero_score() -> None:
    result = compute_calibrated_scoring_result(
        distance=0.1,
        calibration_params=_params(),
        force_zero=True,
        reason="closer_to_zero_anchors",
    )

    assert result.dtw_score == 0.0
    assert result.reason == "closer_to_zero_anchors"
