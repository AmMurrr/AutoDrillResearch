from dataclasses import dataclass

import numpy as np
from app.logging_config import get_logger

from scoring.anchor_calibration import (
    AnchorDistanceProfile,
    MultiAnchorSigmoidParams,
    SigmoidCalibrationParams,
    score_from_anchor_profile,
    sigmoid_score,
)


logger = get_logger(__name__)


def _verdict_from_score(score: float) -> str:
    if score >= 80.0:
        return "хорошо"
    if score >= 60.0:
        return "удовлетворительно"
    return "неудовлетворительно"


def _build_scoring_result(
    score: float,
    distance: float,
    status: str = "ok",
    reason: str = "",
) -> "ScoringResult":
    clipped_score = float(np.clip(float(score), 0.0, 100.0))
    return ScoringResult(
        dtw_score=clipped_score,
        verdict=_verdict_from_score(clipped_score),
        distance=float(distance),
        status=status,
        reason=reason,
    )


@dataclass
class ScoringResult:
    dtw_score: float
    verdict: str
    distance: float
    status: str = "ok"
    reason: str = ""
    d100: float | None = None
    d0: float | None = None
    moderate_distance: float | None = None
    fail_distance: float | None = None


def ComputeScoringResult(
    dtw_score: float,
    distance: float,
    status: str = "ok",
    reason: str = "",
    d100: float | None = None,
    d0: float | None = None,
    moderate_distance: float | None = None,
    fail_distance: float | None = None,
) -> ScoringResult:
    result = _build_scoring_result(
        score=float(dtw_score),
        distance=distance,
        status=status,
        reason=reason,
    )
    result.d100 = d100
    result.d0 = d0
    result.moderate_distance = moderate_distance
    result.fail_distance = fail_distance
    return result


def compute_calibrated_scoring_result(
    distance: float,
    calibration_params: SigmoidCalibrationParams,
    status: str = "ok",
    reason: str = "",
    force_zero: bool = False,
) -> ScoringResult:
    if status != "ok":
        logger.warning("Classic scoring forced to zero because status=%s", status)
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float(distance),
            status=status,
            reason=reason,
            d100=calibration_params.d100,
            d0=calibration_params.d0,
        )

    if force_zero or not np.isfinite(distance):
        logger.warning(
            "Classic scoring forced to zero (force_zero=%s, distance=%s)", force_zero, distance
        )
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float(distance),
            status=status,
            reason=reason,
            d100=calibration_params.d100,
            d0=calibration_params.d0,
        )

    calibrated_score = sigmoid_score(float(distance), calibration_params)
    logger.info(
        "Classic scoring calibrated: distance=%.6f score=%.2f", float(distance), calibrated_score
    )
    return ComputeScoringResult(
        dtw_score=calibrated_score,
        distance=float(distance),
        status=status,
        reason=reason,
        d100=calibration_params.d100,
        d0=calibration_params.d0,
    )


def compute_profile_calibrated_scoring_result(
    profile: AnchorDistanceProfile,
    calibration_params: MultiAnchorSigmoidParams,
    status: str = "ok",
    reason: str = "",
) -> ScoringResult:
    if status != "ok":
        logger.warning("Classic multi-anchor scoring forced to zero because status=%s", status)
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float(profile.perfect_distance),
            status=status,
            reason=reason,
            d100=calibration_params.d100,
            d0=calibration_params.d0,
            moderate_distance=profile.moderate_distance,
            fail_distance=profile.fail_distance,
        )

    if not profile.is_valid:
        logger.warning("Classic multi-anchor scoring got invalid profile: %s", profile)
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float(profile.perfect_distance),
            status="invalid_reference",
            reason="invalid_anchor_profile",
            d100=calibration_params.d100,
            d0=calibration_params.d0,
            moderate_distance=profile.moderate_distance,
            fail_distance=profile.fail_distance,
        )

    calibrated_score = score_from_anchor_profile(profile, calibration_params)
    logger.info(
        "Classic multi-anchor scoring calibrated: perfect=%.6f moderate=%.6f fail=%.6f score=%.2f",
        float(profile.perfect_distance),
        float(profile.moderate_distance),
        float(profile.fail_distance),
        calibrated_score,
    )
    return ComputeScoringResult(
        dtw_score=calibrated_score,
        distance=float(profile.perfect_distance),
        status=status,
        reason=reason,
        d100=calibration_params.d100,
        d0=calibration_params.d0,
        moderate_distance=profile.moderate_distance,
        fail_distance=profile.fail_distance,
    )


def compute_scoring_result_from_distance(
    distance: float,
    user_frames: int,
    reference_frames: int,
    status: str = "ok",
    reason: str = "",
    calibration_params: SigmoidCalibrationParams | None = None,
) -> ScoringResult:
    del user_frames
    del reference_frames

    if calibration_params is None:
        calibration_params = SigmoidCalibrationParams(
            d100=0.15,
            d0=0.45,
            a=25.944166,
            b=0.30,
            epsilon=0.02,
        )

    return compute_calibrated_scoring_result(
        distance=distance,
        calibration_params=calibration_params,
        status=status,
        reason=reason,
    )


def aggregate_scoring_results(results: list[ScoringResult]) -> ScoringResult:
    if not results:
        logger.warning("Classic scoring aggregation got empty results")
        return _build_scoring_result(
            score=0.0,
            distance=float("inf"),
            status="invalid_reference",
            reason="no_results",
        )

    aggregated_score = float(np.mean([result.dtw_score for result in results]))

    finite_distances = [result.distance for result in results if np.isfinite(result.distance)]
    aggregated_distance = float(np.mean(finite_distances)) if finite_distances else float("inf")

    logger.info(
        "Classic scoring aggregated %s results: score=%.2f distance=%s",
        len(results),
        aggregated_score,
        aggregated_distance,
    )

    return _build_scoring_result(
        score=aggregated_score,
        distance=aggregated_distance,
    )
