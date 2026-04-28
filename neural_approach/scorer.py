from __future__ import annotations

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


@dataclass
class ScoringResult:
    pronunciation_score: float
    similarity: float
    temporal_distance: float
    verdict: str
    metric: str
    model_name: str
    status: str = "ok"
    reason: str = ""
    raw_distance: float = float("inf")
    d100: float | None = None
    d0: float | None = None
    moderate_raw_distance: float | None = None
    fail_raw_distance: float | None = None
    duration_distance: float | None = None
    duration_score: float | None = None
    embedding_score: float | None = None


def _verdict_from_score(score: float) -> str:
    if score >= 80.0:
        return "хорошо"
    if score >= 60.0:
        return "удовлетворительно"
    return "неудовлетворительно"


def _build_scoring_result(
    pronunciation_score: float,
    similarity: float,
    temporal_distance: float,
    metric: str,
    model_name: str,
    status: str = "ok",
    reason: str = "",
    raw_distance: float = float("inf"),
    d100: float | None = None,
    d0: float | None = None,
    moderate_raw_distance: float | None = None,
    fail_raw_distance: float | None = None,
) -> ScoringResult:
    clipped_score = float(np.clip(float(pronunciation_score), 0.0, 100.0))
    metric_key = metric.strip().lower()
    return ScoringResult(
        pronunciation_score=clipped_score,
        similarity=float(similarity),
        temporal_distance=float(temporal_distance),
        verdict=_verdict_from_score(clipped_score),
        metric=metric_key,
        model_name=model_name,
        status=status,
        reason=reason,
        raw_distance=float(raw_distance),
        d100=d100,
        d0=d0,
        moderate_raw_distance=moderate_raw_distance,
        fail_raw_distance=fail_raw_distance,
    )


def _resolve_metric(metric: str) -> str:
    metric_key = metric.strip().lower()
    if metric_key != "cosine":
        raise ValueError("Only 'cosine' metric is supported")
    return metric_key


def _semantic_distance_from_similarity(similarity: float, metric: str) -> float:
    metric_key = _resolve_metric(metric)
    if metric_key != "cosine":
        raise ValueError("Only 'cosine' metric is supported")
    cosine_quality = (float(similarity) + 1.0) / 2.0
    cosine_quality = float(np.clip(cosine_quality, 0.0, 1.0))
    return float(1.0 - cosine_quality)


def compute_raw_distance(
    similarity: float,
    temporal_distance: float,
    metric: str,
    alpha: float = 0.65,
) -> float:
    metric_key = _resolve_metric(metric)
    if metric_key != "cosine":
        raise ValueError("Only 'cosine' metric is supported")

    temporal = float(max(float(temporal_distance), 0.0))
    if not np.isfinite(temporal):
        return float("inf")

    weight = float(np.clip(float(alpha), 0.0, 1.0))
    semantic = _semantic_distance_from_similarity(float(similarity), metric_key)
    return float(weight * temporal + (1.0 - weight) * semantic)


def _default_calibration_params() -> SigmoidCalibrationParams:
    return SigmoidCalibrationParams(
        d100=0.15,
        d0=0.55,
        a=19.459101,
        b=0.35,
        epsilon=0.02,
    )


def compute_calibrated_scoring_result(
    similarity: float,
    temporal_distance: float,
    metric: str,
    model_name: str,
    raw_distance: float,
    calibration_params: SigmoidCalibrationParams,
    status: str = "ok",
    reason: str = "",
    force_zero: bool = False,
) -> ScoringResult:
    if status != "ok":
        logger.warning("Neural scoring forced to zero because status=%s", status)
        return _build_scoring_result(
            pronunciation_score=0.0,
            similarity=float(similarity),
            temporal_distance=float(temporal_distance),
            metric=metric,
            model_name=model_name,
            status=status,
            reason=reason,
            raw_distance=float(raw_distance),
            d100=calibration_params.d100,
            d0=calibration_params.d0,
        )

    if force_zero or not np.isfinite(raw_distance):
        logger.warning(
            "Neural scoring forced to zero (force_zero=%s, raw_distance=%s)",
            force_zero,
            raw_distance,
        )
        return _build_scoring_result(
            pronunciation_score=0.0,
            similarity=float(similarity),
            temporal_distance=float(temporal_distance),
            metric=metric,
            model_name=model_name,
            status=status,
            reason=reason,
            raw_distance=float(raw_distance),
            d100=calibration_params.d100,
            d0=calibration_params.d0,
        )

    score = sigmoid_score(float(raw_distance), calibration_params)
    logger.info(
        "Neural scoring calibrated: raw_distance=%.6f score=%.2f", float(raw_distance), score
    )
    return _build_scoring_result(
        pronunciation_score=score,
        similarity=float(similarity),
        temporal_distance=float(temporal_distance),
        metric=metric,
        model_name=model_name,
        status=status,
        reason=reason,
        raw_distance=float(raw_distance),
        d100=calibration_params.d100,
        d0=calibration_params.d0,
    )


def compute_anchor_profile_calibrated_scoring_result(
    similarity: float,
    temporal_distance: float,
    metric: str,
    model_name: str,
    profile: AnchorDistanceProfile,
    calibration_params: MultiAnchorSigmoidParams,
    status: str = "ok",
    reason: str = "",
) -> ScoringResult:
    if status != "ok":
        logger.warning("Neural multi-anchor scoring forced to zero because status=%s", status)
        return _build_scoring_result(
            pronunciation_score=0.0,
            similarity=float(similarity),
            temporal_distance=float(temporal_distance),
            metric=metric,
            model_name=model_name,
            status=status,
            reason=reason,
            raw_distance=float(profile.perfect_distance),
            d100=calibration_params.d100,
            d0=calibration_params.d0,
            moderate_raw_distance=profile.moderate_distance,
            fail_raw_distance=profile.fail_distance,
        )

    if not profile.is_valid:
        logger.warning("Neural multi-anchor scoring got invalid profile: %s", profile)
        return _build_scoring_result(
            pronunciation_score=0.0,
            similarity=float(similarity),
            temporal_distance=float(temporal_distance),
            metric=metric,
            model_name=model_name,
            status="invalid_reference",
            reason="invalid_anchor_profile",
            raw_distance=float(profile.perfect_distance),
            d100=calibration_params.d100,
            d0=calibration_params.d0,
            moderate_raw_distance=profile.moderate_distance,
            fail_raw_distance=profile.fail_distance,
        )

    score = score_from_anchor_profile(profile, calibration_params)
    logger.info(
        "Neural multi-anchor scoring calibrated: perfect=%.6f moderate=%.6f fail=%.6f score=%.2f",
        float(profile.perfect_distance),
        float(profile.moderate_distance),
        float(profile.fail_distance),
        score,
    )
    return _build_scoring_result(
        pronunciation_score=score,
        similarity=float(similarity),
        temporal_distance=float(temporal_distance),
        metric=metric,
        model_name=model_name,
        status=status,
        reason=reason,
        raw_distance=float(profile.perfect_distance),
        d100=calibration_params.d100,
        d0=calibration_params.d0,
        moderate_raw_distance=profile.moderate_distance,
        fail_raw_distance=profile.fail_distance,
    )


def compute_scoring_result(
    similarity: float,
    temporal_distance: float,
    metric: str,
    model_name: str,
    user_frames: int | None = None,
    reference_frames: int | None = None,
    status: str = "ok",
    reason: str = "",
    alpha: float = 0.65,
) -> ScoringResult:
    del user_frames
    del reference_frames

    raw_distance = compute_raw_distance(
        similarity=float(similarity),
        temporal_distance=float(temporal_distance),
        metric=metric,
        alpha=alpha,
    )

    return compute_calibrated_scoring_result(
        similarity=float(similarity),
        temporal_distance=float(temporal_distance),
        metric=metric,
        model_name=model_name,
        raw_distance=raw_distance,
        calibration_params=_default_calibration_params(),
        status=status,
        reason=reason,
    )


def blend_with_duration_score(
    result: ScoringResult,
    duration_distance: float,
    duration_score: float,
    duration_weight: float = 0.85,
) -> ScoringResult:
    result.duration_distance = float(duration_distance)
    result.duration_score = float(duration_score)

    if result.status != "ok":
        return result

    weight = float(np.clip(float(duration_weight), 0.0, 1.0))
    embedding_score = float(result.pronunciation_score)
    result.embedding_score = embedding_score
    blended_score = weight * float(duration_score) + (1.0 - weight) * embedding_score
    result.pronunciation_score = float(np.clip(blended_score, 0.0, 100.0))
    result.verdict = _verdict_from_score(result.pronunciation_score)
    logger.info(
        "Neural duration-aware score blended: embedding_score=%.2f duration_score=%.2f final=%.2f",
        embedding_score,
        float(duration_score),
        result.pronunciation_score,
    )
    return result


def aggregate_scoring_results(results: list[ScoringResult]) -> ScoringResult:
    if not results:
        logger.warning("Neural scoring aggregation got empty results")
        return _build_scoring_result(
            pronunciation_score=0.0,
            similarity=0.0,
            temporal_distance=float("inf"),
            metric="cosine",
            model_name="unknown",
            status="invalid_reference",
            reason="no_results",
        )

    aggregated_score = float(np.mean([result.pronunciation_score for result in results]))
    aggregated_similarity = float(np.mean([result.similarity for result in results]))
    finite_raw_distances = [
        result.raw_distance for result in results if np.isfinite(result.raw_distance)
    ]
    aggregated_raw_distance = (
        float(np.mean(finite_raw_distances)) if finite_raw_distances else float("inf")
    )

    finite_temporal_distances = [
        result.temporal_distance for result in results if np.isfinite(result.temporal_distance)
    ]
    aggregated_temporal_distance = (
        float(np.mean(finite_temporal_distances)) if finite_temporal_distances else float("inf")
    )

    logger.info(
        "Neural scoring aggregated %s results: score=%.2f raw_distance=%s",
        len(results),
        aggregated_score,
        aggregated_raw_distance,
    )

    first_result = results[0]
    return _build_scoring_result(
        pronunciation_score=aggregated_score,
        similarity=aggregated_similarity,
        temporal_distance=aggregated_temporal_distance,
        metric=first_result.metric,
        model_name=first_result.model_name,
        raw_distance=aggregated_raw_distance,
        d100=first_result.d100,
        d0=first_result.d0,
    )
