from dataclasses import dataclass

import numpy as np


def _distance_to_score(distance: float, user_frames: int, reference_frames: int) -> float:
    DISTANCE_MID = 0.25
    SHAPE = 20.0
    base_score = 100.0 / (1.0 + (max(distance, 0.0) / DISTANCE_MID) ** SHAPE)

    frame_ratio = max(user_frames, reference_frames) / max(1, min(user_frames, reference_frames))
    log_ratio = abs(np.log(frame_ratio))
    duration_penalty = np.exp(-0.8 * (log_ratio**2))

    # Дополнительный штраф для очень длинных попыток (частый признак нецелевой речи).
    long_utterance_penalty = np.exp(-0.35 * max(frame_ratio - 3.0, 0.0))
    duration_penalty = float(np.clip(duration_penalty * long_utterance_penalty, 0.0, 1.0))

    return float(np.clip(base_score * duration_penalty, 0.0, 100.0))


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


def ComputeScoringResult(
    dtw_score: float,
    distance: float,
    status: str = "ok",
    reason: str = "",
) -> ScoringResult:
    return _build_scoring_result(
        score=float(dtw_score),
        distance=distance,
        status=status,
        reason=reason,
    )


def compute_scoring_result_from_distance(
    distance: float,
    user_frames: int,
    reference_frames: int,
    status: str = "ok",
    reason: str = "",
) -> ScoringResult:
    if status != "ok":
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float(distance),
            status=status,
            reason=reason,
        )

    if not np.isfinite(distance):
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float("inf"),
            status=status,
            reason=reason,
        )

    dtw_score = _distance_to_score(float(distance), int(user_frames), int(reference_frames))
    return ComputeScoringResult(
        dtw_score=dtw_score,
        distance=float(distance),
        status=status,
        reason=reason,
    )


def aggregate_scoring_results(results: list[ScoringResult]) -> ScoringResult:
    if not results:
        return _build_scoring_result(
            score=0.0,
            distance=float("inf"),
            status="invalid_reference",
            reason="no_results",
        )

    aggregated_score = float(np.mean([result.dtw_score for result in results]))

    finite_distances = [result.distance for result in results if np.isfinite(result.distance)]
    aggregated_distance = float(np.mean(finite_distances)) if finite_distances else float("inf")

    return _build_scoring_result(
        score=aggregated_score,
        distance=aggregated_distance,
    )