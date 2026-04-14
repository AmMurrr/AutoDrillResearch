from collections import Counter
from dataclasses import dataclass
from typing import Any, List

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


def _issue_penalty(phoneme_issues: list[str] | None) -> float:
    issues = list(phoneme_issues or [])
    if not issues:
        return 1.0

    # Локализационные аномалии по слову считаем сильным признаком нецелевого произношения.
    penalty = np.exp(-1.6 * len(set(issues)))
    return float(np.clip(penalty, 0.1, 1.0))


def _verdict_from_score(score: float) -> str:
    if score >= 80.0:
        return "хорошо"
    if score >= 60.0:
        return "удовлетворительно"
    return "неудовлетворительно"


def _build_scoring_result(
    score: float,
    phoneme_issues: list[str] | None,
    distance: float,
    error_localization: list[dict[str, Any]] | None,
    status: str = "ok",
    reason: str = "",
) -> "ScoringResult":
    clipped_score = float(np.clip(float(score), 0.0, 100.0))
    return ScoringResult(
        dtw_score=clipped_score,
        problematic_phonemes=list(phoneme_issues or []),
        verdict=_verdict_from_score(clipped_score),
        distance=float(distance),
        error_localization=list(error_localization or []),
        status=status,
        reason=reason,
    )



@dataclass
class ScoringResult:
    dtw_score: float
    problematic_phonemes: List[str]
    verdict: str
    distance: float
    error_localization: List[dict[str, Any]]
    status: str = "ok"
    reason: str = ""


def ComputeScoringResult(
    dtw_score,
    phoneme_issues,
    distance,
    error_localization=None,
    status: str = "ok",
    reason: str = "",
) -> ScoringResult:
    score = float(dtw_score) * _issue_penalty(phoneme_issues)
    return _build_scoring_result(
        score=score,
        phoneme_issues=phoneme_issues,
        distance=distance,
        error_localization=list(error_localization or []),
        status=status,
        reason=reason,
    )


def compute_scoring_result_from_distance(
    distance: float,
    user_frames: int,
    reference_frames: int,
    phoneme_issues: list[str] | None = None,
    error_localization: list[dict[str, Any]] | None = None,
    status: str = "ok",
    reason: str = "",
) -> ScoringResult:
    if not np.isfinite(distance):
        return ComputeScoringResult(
            dtw_score=0.0,
            phoneme_issues=phoneme_issues,
            distance=float("inf"),
            error_localization=error_localization,
            status=status,
            reason=reason,
        )

    dtw_score = _distance_to_score(float(distance), int(user_frames), int(reference_frames))
    return ComputeScoringResult(
        dtw_score=dtw_score,
        phoneme_issues=phoneme_issues,
        distance=float(distance),
        error_localization=error_localization,
        status=status,
        reason=reason,
    )


def aggregate_scoring_results(results: list[ScoringResult]) -> ScoringResult:
    if not results:
        return _build_scoring_result(
            score=0.0,
            phoneme_issues=[],
            distance=float("inf"),
            error_localization=[],
        )

    aggregated_score = float(np.mean([result.dtw_score for result in results]))

    finite_distances = [result.distance for result in results if np.isfinite(result.distance)]
    aggregated_distance = float(np.mean(finite_distances)) if finite_distances else float("inf")

    issue_counter: Counter[str] = Counter(
        issue for result in results for issue in result.problematic_phonemes
    )
    majority_threshold = (len(results) // 2) + 1
    aggregated_issues = sorted(
        issue for issue, count in issue_counter.items() if count >= majority_threshold
    )

    representative_result = min(
        results,
        key=lambda result: (
            not np.isfinite(result.distance),
            result.distance if np.isfinite(result.distance) else float("inf"),
        ),
    )

    return _build_scoring_result(
        score=aggregated_score,
        phoneme_issues=aggregated_issues,
        distance=aggregated_distance,
        error_localization=representative_result.error_localization,
    )