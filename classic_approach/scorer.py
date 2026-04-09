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



@dataclass
class ScoringResult:
    dtw_score: float
    problematic_phonemes: List[str]
    verdict: str
    distance: float
    error_localization: List[dict[str, Any]]


def ComputeScoringResult(
    dtw_score,
    phoneme_issues,
    distance,
    error_localization=None,
) -> ScoringResult:
    score = max(0.0, min(100.0, float(dtw_score)))
    if score >= 80.0:
        verdict = "хорошо"
    elif score >= 60.0:
        verdict = "удовлетворительно"
    else:
        verdict = "неудовлетворительно"

    return ScoringResult(
        dtw_score=score,
        problematic_phonemes=list(phoneme_issues or []),
        verdict=verdict,
        distance=distance,
        error_localization=list(error_localization or []),
    )