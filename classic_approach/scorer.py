from dataclasses import dataclass
from typing import Any, List



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