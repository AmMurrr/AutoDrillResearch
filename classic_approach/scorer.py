from dataclasses import dataclass
from typing import List



@dataclass
class ScoringResult:
    dtw_score: float
    problematic_phonemes: List[str]
    verdict: str


def ComputeScoringResult(dtw_score, phoneme_issues) -> ScoringResult:
    score = max(0.0, min(100.0, float(dtw_score)))
    if score >= 80.0:
        verdict = "good"
    elif score >= 60.0:
        verdict = "acceptable"
    else:
        verdict = "needs_improvement"

    return ScoringResult(
        dtw_score=score,
        problematic_phonemes=list(phoneme_issues or []),
        verdict=verdict,
    )