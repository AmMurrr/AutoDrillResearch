from dataclasses import dataclass
from typing import List



@dataclass
class ScoringResult:
    dtw_score: float
    problematic_phonemes: List[str]
    verdict: str


def ComputeScoringResult(dtw_score, phoneme_issues) -> ScoringResult:
    pass