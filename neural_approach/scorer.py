from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ScoringResult:
	pronunciation_score: float
	similarity: float
	temporal_distance: float
	problematic_phonemes: List[str]
	verdict: str
	metric: str
	model_name: str


def _similarity_to_quality(similarity: float, metric: str) -> float:
	metric_key = metric.strip().lower()
	if metric_key == "cosine":
		# Map cosine from [-1, 1] to [0, 1] for stable score fusion.
		quality = (float(similarity) + 1.0) / 2.0
	else:
		quality = float(similarity)
	return float(np.clip(quality, 0.0, 1.0))


def _temporal_distance_to_quality(distance: float) -> float:
	if not np.isfinite(distance):
		return 0.0
	# Exponential decay: small DTW distances stay near 1.0, bad alignments drop quickly.
	quality = np.exp(-3.5 * max(float(distance), 0.0))
	return float(np.clip(quality, 0.0, 1.0))


def compute_scoring_result(
	similarity: float,
	temporal_distance: float,
	metric: str,
	model_name: str,
	phoneme_issues: list[str] | None = None,
) -> ScoringResult:
	similarity_quality = _similarity_to_quality(similarity, metric)
	temporal_quality = _temporal_distance_to_quality(temporal_distance)

	# Keep global pronunciation similarity primary, but still account for time alignment.
	score = 100.0 * (0.7 * similarity_quality + 0.3 * temporal_quality)
	score = float(np.clip(score, 0.0, 100.0))

	if score >= 80.0:
		verdict = "good"
	elif score >= 60.0:
		verdict = "acceptable"
	else:
		verdict = "needs_improvement"

	return ScoringResult(
		pronunciation_score=score,
		similarity=float(similarity),
		temporal_distance=float(temporal_distance),
		problematic_phonemes=list(phoneme_issues or []),
		verdict=verdict,
		metric=metric.strip().lower(),
		model_name=model_name,
	)
