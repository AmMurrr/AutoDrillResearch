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
		# Переводим cosine из диапазона [-1, 1] в [0, 1] для стабильного объединения метрик.
		quality = (float(similarity) + 1.0) / 2.0
	else:
		quality = float(similarity)
	return float(np.clip(quality, 0.0, 1.0))


def _temporal_distance_to_quality(distance: float) -> float:
	if not np.isfinite(distance):
		return 0.0
	# Экспоненциальный спад: малые DTW-дистанции дают качество близкое к 1.0,
	# а плохое выравнивание быстро снижает итог.
	quality = np.exp(-3.5 * max(float(distance), 0.0))
	return float(np.clip(quality, 0.0, 1.0))


def _duration_penalty(user_frames: int | None, reference_frames: int | None) -> float:
	"""Штраф за существенное расхождение длительностей попытки и эталона."""
	if user_frames is None or reference_frames is None:
		return 1.0

	if user_frames <= 0 or reference_frames <= 0:
		return 0.0

	frame_ratio = max(user_frames, reference_frames) / max(1, min(user_frames, reference_frames))
	log_ratio = abs(np.log(frame_ratio))

	# Плавный штраф при умеренной разнице и более сильный при явной растяжке.
	base_penalty = np.exp(-0.6 * (log_ratio**2))
	long_utterance_penalty = np.exp(-0.35 * max(frame_ratio - 3.0, 0.0))

	return float(np.clip(base_penalty * long_utterance_penalty, 0.0, 1.0))


def compute_scoring_result(
	similarity: float,
	temporal_distance: float,
	metric: str,
	model_name: str,
	phoneme_issues: list[str] | None = None,
	user_frames: int | None = None,
	reference_frames: int | None = None,
) -> ScoringResult:
	similarity_quality = _similarity_to_quality(similarity, metric)
	temporal_quality = _temporal_distance_to_quality(temporal_distance)
	duration_quality = _duration_penalty(user_frames, reference_frames)

	# Основной вклад даёт общее сходство произношения,
	# но временное выравнивание тоже учитывается.
	base_score = 100.0 * (0.7 * similarity_quality + 0.3 * temporal_quality)
	score = base_score * duration_quality
	score = float(np.clip(score, 0.0, 100.0))

	if score >= 80.0:
		verdict = "хорошо"
	elif score >= 60.0:
		verdict = "удовлетворительно"
	else:
		verdict = "неудовлетворительно"

	return ScoringResult(
		pronunciation_score=score,
		similarity=float(similarity),
		temporal_distance=float(temporal_distance),
		problematic_phonemes=list(phoneme_issues or []),
		verdict=verdict,
		metric=metric.strip().lower(),
		model_name=model_name,
	)
