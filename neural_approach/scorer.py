from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
	)


def _similarity_to_quality(similarity: float, metric: str) -> float:
	metric_key = metric.strip().lower()
	if metric_key != "cosine":
		raise ValueError("Only 'cosine' metric is supported")
	# Переводим cosine из диапазона [-1, 1] в [0, 1] для стабильного объединения метрик.
	quality = (float(similarity) + 1.0) / 2.0
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
	user_frames: int | None = None,
	reference_frames: int | None = None,
	status: str = "ok",
	reason: str = "",
) -> ScoringResult:
	if status != "ok":
		return _build_scoring_result(
			pronunciation_score=0.0,
			similarity=float(similarity),
			temporal_distance=float(temporal_distance),
			metric=metric,
			model_name=model_name,
			status=status,
			reason=reason,
		)

	similarity_quality = _similarity_to_quality(similarity, metric)
	temporal_quality = _temporal_distance_to_quality(temporal_distance)
	duration_quality = _duration_penalty(user_frames, reference_frames)

	# Основной вклад даёт общее сходство произношения,
	# но временное выравнивание тоже учитывается.
	base_score = 100.0 * (0.7 * similarity_quality + 0.3 * temporal_quality)
	score = base_score * duration_quality
	score = float(np.clip(score, 0.0, 100.0))

	return _build_scoring_result(
		pronunciation_score=score,
		similarity=float(similarity),
		temporal_distance=float(temporal_distance),
		metric=metric,
		model_name=model_name,
		status=status,
		reason=reason,
	)


def aggregate_scoring_results(results: list[ScoringResult]) -> ScoringResult:
	if not results:
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

	finite_distances = [result.temporal_distance for result in results if np.isfinite(result.temporal_distance)]
	aggregated_temporal_distance = (
		float(np.mean(finite_distances)) if finite_distances else float("inf")
	)

	first_result = results[0]
	return _build_scoring_result(
		pronunciation_score=aggregated_score,
		similarity=aggregated_similarity,
		temporal_distance=aggregated_temporal_distance,
		metric=first_result.metric,
		model_name=first_result.model_name,
	)
