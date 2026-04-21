from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from asr.vosk import VoskError, check_expected_text_for_preprocessed_audio
from scoring.anchor_calibration import (
	describe_anchor_set,
	fit_sigmoid_from_anchor_distances,
	get_word_anchor_set,
	is_known_zero_anchor,
	median_or_default,
	normalize_word,
)

from .embedding_comparator import compare_embeddings
from .input_gate import validate_speech_signal
from .preprocessing import preprocess_audio
from .scorer import (
	ScoringResult,
	compute_calibrated_scoring_result,
	compute_raw_distance,
	compute_scoring_result,
)
from .wav2vec_extractor import DEFAULT_MODEL_NAME, extract_wav2vec_embeddings


DEFAULT_MAX_ANCHORS_PER_CLASS = 12
DEFAULT_RAW_DISTANCE_ALPHA = 0.65


@lru_cache(maxsize=1024)
def _extract_anchor_embeddings_cached(
	anchor_path: str,
	mtime_ns: int,
	file_size: int,
	model_name: str,
	device: str | None,
	hf_token: str | None,
) -> np.ndarray:
	del mtime_ns
	del file_size

	anchor_audio = preprocess_audio(anchor_path)
	anchor_embeddings = extract_wav2vec_embeddings(
		anchor_audio.samples,
		anchor_audio.sample_rate,
		model_name=model_name,
		device=device,
		hf_token=hf_token,
	)
	return anchor_embeddings.frame_embeddings


def _extract_anchor_embeddings(
	anchor_path: str,
	model_name: str,
	device: str | None,
	hf_token: str | None,
) -> np.ndarray:
	anchor_file = Path(anchor_path)
	stats = anchor_file.stat()
	return _extract_anchor_embeddings_cached(
		str(anchor_file.resolve()),
		int(stats.st_mtime_ns),
		int(stats.st_size),
		model_name,
		device,
		hf_token,
	)


def _collect_valid_anchor_paths(
	paths: tuple[str, ...],
	model_name: str,
	device: str | None,
	hf_token: str | None,
) -> list[str]:
	valid_paths: list[str] = []
	for path in paths:
		try:
			frame_embeddings = _extract_anchor_embeddings(
				anchor_path=path,
				model_name=model_name,
				device=device,
				hf_token=hf_token,
			)
		except Exception:
			continue

		if int(frame_embeddings.shape[0]) == 0:
			continue
		valid_paths.append(path)

	return valid_paths


def _compare_raw_distance(
	embeddings_a: np.ndarray,
	embeddings_b: np.ndarray,
	similarity_metric: str,
	sakoe_chiba_radius: int | None,
	raw_distance_alpha: float,
):
	comparison = compare_embeddings(
		user_embeddings=embeddings_a,
		reference_embeddings=embeddings_b,
		metric=similarity_metric,
		sakoe_chiba_radius=sakoe_chiba_radius,
	)
	raw_distance = compute_raw_distance(
		similarity=comparison.similarity,
		temporal_distance=comparison.temporal_distance,
		metric=comparison.metric,
		alpha=raw_distance_alpha,
	)
	return comparison, raw_distance


def _distance_between_anchor_paths(
	anchor_path_a: str,
	anchor_path_b: str,
	similarity_metric: str,
	model_name: str,
	device: str | None,
	hf_token: str | None,
	sakoe_chiba_radius: int | None,
	raw_distance_alpha: float,
) -> float:
	embeddings_a = _extract_anchor_embeddings(
		anchor_path=anchor_path_a,
		model_name=model_name,
		device=device,
		hf_token=hf_token,
	)
	embeddings_b = _extract_anchor_embeddings(
		anchor_path=anchor_path_b,
		model_name=model_name,
		device=device,
		hf_token=hf_token,
	)

	_, raw_distance = _compare_raw_distance(
		embeddings_a=embeddings_a,
		embeddings_b=embeddings_b,
		similarity_metric=similarity_metric,
		sakoe_chiba_radius=sakoe_chiba_radius,
		raw_distance_alpha=raw_distance_alpha,
	)
	return float(raw_distance)


def _build_anchor_calibration(
	word: str,
	similarity_metric: str,
	model_name: str,
	device: str | None,
	hf_token: str | None,
	sakoe_chiba_radius: int | None,
	raw_distance_alpha: float,
	max_anchors_per_class: int,
	anchor_root: str | None,
):
	anchor_set = get_word_anchor_set(
		word=word,
		anchor_root=anchor_root,
		max_anchors_per_class=max_anchors_per_class,
	)

	valid_perfect_paths = _collect_valid_anchor_paths(
		paths=anchor_set.perfect_paths,
		model_name=model_name,
		device=device,
		hf_token=hf_token,
	)
	valid_zero_paths = _collect_valid_anchor_paths(
		paths=anchor_set.zero_paths,
		model_name=model_name,
		device=device,
		hf_token=hf_token,
	)

	if not valid_perfect_paths:
		raise ValueError("missing_perfect_anchors")
	if not valid_zero_paths:
		raise ValueError("missing_zero_anchors")

	distances_100: list[float] = []
	for left_idx in range(len(valid_perfect_paths)):
		for right_idx in range(left_idx + 1, len(valid_perfect_paths)):
			distance = _distance_between_anchor_paths(
				valid_perfect_paths[left_idx],
				valid_perfect_paths[right_idx],
				similarity_metric=similarity_metric,
				model_name=model_name,
				device=device,
				hf_token=hf_token,
				sakoe_chiba_radius=sakoe_chiba_radius,
				raw_distance_alpha=raw_distance_alpha,
			)
			if np.isfinite(distance):
				distances_100.append(float(distance))

	if not distances_100:
		distances_100 = [0.0]

	distances_0: list[float] = []
	for perfect_path in valid_perfect_paths:
		for zero_path in valid_zero_paths:
			distance = _distance_between_anchor_paths(
				perfect_path,
				zero_path,
				similarity_metric=similarity_metric,
				model_name=model_name,
				device=device,
				hf_token=hf_token,
				sakoe_chiba_radius=sakoe_chiba_radius,
				raw_distance_alpha=raw_distance_alpha,
			)
			if np.isfinite(distance):
				distances_0.append(float(distance))

	if not distances_0:
		raise ValueError("invalid_zero_anchor_distances")

	calibration_params = fit_sigmoid_from_anchor_distances(
		distances_100=distances_100,
		distances_0=distances_0,
		epsilon=0.02,
	)

	return anchor_set, calibration_params, valid_perfect_paths, valid_zero_paths


def _resolve_similarity_metric(metric: str) -> str:
	metric_key = metric.strip().lower()
	if metric_key != "cosine":
		raise ValueError("Only 'cosine' metric is supported")
	return metric_key


def analyze(
	user_audio_path: str,
	transcript: str,
	similarity: str = "cosine",
	model_name: str = DEFAULT_MODEL_NAME,
	device: str | None = None,
	hf_token: str | None = None,
	sakoe_chiba_radius: int | None = None,
	use_vosk: bool = True,
	raw_distance_alpha: float = DEFAULT_RAW_DISTANCE_ALPHA,
	max_anchors_per_class: int = DEFAULT_MAX_ANCHORS_PER_CLASS,
	anchor_root: str | None = None,
) -> ScoringResult:
	metric_key = _resolve_similarity_metric(similarity)
	normalized_word = normalize_word(transcript)
	if not normalized_word:
		return compute_scoring_result(
			similarity=0.0,
			temporal_distance=float("inf"),
			metric=metric_key,
			model_name=model_name,
			status="invalid_reference",
			reason="empty_transcript",
		)

	user_audio = preprocess_audio(user_audio_path)
	speech_gate = validate_speech_signal(
		user_audio.samples,
		sample_rate=user_audio.sample_rate,
	)
	if not speech_gate.passed:
		return compute_scoring_result(
			similarity=0.0,
			temporal_distance=float("inf"),
			metric=metric_key,
			model_name=model_name,
			status="empty_audio",
			reason="insufficient_speech",
		)

	if use_vosk:
		try:
			transcript_check = check_expected_text_for_preprocessed_audio(
				samples=user_audio.samples,
				sample_rate=user_audio.sample_rate,
				expected_text=transcript,
			)
		except (VoskError, ValueError) as exc:
			return compute_scoring_result(
				similarity=0.0,
				temporal_distance=float("inf"),
				metric=metric_key,
				model_name=model_name,
				status="asr_error",
				reason=f"vosk_failure:{exc}",
			)

		if not transcript_check.is_match:
			reason = (
				f"recognized:{transcript_check.recognized_text};"
				f"expected:{transcript_check.expected_text}"
			)
			return compute_scoring_result(
				similarity=0.0,
				temporal_distance=float("inf"),
				metric=metric_key,
				model_name=model_name,
				status="wrong_word",
				reason=reason,
			)

	try:
		anchor_set, calibration_params, valid_perfect_paths, valid_zero_paths = _build_anchor_calibration(
			word=normalized_word,
			similarity_metric=metric_key,
			model_name=model_name,
			device=device,
			hf_token=hf_token,
			sakoe_chiba_radius=sakoe_chiba_radius,
			raw_distance_alpha=raw_distance_alpha,
			max_anchors_per_class=max_anchors_per_class,
			anchor_root=anchor_root,
		)
	except ValueError as exc:
		return compute_scoring_result(
			similarity=0.0,
			temporal_distance=float("inf"),
			metric=metric_key,
			model_name=model_name,
			status="invalid_reference",
			reason=str(exc),
		)

	user_embeddings = extract_wav2vec_embeddings(
		user_audio.samples,
		user_audio.sample_rate,
		model_name=model_name,
		device=device,
		hf_token=hf_token,
	)
	if int(user_embeddings.frame_embeddings.shape[0]) == 0:
		return compute_scoring_result(
			similarity=0.0,
			temporal_distance=float("inf"),
			metric=metric_key,
			model_name=model_name,
			status="empty_audio",
			reason="empty_embeddings",
		)

	user_similarity_to_perfect: list[float] = []
	user_temporal_to_perfect: list[float] = []
	user_raw_to_perfect: list[float] = []

	for perfect_path in valid_perfect_paths:
		perfect_embeddings = _extract_anchor_embeddings(
			anchor_path=perfect_path,
			model_name=model_name,
			device=device,
			hf_token=hf_token,
		)
		comparison, raw_distance = _compare_raw_distance(
			embeddings_a=user_embeddings.frame_embeddings,
			embeddings_b=perfect_embeddings,
			similarity_metric=metric_key,
			sakoe_chiba_radius=sakoe_chiba_radius,
			raw_distance_alpha=raw_distance_alpha,
		)
		if np.isfinite(raw_distance):
			user_similarity_to_perfect.append(float(comparison.similarity))
			user_temporal_to_perfect.append(float(comparison.temporal_distance))
			user_raw_to_perfect.append(float(raw_distance))

	if not user_raw_to_perfect:
		return compute_scoring_result(
			similarity=0.0,
			temporal_distance=float("inf"),
			metric=metric_key,
			model_name=model_name,
			status="invalid_reference",
			reason="invalid_perfect_anchor_distances",
		)

	user_raw_to_zero: list[float] = []
	for zero_path in valid_zero_paths:
		zero_embeddings = _extract_anchor_embeddings(
			anchor_path=zero_path,
			model_name=model_name,
			device=device,
			hf_token=hf_token,
		)
		_, raw_distance = _compare_raw_distance(
			embeddings_a=user_embeddings.frame_embeddings,
			embeddings_b=zero_embeddings,
			similarity_metric=metric_key,
			sakoe_chiba_radius=sakoe_chiba_radius,
			raw_distance_alpha=raw_distance_alpha,
		)
		if np.isfinite(raw_distance):
			user_raw_to_zero.append(float(raw_distance))

	aggregated_similarity = float(np.mean(user_similarity_to_perfect))
	aggregated_temporal_distance = float(np.mean(user_temporal_to_perfect))
	user_raw_distance = median_or_default(user_raw_to_perfect, default=float("inf"))
	user_zero_distance = median_or_default(user_raw_to_zero, default=float("inf"))

	force_zero = False
	reason = ""
	if is_known_zero_anchor(user_audio_path, anchor_set):
		force_zero = True
		reason = "known_zero_anchor"
	elif np.isfinite(user_zero_distance) and np.isfinite(user_raw_distance) and user_zero_distance <= user_raw_distance:
		force_zero = True
		reason = "closer_to_zero_anchors"

	result = compute_calibrated_scoring_result(
		similarity=aggregated_similarity,
		temporal_distance=aggregated_temporal_distance,
		metric=metric_key,
		model_name=model_name,
		raw_distance=user_raw_distance,
		calibration_params=calibration_params,
		status="ok",
		reason=reason,
		force_zero=force_zero,
	)

	if not result.reason:
		anchor_stats = describe_anchor_set(anchor_set)
		result.reason = (
			f"anchors:perfect={anchor_stats['perfect']};"
			f"wrong={anchor_stats['wrong']};"
			f"moderate={anchor_stats['moderate']};"
			f"empty={anchor_stats['empty_word']}"
		)

	return result
 