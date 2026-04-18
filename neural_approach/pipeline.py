from __future__ import annotations

import numpy as np
from asr.vosk import VoskError, check_expected_text_for_preprocessed_audio

from .embedding_comparator import compare_embeddings
from .input_gate import validate_speech_signal
from .preprocessing import preprocess_audio
from .scorer import ScoringResult, aggregate_scoring_results, compute_scoring_result
from .wav2vec_extractor import DEFAULT_MODEL_NAME, extract_wav2vec_embeddings


def _resolve_reference_paths(reference_audio_path: str | list[str]) -> list[str]:
	if isinstance(reference_audio_path, str):
		path = reference_audio_path.strip()
		return [path] if path else []

	paths: list[str] = []
	for path in reference_audio_path:
		normalized = str(path).strip()
		if normalized:
			paths.append(normalized)
	return paths


def _resolve_similarity_metric(metric: str) -> str:
	metric_key = metric.strip().lower()
	if metric_key != "cosine":
		raise ValueError("Only 'cosine' metric is supported")
	return metric_key


def _analyze_against_single_reference(
	user_frame_embeddings: np.ndarray,
	reference_audio_path: str,
	similarity: str,
	model_name: str,
	device: str | None,
	hf_token: str | None,
	sakoe_chiba_radius: int | None,
) -> ScoringResult:
	reference_audio = preprocess_audio(reference_audio_path)
	reference_embeddings = extract_wav2vec_embeddings(
		reference_audio.samples,
		reference_audio.sample_rate,
		model_name=model_name,
		device=device,
		hf_token=hf_token,
	)

	comparison = compare_embeddings(
		user_embeddings=user_frame_embeddings,
		reference_embeddings=reference_embeddings.frame_embeddings,
		metric=similarity,
		sakoe_chiba_radius=sakoe_chiba_radius,
	)

	return compute_scoring_result(
		similarity=comparison.similarity,
		temporal_distance=comparison.temporal_distance,
		metric=comparison.metric,
		model_name=model_name,
		user_frames=int(user_frame_embeddings.shape[0]),
		reference_frames=int(reference_embeddings.frame_embeddings.shape[0]),
	)


def analyze(
	user_audio_path: str,
	reference_audio_path: str | list[str],
	transcript: str,
	similarity: str = "cosine",
	model_name: str = DEFAULT_MODEL_NAME,
	device: str | None = None,
	hf_token: str | None = None,
	sakoe_chiba_radius: int | None = None,
	use_vosk: bool = True,
) -> ScoringResult:
	metric_key = _resolve_similarity_metric(similarity)

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

	reference_paths = _resolve_reference_paths(reference_audio_path)
	if not reference_paths:
		return compute_scoring_result(
			similarity=0.0,
			temporal_distance=float("inf"),
			metric=metric_key,
			model_name=model_name,
			status="invalid_reference",
			reason="no_reference_paths",
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

	per_reference_results = [
		_analyze_against_single_reference(
			user_frame_embeddings=user_embeddings.frame_embeddings,
			reference_audio_path=path,
			similarity=metric_key,
			model_name=model_name,
			device=device,
			hf_token=hf_token,
			sakoe_chiba_radius=sakoe_chiba_radius,
		)
		for path in reference_paths
	]

	return aggregate_scoring_results(per_reference_results)
 