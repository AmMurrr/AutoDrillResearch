from __future__ import annotations

import numpy as np

from .embedding_comparator import compare_embeddings
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


def _normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	return matrix / np.maximum(norms, eps)


def _aligned_low_quantile_cosine(
	user_embeddings: np.ndarray,
	reference_embeddings: np.ndarray,
	quantile: float = 0.10,
) -> float:
	user = np.asarray(user_embeddings, dtype=np.float32)
	reference = np.asarray(reference_embeddings, dtype=np.float32)
	if user.ndim != 2 or reference.ndim != 2:
		return 0.0
	if user.shape[0] == 0 or reference.shape[0] == 0:
		return 0.0

	user = _normalize_rows(user)
	reference = _normalize_rows(reference)

	n_points = max(user.shape[0], reference.shape[0])
	user_idx = np.clip(
		np.round(np.linspace(0, user.shape[0] - 1, num=n_points)).astype(np.int64),
		0,
		user.shape[0] - 1,
	)
	reference_idx = np.clip(
		np.round(np.linspace(0, reference.shape[0] - 1, num=n_points)).astype(np.int64),
		0,
		reference.shape[0] - 1,
	)

	aligned_cosine = np.sum(user[user_idx] * reference[reference_idx], axis=1)
	q = float(np.clip(quantile, 0.0, 1.0))
	return float(np.quantile(aligned_cosine, q))


def _detect_word_mismatch_issue(
	metric: str,
	pooled_similarity: float,
	user_embeddings: np.ndarray,
	reference_embeddings: np.ndarray,
) -> list[str]:
	if metric != "cosine":
		return []

	low_quantile = _aligned_low_quantile_cosine(user_embeddings, reference_embeddings, quantile=0.10)
	consistency_gap = float(pooled_similarity) - low_quantile

	# Подозрение на другое слово: глобально высокое сходство при провалах в локальной
	# согласованности по кадрам.
	if float(pooled_similarity) >= 0.95 and consistency_gap >= 0.50:
		return ["word:possible_mismatch"]

	return []


def _analyze_against_single_reference(
	user_frame_embeddings: np.ndarray,
	reference_audio_path: str,
	similarity: str,
	model_name: str,
	device: str | None,
	hf_token: str | None,
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
	)

	phoneme_issues = _detect_word_mismatch_issue(
		metric=comparison.metric,
		pooled_similarity=comparison.similarity,
		user_embeddings=user_frame_embeddings,
		reference_embeddings=reference_embeddings.frame_embeddings,
	)

	return compute_scoring_result(
		similarity=comparison.similarity,
		temporal_distance=comparison.temporal_distance,
		metric=comparison.metric,
		model_name=model_name,
		phoneme_issues=phoneme_issues,
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
) -> ScoringResult:
	# transcript пока не используется в логике, оставлен для будущей
	# диагностики по словам/фонемам.
	_ = transcript

	user_audio = preprocess_audio(user_audio_path)
	reference_paths = _resolve_reference_paths(reference_audio_path)
	if not reference_paths:
		return compute_scoring_result(
			similarity=0.0,
			temporal_distance=float("inf"),
			metric=similarity,
			model_name=model_name,
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
			metric=similarity,
			model_name=model_name,
		)

	per_reference_results = [
		_analyze_against_single_reference(
			user_frame_embeddings=user_embeddings.frame_embeddings,
			reference_audio_path=path,
			similarity=similarity,
			model_name=model_name,
			device=device,
			hf_token=hf_token,
		)
		for path in reference_paths
	]

	return aggregate_scoring_results(per_reference_results)
 