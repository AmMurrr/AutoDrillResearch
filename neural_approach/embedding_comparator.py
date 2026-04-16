from __future__ import annotations

from dataclasses import dataclass

import dtaidistance.dtw_ndim as dtw_ndim
import numpy as np


@dataclass
class EmbeddingComparisonResult:
	similarity: float
	temporal_distance: float
	metric: str


def _resolve_window(sakoe_chiba_radius: int | None) -> int | None:
	if sakoe_chiba_radius is None:
		return None

	radius = int(sakoe_chiba_radius)
	return radius if radius > 0 else None


def _resolve_metric(metric: str) -> str:
	metric_key = metric.strip().lower()
	if metric_key != "cosine":
		raise ValueError("Only 'cosine' metric is supported")
	return metric_key


def _as_frame_matrix(embeddings: np.ndarray) -> np.ndarray:
	arr = np.asarray(embeddings, dtype=np.float32)
	if arr.ndim != 2:
		raise ValueError("Expected frame embeddings with shape (n_frames, emb_dim)")
	return arr


def _l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	return matrix / np.maximum(norms, eps)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-8) -> float:
	denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
	if denom <= eps:
		return 0.0
	return float(np.dot(vec_a, vec_b) / denom)


def _pooled_embedding(frame_embeddings: np.ndarray) -> np.ndarray:
	return frame_embeddings.mean(axis=0, dtype=np.float32)


def _dtw_temporal_distance(
	embeddings_a: np.ndarray,
	embeddings_b: np.ndarray,
	sakoe_chiba_radius: int | None = None,
) -> float:
	seq_a = _l2_normalize_rows(_as_frame_matrix(embeddings_a)).astype(np.double)
	seq_b = _l2_normalize_rows(_as_frame_matrix(embeddings_b)).astype(np.double)

	n, m = seq_a.shape[0], seq_b.shape[0]
	if n == 0 or m == 0:
		return float("inf")

	feature_dim = max(1, seq_a.shape[1])
	window = _resolve_window(sakoe_chiba_radius)
	raw_distance = float(dtw_ndim.distance(seq_a, seq_b, window=window))
	return raw_distance / (max(n, m) * np.sqrt(feature_dim))


def compare_embeddings(
	user_embeddings: np.ndarray,
	reference_embeddings: np.ndarray,
	metric: str = "cosine",
	sakoe_chiba_radius: int | None = None,
) -> EmbeddingComparisonResult:
	metric_key = _resolve_metric(metric)

	user_frames = _as_frame_matrix(user_embeddings)
	ref_frames = _as_frame_matrix(reference_embeddings)

	user_pooled = _pooled_embedding(user_frames)
	ref_pooled = _pooled_embedding(ref_frames)

	# Косинусное сходство естественно лежит в диапазоне [-1, 1].
	similarity = _cosine_similarity(user_pooled, ref_pooled)

	temporal_distance = _dtw_temporal_distance(
		user_frames,
		ref_frames,
		sakoe_chiba_radius=sakoe_chiba_radius,
	)

	return EmbeddingComparisonResult(
		similarity=similarity,
		temporal_distance=temporal_distance,
		metric=metric_key,
	)
