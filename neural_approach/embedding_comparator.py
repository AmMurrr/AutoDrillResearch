from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dtaidistance.dtw_ndim as dtw_ndim
import numpy as np
from app.logging_config import get_logger

from .wav2vec_extractor import statistical_pooling


logger = get_logger(__name__)


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


def _as_pooled_vector(embedding: np.ndarray) -> np.ndarray:
	arr = np.asarray(embedding, dtype=np.float32)
	if arr.ndim != 1:
		raise ValueError("Expected pooled embedding with shape (pooled_dim,)")
	return arr


def _embedding_parts(
	embeddings: Any,
	pooled_embedding: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	if hasattr(embeddings, "frame_embeddings"):
		frames = _as_frame_matrix(getattr(embeddings, "frame_embeddings"))
		precomputed_pool = pooled_embedding
		if precomputed_pool is None:
			precomputed_pool = getattr(embeddings, "pooled_embedding", None)
		if precomputed_pool is not None:
			return frames, _as_pooled_vector(precomputed_pool)
		return frames, statistical_pooling(frames)

	frames = _as_frame_matrix(embeddings)
	if pooled_embedding is not None:
		return frames, _as_pooled_vector(pooled_embedding)
	return frames, statistical_pooling(frames)


def _l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	return matrix / np.maximum(norms, eps)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-8) -> float:
	denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
	if denom <= eps:
		return 0.0
	return float(np.dot(vec_a, vec_b) / denom)


def _dtw_temporal_distance(
	embeddings_a: np.ndarray,
	embeddings_b: np.ndarray,
	sakoe_chiba_radius: int | None = None,
) -> float:
	seq_a = _l2_normalize_rows(_as_frame_matrix(embeddings_a)).astype(np.double)
	seq_b = _l2_normalize_rows(_as_frame_matrix(embeddings_b)).astype(np.double)

	n, m = seq_a.shape[0], seq_b.shape[0]
	if n == 0 or m == 0:
		logger.warning("Neural temporal DTW got empty sequence (n=%s, m=%s)", n, m)
		return float("inf")

	feature_dim = max(1, seq_a.shape[1])
	window = _resolve_window(sakoe_chiba_radius)
	raw_distance = float(dtw_ndim.distance(seq_a, seq_b, window=window))
	normalized = raw_distance / (max(n, m) * np.sqrt(feature_dim))
	logger.debug("Neural temporal DTW distance computed: %.6f", normalized)
	return normalized


def compare_embeddings(
	user_embeddings: Any,
	reference_embeddings: Any,
	metric: str = "cosine",
	sakoe_chiba_radius: int | None = None,
	user_pooled_embedding: np.ndarray | None = None,
	reference_pooled_embedding: np.ndarray | None = None,
) -> EmbeddingComparisonResult:
	metric_key = _resolve_metric(metric)

	user_frames, user_pooled = _embedding_parts(user_embeddings, user_pooled_embedding)
	ref_frames, ref_pooled = _embedding_parts(reference_embeddings, reference_pooled_embedding)

	# Косинусное сходство естественно лежит в диапазоне [-1, 1].
	similarity = _cosine_similarity(user_pooled, ref_pooled)

	temporal_distance = _dtw_temporal_distance(
		user_frames,
		ref_frames,
		sakoe_chiba_radius=sakoe_chiba_radius,
	)
	logger.debug(
		"Compared embeddings: similarity=%.6f temporal_distance=%.6f metric=%s",
		similarity,
		temporal_distance,
		metric_key,
	)

	return EmbeddingComparisonResult(
		similarity=similarity,
		temporal_distance=temporal_distance,
		metric=metric_key,
	)
