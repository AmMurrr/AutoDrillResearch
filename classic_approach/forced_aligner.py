from __future__ import annotations

import re
from typing import Any

import numpy as np

REGION_NAMES = ("начало", "середина", "конец")


def _as_frame_matrix(features: np.ndarray) -> np.ndarray:
	"""Конвертирует MFCC в матрицу (n_frames, n_mfcc) для DTW и локализации ошибок."""
	arr = np.asarray(features, dtype=np.float32)
	if arr.ndim != 2:
		raise ValueError("Ожидалась MFCC-матрица формы (n_mfcc, n_frames)")
	return arr.T


def _compute_local_cost_matrix(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
	"""Вычисляет матрицу попарных евклидовых расстояний между кадрами."""
	diff = seq_a[:, None, :] - seq_b[None, :, :]
	return np.linalg.norm(diff, axis=2).astype(np.float32)


def _build_dtw_warping_path(local_cost: np.ndarray) -> list[tuple[int, int]]:
	"""Строит DTW-путь выравнивания обратным проходом по накопленной стоимости."""
	n, m = local_cost.shape
	if n == 0 or m == 0:
		return []

	acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
	acc[0, 0] = 0.0

	for i in range(1, n + 1):
		for j in range(1, m + 1):
			acc[i, j] = float(local_cost[i - 1, j - 1]) + min(
				acc[i - 1, j],
				acc[i, j - 1],
				acc[i - 1, j - 1],
			)

	i, j = n, m
	path: list[tuple[int, int]] = []
	while i > 0 and j > 0:
		path.append((i - 1, j - 1))
		prev_costs = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
		step = int(np.argmin(prev_costs))
		if step == 0:
			i -= 1
			j -= 1
		elif step == 1:
			i -= 1
		else:
			j -= 1

	while i > 0:
		i -= 1
		path.append((i, 0))
	while j > 0:
		j -= 1
		path.append((0, j))

	path.reverse()
	return path


def _aggregate_reference_frame_errors(
	local_cost: np.ndarray,
	path: list[tuple[int, int]],
	ref_frames: int,
) -> np.ndarray:
	sums = np.zeros(ref_frames, dtype=np.float64)
	counts = np.zeros(ref_frames, dtype=np.int32)

	for user_idx, ref_idx in path:
		sums[ref_idx] += float(local_cost[user_idx, ref_idx])
		counts[ref_idx] += 1

	return sums / np.maximum(counts, 1)


def _extract_words(transcript: str) -> list[str]:
	if not transcript or not transcript.strip():
		return ["фраза"]

	words: list[str] = []
	for token in transcript.strip().split():
		cleaned = re.sub(r"[^\w']+", "", token, flags=re.UNICODE)
		cleaned = cleaned.strip("_")
		if cleaned:
			words.append(cleaned)

	return words or ["фраза"]


def _word_spans(total_frames: int, words: list[str]) -> list[tuple[int, int]]:
	if total_frames <= 0 or not words:
		return []

	# Если токенов больше, чем кадров, объединяем хвост в последнее "слово".
	if len(words) > total_frames and total_frames > 1:
		prefix = words[: total_frames - 1]
		tail = " ".join(words[total_frames - 1 :])
		words[:] = prefix + [tail]
	elif len(words) > total_frames:
		words[:] = [" ".join(words)]

	bounds = np.linspace(0, total_frames, num=len(words) + 1, dtype=int)
	spans: list[tuple[int, int]] = []
	for idx in range(len(words)):
		start = int(bounds[idx])
		end = int(bounds[idx + 1])
		if end <= start:
			end = min(total_frames, start + 1)
		spans.append((start, end))

	if spans:
		spans[-1] = (spans[-1][0], total_frames)
	return spans


def _zone_means(word_errors: np.ndarray) -> list[float]:
	chunks = np.array_split(word_errors, 3)
	fallback = float(np.mean(word_errors))
	means: list[float] = []
	for chunk in chunks:
		means.append(float(np.mean(chunk)) if chunk.size else fallback)
	return means


def pseudo_localize_errors(
	user_mfcc: np.ndarray,
	reference_mfcc: np.ndarray,
	transcript: str,
	issue_ratio_threshold: float = 1.12,
) -> list[dict[str, Any]]:
	"""
	Псевдо-локализация ошибок на основе анализа DTW-пути.

	Это не полноценный forced alignment по фонемам. Метод оценивает слабые зоны
	в каждом слове (начало/середина/конец), проецируя покадровую рассинхронизацию
	DTW на временную шкалу эталона и равномерно деля слово по времени.
	"""
	user_seq = _as_frame_matrix(user_mfcc)
	ref_seq = _as_frame_matrix(reference_mfcc)

	if user_seq.shape[0] == 0 or ref_seq.shape[0] == 0:
		return []

	local_cost = _compute_local_cost_matrix(user_seq, ref_seq)
	path = _build_dtw_warping_path(local_cost)
	if not path:
		return []

	ref_frame_errors = _aggregate_reference_frame_errors(local_cost, path, ref_seq.shape[0])
	global_mean = float(np.mean(ref_frame_errors))
	if global_mean <= 0.0:
		global_mean = 1e-8

	words = _extract_words(transcript)
	spans = _word_spans(ref_seq.shape[0], words)

	diagnostics: list[dict[str, Any]] = []
	for index, ((start, end), word) in enumerate(zip(spans, words)):
		if end <= start:
			continue

		word_errors = ref_frame_errors[start:end]
		zone_errors = _zone_means(word_errors)
		worst_zone_idx = int(np.argmax(zone_errors))
		worst_zone_name = REGION_NAMES[worst_zone_idx]
		worst_zone_error = float(zone_errors[worst_zone_idx])
		word_error = float(np.mean(word_errors))
		relative_error = worst_zone_error / (global_mean + 1e-8)

		diagnostics.append(
			{
				"word": word,
				"word_index": index,
				"problem_zone": worst_zone_name,
				"is_problematic": bool(relative_error >= issue_ratio_threshold),
				"relative_error": float(round(relative_error, 4)),
				"word_error": float(round(word_error, 6)),
				"zone_errors": {
					"начало": float(round(zone_errors[0], 6)),
					"середина": float(round(zone_errors[1], 6)),
					"конец": float(round(zone_errors[2], 6)),
				},
			}
		)

	return diagnostics
