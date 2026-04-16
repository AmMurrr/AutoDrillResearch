from __future__ import annotations

import dtaidistance.dtw_ndim as dtw_ndim
import numpy as np


def _resolve_window(sakoe_chiba_radius: int | None) -> int | None:
    if sakoe_chiba_radius is None:
        return None

    radius = int(sakoe_chiba_radius)
    return radius if radius > 0 else None

# Преобразование 2D массива MFCC (n_mfcc x n_frames) в 2D массив (n_frames x n_mfcc) для DTW
def _as_frame_matrix(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("DTW требует 2D массив признаков (n_mfcc x n_frames)")
    return arr.T

# Вычисление DTW расстояния между двумя последовательностями MFCC
def dtw_distance(
    features_a: np.ndarray,
    features_b: np.ndarray,
    sakoe_chiba_radius: int | None = None,
) -> float:

    seq_a = _as_frame_matrix(features_a)
    seq_b = _as_frame_matrix(features_b)

    n, m = seq_a.shape[0], seq_b.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    # Многомерное DTW расстояние с нормализацией по длине и размерности признаков
    seq_a_nd = seq_a.astype(np.double)
    seq_b_nd = seq_b.astype(np.double)
    feature_dim = max(1, seq_a_nd.shape[1])
    window = _resolve_window(sakoe_chiba_radius)

    distance = float(dtw_ndim.distance(seq_a_nd, seq_b_nd, window=window))
   
    return distance / (max(n, m) * np.sqrt(feature_dim))