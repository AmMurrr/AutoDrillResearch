from __future__ import annotations

import dtaidistance
import numpy as np

# Преобразование 2D массива MFCC (n_mfcc x n_frames) в 2D массив (n_frames x n_mfcc) для DTW
def _as_frame_matrix(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("DTW требует 2D массив признаков (n_mfcc x n_frames)")
    return arr.T

# Вычисление DTW расстояния между двумя последовательностями MFCC
def dtw_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:

    seq_a = _as_frame_matrix(features_a)
    seq_b = _as_frame_matrix(features_b)

    n, m = seq_a.shape[0], seq_b.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    seq_a_1d = np.linalg.norm(seq_a, axis=1).astype(np.double)
    seq_b_1d = np.linalg.norm(seq_b, axis=1).astype(np.double)

    distance = float(dtaidistance.dtw.distance(seq_a_1d, seq_b_1d))
    return distance / (n + m)