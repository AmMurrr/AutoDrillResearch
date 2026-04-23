# from __future__ import annotations

import librosa
import numpy as np
from app.logging_config import get_logger


logger = get_logger(__name__)


def apply_cmvn(mfcc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
	"""CMVN: нормализация кепстральных признаков по среднему и дисперсии."""
	if mfcc.ndim != 2 or mfcc.shape[1] == 0:
		return mfcc

	mean = np.mean(mfcc, axis=1, keepdims=True)
	std = np.std(mfcc, axis=1, keepdims=True)
	std = np.maximum(std, eps)
	return (mfcc - mean) / std


def extract_mfcc(
	samples: np.ndarray,
	sample_rate: int,
	n_mfcc: int = 20,
	frame_ms: int = 25,
	hop_ms: int = 10,
	use_cmvn: bool = True,
) -> np.ndarray:
	if np.asarray(samples).size == 0:
		logger.warning("MFCC extraction got empty samples")
		return np.zeros((int(n_mfcc), 0), dtype=np.float32)

	win_length = int(sample_rate * (frame_ms / 1000.0))
	hop_length = int(sample_rate * (hop_ms / 1000.0))
	n_fft = max(512, int(2 ** np.ceil(np.log2(max(win_length, 1)))))

	mfcc = librosa.feature.mfcc(
		y=samples,
		sr=sample_rate,
		n_mfcc=n_mfcc,
		n_fft=n_fft,
		hop_length=max(hop_length, 1),
		win_length=max(win_length, 1),
	)
	logger.debug("MFCC extracted: shape=%s n_mfcc=%s", mfcc.shape, n_mfcc)

	if use_cmvn:
		return apply_cmvn(mfcc)
	return mfcc
