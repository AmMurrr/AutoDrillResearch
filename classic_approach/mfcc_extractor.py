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
	use_deltas: bool = True,
	delta_width: int = 9,
) -> np.ndarray:
	if np.asarray(samples).size == 0:
		logger.warning("MFCC extraction got empty samples")
		feature_count = int(n_mfcc) * (3 if use_deltas else 1)
		return np.zeros((feature_count, 0), dtype=np.float32)

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
	features = mfcc
	if use_deltas:
		delta = _compute_delta(mfcc, order=1, width=delta_width)
		delta_delta = _compute_delta(mfcc, order=2, width=delta_width)
		features = np.vstack([mfcc, delta, delta_delta])

	logger.debug(
		"MFCC features extracted: shape=%s n_mfcc=%s use_deltas=%s",
		features.shape,
		n_mfcc,
		use_deltas,
	)

	if use_cmvn:
		return apply_cmvn(features)
	return features


def _resolved_delta_width(n_frames: int, requested_width: int) -> int | None:
	if n_frames < 3:
		return None

	width = max(3, int(requested_width))
	if width % 2 == 0:
		width += 1

	if width > n_frames:
		width = n_frames if n_frames % 2 == 1 else n_frames - 1

	return width if width >= 3 else None


def _compute_delta(mfcc: np.ndarray, order: int, width: int) -> np.ndarray:
	resolved_width = _resolved_delta_width(mfcc.shape[1], width)
	if resolved_width is None:
		return np.zeros_like(mfcc, dtype=np.float32)

	return librosa.feature.delta(
		mfcc,
		order=order,
		width=resolved_width,
		mode="nearest",
	).astype(np.float32, copy=False)
