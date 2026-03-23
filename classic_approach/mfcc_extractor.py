# from __future__ import annotations

import librosa
import numpy as np


def extract_mfcc(
	samples: np.ndarray,
	sample_rate: int,
	n_mfcc: int = 20,
	frame_ms: int = 25,
	hop_ms: int = 10,
) -> np.ndarray:
	win_length = int(sample_rate * (frame_ms / 1000.0))
	hop_length = int(sample_rate * (hop_ms / 1000.0))
	n_fft = max(512, int(2 ** np.ceil(np.log2(max(win_length, 1)))))

	return librosa.feature.mfcc(
		y=samples,
		sr=sample_rate,
		n_mfcc=n_mfcc,
		n_fft=n_fft,
		hop_length=max(hop_length, 1),
		win_length=max(win_length, 1),
	)
