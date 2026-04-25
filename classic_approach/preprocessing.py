from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from app.logging_config import get_logger
import soundfile as sf


TARGET_SAMPLE_RATE = 16000
TARGET_DBFS = -20.0
logger = get_logger(__name__)


@dataclass
class PreprocessedAudio:
    samples: np.ndarray
    sample_rate: int


def load_audio(path: str) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(path, always_2d=False)
    logger.debug(
        "Loaded audio from %s (sample_rate=%s, shape=%s)", path, sample_rate, np.shape(samples)
    )
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return np.mean(samples, axis=1, dtype=np.float32)


def resample_audio(
    samples: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    if orig_sr == target_sr:
        return samples.astype(np.float32, copy=False)
    return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(samples: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(samples, top_db=top_db)
    return trimmed if trimmed.size > 0 else samples


def normalize_loudness(samples: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 1e-8:
        return samples

    current_dbfs = 20.0 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    gain = 10.0 ** (gain_db / 20.0)
    normalized = samples * gain
    return np.clip(normalized, -1.0, 1.0)


def preprocess_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> PreprocessedAudio:
    logger.info("Classic preprocessing started: %s", path)
    samples, sample_rate = load_audio(path)
    samples = to_mono(samples)
    samples = resample_audio(samples, orig_sr=sample_rate, target_sr=target_sr)
    samples = trim_silence(samples)
    samples = normalize_loudness(samples)
    if samples.size == 0:
        logger.warning("Classic preprocessing produced empty waveform: %s", path)
    else:
        duration_sec = float(samples.size / float(target_sr))
        logger.info("Classic preprocessing finished: %s (duration_sec=%.3f)", path, duration_sec)
    return PreprocessedAudio(samples=samples.astype(np.float32, copy=False), sample_rate=target_sr)
