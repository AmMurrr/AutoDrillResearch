from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from app.logging_config import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class SpeechGateResult:
    passed: bool
    status: str
    reason: str
    duration_sec: float
    rms_dbfs: float
    voiced_ratio: float
    clipping_ratio: float


def _safe_rms_dbfs(samples: np.ndarray, eps: float = 1e-10) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if not np.isfinite(rms) or rms <= eps:
        return float("-inf")
    return float(20.0 * np.log10(rms))


def _frame_rms(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if samples.size == 0 or sample_rate <= 0:
        return np.array([], dtype=np.float32)

    frame_length = max(1, int(0.030 * sample_rate))
    hop_length = max(1, int(0.010 * sample_rate))
    return librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length,
    ).reshape(-1)


def _voiced_ratio_from_rms(rms_frames: np.ndarray, top_db: float = 35.0) -> float:
    if rms_frames.size == 0:
        return 0.0

    max_rms = float(np.max(rms_frames))
    if not np.isfinite(max_rms) or max_rms <= 1e-10:
        return 0.0

    threshold = max_rms * (10.0 ** (-float(top_db) / 20.0))
    return float(np.mean(rms_frames > threshold))


def _energy_dynamics_ratio(rms_frames: np.ndarray) -> float:
    if rms_frames.size == 0:
        return 0.0

    p95 = float(np.percentile(rms_frames, 95))
    p5 = float(np.percentile(rms_frames, 5))
    if p95 <= 1e-8:
        return 0.0

    return float(np.clip((p95 - p5) / (p95 + 1e-8), 0.0, 1.0))


def _spectral_flatness_mean(samples: np.ndarray, sample_rate: int) -> float:
    if samples.size == 0 or sample_rate <= 0:
        return 1.0

    hop_length = max(1, int(0.010 * sample_rate))
    flatness = librosa.feature.spectral_flatness(
        y=samples,
        n_fft=1024,
        hop_length=hop_length,
    )
    return float(np.mean(flatness))


def validate_speech_signal(
    samples: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 0.25,
    min_rms_dbfs: float = -45.0,
    min_voiced_ratio: float = 0.15,
    max_clipping_ratio: float = 0.02,
    min_energy_dynamics_ratio: float = 0.12,
    max_spectral_flatness: float = 0.35,
    stationary_duration_sec: float = 2.0,
    min_stationary_dynamics_ratio: float = 0.75,
    clipping_threshold: float = 0.999,
) -> SpeechGateResult:
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)

    if waveform.size == 0 or sample_rate <= 0:
        logger.warning("Classic speech gate failed: empty waveform or invalid sample rate")
        return SpeechGateResult(
            passed=False,
            status="empty_audio",
            reason="insufficient_speech",
            duration_sec=0.0,
            rms_dbfs=float("-inf"),
            voiced_ratio=0.0,
            clipping_ratio=0.0,
        )

    duration_sec = float(waveform.size / float(sample_rate))
    rms_dbfs = _safe_rms_dbfs(waveform)
    rms_frames = _frame_rms(waveform, sample_rate=sample_rate)
    voiced_ratio = _voiced_ratio_from_rms(rms_frames)
    dynamics_ratio = _energy_dynamics_ratio(rms_frames)
    spectral_flatness = _spectral_flatness_mean(waveform, sample_rate=sample_rate)
    clipping_ratio = float(np.mean(np.abs(waveform) >= float(clipping_threshold)))

    # Для single-word MVP длинный квазистационарный сигнал обычно означает
    # пустую запись/фон, а не целевое слово.
    is_long_stationary = duration_sec >= float(stationary_duration_sec) and dynamics_ratio < float(
        min_stationary_dynamics_ratio
    )

    is_valid = (
        duration_sec >= float(min_duration_sec)
        and rms_dbfs > float(min_rms_dbfs)
        and voiced_ratio >= float(min_voiced_ratio)
        and clipping_ratio <= float(max_clipping_ratio)
        and dynamics_ratio >= float(min_energy_dynamics_ratio)
        and spectral_flatness <= float(max_spectral_flatness)
        and not is_long_stationary
    )

    if not is_valid:
        logger.warning(
            "Classic speech gate rejected audio: duration=%.3f rms_dbfs=%.2f voiced_ratio=%.3f clipping_ratio=%.3f",
            duration_sec,
            rms_dbfs,
            voiced_ratio,
            clipping_ratio,
        )
    else:
        logger.info("Classic speech gate passed: duration=%.3f", duration_sec)

    return SpeechGateResult(
        passed=bool(is_valid),
        status="ok" if is_valid else "empty_audio",
        reason="" if is_valid else "insufficient_speech",
        duration_sec=duration_sec,
        rms_dbfs=rms_dbfs,
        voiced_ratio=voiced_ratio,
        clipping_ratio=clipping_ratio,
    )
