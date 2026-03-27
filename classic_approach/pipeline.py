from __future__ import annotations

import numpy as np

from .dtw import dtw_distance
from .mfcc_extractor import extract_mfcc
from .preprocessing import preprocess_audio
from .scorer import ComputeScoringResult, ScoringResult



def analyze(
    user_audio_path: str,
    reference_audio_path: str,
    transcript: str,
    n_mfcc: int = 20,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> ScoringResult:
    # препроцессинг
    user_audio = preprocess_audio(user_audio_path)
    reference_audio = preprocess_audio(reference_audio_path)

    # извлечение MFCC
    user_mfcc = extract_mfcc(
        user_audio.samples,
        user_audio.sample_rate,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    reference_mfcc = extract_mfcc(
        reference_audio.samples,
        reference_audio.sample_rate,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    # если MFCC не удалось извлечь
    if user_mfcc.shape[1] == 0 or reference_mfcc.shape[1] == 0:
        return ComputeScoringResult(0.0, [])

    # вычисление DTW расстояния 
    distance = dtw_distance(user_mfcc, reference_mfcc)
    if not np.isfinite(distance):
        return ComputeScoringResult(0.0, [])

    # простая формула для преобразования расстояния в оценку от 0 до 100
    score = 100.0 / (1.0 + distance)
    return ComputeScoringResult(score, [])