from __future__ import annotations

import numpy as np

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
    
    user_audio = preprocess_audio(user_audio_path)
    reference_audio = preprocess_audio(reference_audio_path)

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

    # Временно
    common_frames = min(user_mfcc.shape[1], reference_mfcc.shape[1])
    if common_frames == 0:
        return ComputeScoringResult(0.0, [])

    user_cut = user_mfcc[:, :common_frames]
    ref_cut = reference_mfcc[:, :common_frames]
    distance = float(np.mean(np.linalg.norm(user_cut - ref_cut, axis=0)))

    # Конвертируем расстояние в оценку от 0 до 100, где меньшее расстояние - лучшая оценка.
    score = max(0.0, 100.0 - distance)
    return ComputeScoringResult(score, [])