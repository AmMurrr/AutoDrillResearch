from __future__ import annotations

import numpy as np

from .dtw import dtw_distance
from .forced_aligner import pseudo_localize_errors
from .mfcc_extractor import extract_mfcc
from .preprocessing import preprocess_audio
from .scorer import ComputeScoringResult, ScoringResult


def _distance_to_score(distance: float, user_frames: int, reference_frames: int) -> float:
    DISTANCE_MID = 0.25
    SHAPE = 20.0
    base_score = 100.0 / (1.0 + (max(distance, 0.0) / DISTANCE_MID) ** SHAPE)

    
    
    frame_ratio = max(user_frames, reference_frames) / max(1, min(user_frames, reference_frames))
    log_ratio = abs(np.log(frame_ratio))
    duration_penalty = np.exp(-0.8 * (log_ratio**2))

    # Дополнительный штраф для очень длинных попыток (частый признак нецелевой речи).
    long_utterance_penalty = np.exp(-0.35 * max(frame_ratio - 3.0, 0.0))
    duration_penalty = float(np.clip(duration_penalty * long_utterance_penalty, 0.0, 1.0))

    return float(np.clip(base_score * duration_penalty, 0.0, 100.0))


# Основной анализ
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
        return ComputeScoringResult(0.0, [], float("inf"), error_localization=[])

    # вычисление DTW расстояния 
    distance = dtw_distance(user_mfcc, reference_mfcc)
    if not np.isfinite(distance):
        return ComputeScoringResult(0.0, [], float("inf"), error_localization=[])

    # преобразование расстояния в оценку
    user_frames = user_mfcc.shape[1]
    reference_frames = reference_mfcc.shape[1]
    score = _distance_to_score(distance, user_frames, reference_frames)

    error_localization = pseudo_localize_errors(
        user_mfcc=user_mfcc,
        reference_mfcc=reference_mfcc,
        transcript=transcript,
    )

    problematic_regions = [
        f"{entry['word']}:{entry['problem_zone']}"
        for entry in error_localization
        if bool(entry.get("is_problematic"))
    ]

    return ComputeScoringResult(
        score,
        problematic_regions,
        distance,
        error_localization=error_localization,
    )