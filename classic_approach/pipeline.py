from __future__ import annotations

import numpy as np

from .dtw import dtw_distance
from .forced_aligner import pseudo_localize_errors
from .mfcc_extractor import extract_mfcc
from .preprocessing import preprocess_audio
from .scorer import (
    ComputeScoringResult,
    ScoringResult,
    _distance_to_score,
    aggregate_scoring_results,
    compute_scoring_result_from_distance,
)


def _resolve_reference_paths(reference_audio_path: str | list[str]) -> list[str]:
    if isinstance(reference_audio_path, str):
        path = reference_audio_path.strip()
        return [path] if path else []

    paths: list[str] = []
    for path in reference_audio_path:
        normalized = str(path).strip()
        if normalized:
            paths.append(normalized)
    return paths


def _analyze_against_single_reference(
    user_mfcc: np.ndarray,
    reference_audio_path: str,
    transcript: str,
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
) -> ScoringResult:
    reference_audio = preprocess_audio(reference_audio_path)
    reference_mfcc = extract_mfcc(
        reference_audio.samples,
        reference_audio.sample_rate,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    if user_mfcc.shape[1] == 0 or reference_mfcc.shape[1] == 0:
        return ComputeScoringResult(0.0, [], float("inf"), error_localization=[])

    distance = dtw_distance(user_mfcc, reference_mfcc)
    if not np.isfinite(distance):
        return ComputeScoringResult(0.0, [], float("inf"), error_localization=[])

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

    return compute_scoring_result_from_distance(
        distance=float(distance),
        user_frames=int(user_mfcc.shape[1]),
        reference_frames=int(reference_mfcc.shape[1]),
        phoneme_issues=problematic_regions,
        error_localization=error_localization,
    )


# Основной анализ
def analyze(
    user_audio_path: str,
    reference_audio_path: str | list[str],
    transcript: str,
    n_mfcc: int = 20,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> ScoringResult:
    # препроцессинг
    user_audio = preprocess_audio(user_audio_path)
    reference_paths = _resolve_reference_paths(reference_audio_path)

    if not reference_paths:
        return ComputeScoringResult(0.0, [], float("inf"), error_localization=[])

    # извлечение MFCC
    user_mfcc = extract_mfcc(
        user_audio.samples,
        user_audio.sample_rate,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    # если MFCC не удалось извлечь
    if user_mfcc.shape[1] == 0:
        return ComputeScoringResult(0.0, [], float("inf"), error_localization=[])

    per_reference_results = [
        _analyze_against_single_reference(
            user_mfcc=user_mfcc,
            reference_audio_path=path,
            transcript=transcript,
            n_mfcc=n_mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )
        for path in reference_paths
    ]

    return aggregate_scoring_results(per_reference_results)