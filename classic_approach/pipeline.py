from __future__ import annotations

import numpy as np
from asr.vosk import VoskError, check_expected_text_for_preprocessed_audio

from .dtw import dtw_distance
from .forced_aligner import pseudo_localize_errors
from .input_gate import validate_speech_signal
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
    sakoe_chiba_radius: int | None,
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

    distance = dtw_distance(
        user_mfcc,
        reference_mfcc,
        sakoe_chiba_radius=sakoe_chiba_radius,
    )
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
    sakoe_chiba_radius: int | None = None,
    use_vosk: bool = True,
) -> ScoringResult:
    # препроцессинг
    user_audio = preprocess_audio(user_audio_path)
    speech_gate = validate_speech_signal(
        user_audio.samples,
        sample_rate=user_audio.sample_rate,
    )
    if not speech_gate.passed:
        return ComputeScoringResult(
            dtw_score=0.0,
            phoneme_issues=["word:unrecognized"],
            distance=float("inf"),
            error_localization=[],
            status="empty_audio",
            reason="insufficient_speech",
        )

    if use_vosk:
        try:
            transcript_check = check_expected_text_for_preprocessed_audio(
                samples=user_audio.samples,
                sample_rate=user_audio.sample_rate,
                expected_text=transcript,
            )
        except (VoskError, ValueError) as exc:
            return ComputeScoringResult(
                dtw_score=0.0,
                phoneme_issues=["word:asr_error"],
                distance=float("inf"),
                error_localization=[],
                status="asr_error",
                reason=f"vosk_failure:{exc}",
            )

        if not transcript_check.is_match:
            reason = (
                f"recognized:{transcript_check.recognized_text};"
                f"expected:{transcript_check.expected_text}"
            )
            return ComputeScoringResult(
                dtw_score=0.0,
                phoneme_issues=["word:wrong_word"],
                distance=float("inf"),
                error_localization=[],
                status="wrong_word",
                reason=reason,
            )

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
            sakoe_chiba_radius=sakoe_chiba_radius,
        )
        for path in reference_paths
    ]

    return aggregate_scoring_results(per_reference_results)