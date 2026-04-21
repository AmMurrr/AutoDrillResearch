from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from asr.vosk import VoskError, check_expected_text_for_preprocessed_audio
from scoring.anchor_calibration import (
    describe_anchor_set,
    fit_sigmoid_from_anchor_distances,
    get_word_anchor_set,
    is_known_zero_anchor,
    median_or_default,
    normalize_word,
)

from .dtw import dtw_distance
from .input_gate import validate_speech_signal
from .mfcc_extractor import extract_mfcc
from .preprocessing import preprocess_audio
from .scorer import (
    ScoringResult,
    ComputeScoringResult,
    compute_calibrated_scoring_result,
)


DEFAULT_MAX_ANCHORS_PER_CLASS = 12


@lru_cache(maxsize=4096)
def _extract_anchor_mfcc_cached(
    anchor_path: str,
    mtime_ns: int,
    file_size: int,
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
) -> np.ndarray:
    del mtime_ns
    del file_size

    anchor_audio = preprocess_audio(anchor_path)
    return extract_mfcc(
        anchor_audio.samples,
        anchor_audio.sample_rate,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )


def _extract_anchor_mfcc(
    anchor_path: str,
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
) -> np.ndarray:
    anchor_file = Path(anchor_path)
    stats = anchor_file.stat()
    return _extract_anchor_mfcc_cached(
        str(anchor_file.resolve()),
        int(stats.st_mtime_ns),
        int(stats.st_size),
        int(n_mfcc),
        int(frame_ms),
        int(hop_ms),
    )


def _distance_between_anchor_paths(
    anchor_path_a: str,
    anchor_path_b: str,
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
    sakoe_chiba_radius: int | None,
) -> float:
    mfcc_a = _extract_anchor_mfcc(
        anchor_path=anchor_path_a,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    mfcc_b = _extract_anchor_mfcc(
        anchor_path=anchor_path_b,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    if mfcc_a.shape[1] == 0 or mfcc_b.shape[1] == 0:
        return float("inf")

    return float(
        dtw_distance(
            mfcc_a,
            mfcc_b,
            sakoe_chiba_radius=sakoe_chiba_radius,
        )
    )


def _collect_valid_anchor_paths(
    paths: tuple[str, ...],
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
) -> list[str]:
    valid_paths: list[str] = []
    for path in paths:
        try:
            mfcc = _extract_anchor_mfcc(
                anchor_path=path,
                n_mfcc=n_mfcc,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
            )
        except Exception:
            continue

        if mfcc.shape[1] == 0:
            continue
        valid_paths.append(path)

    return valid_paths


def _build_anchor_calibration(
    word: str,
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
    sakoe_chiba_radius: int | None,
    max_anchors_per_class: int,
    anchor_root: str | None,
):
    anchor_set = get_word_anchor_set(
        word=word,
        anchor_root=anchor_root,
        max_anchors_per_class=max_anchors_per_class,
    )

    valid_perfect_paths = _collect_valid_anchor_paths(
        paths=anchor_set.perfect_paths,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    valid_zero_paths = _collect_valid_anchor_paths(
        paths=anchor_set.zero_paths,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    if not valid_perfect_paths:
        raise ValueError("missing_perfect_anchors")
    if not valid_zero_paths:
        raise ValueError("missing_zero_anchors")

    distances_100: list[float] = []
    for left_idx in range(len(valid_perfect_paths)):
        for right_idx in range(left_idx + 1, len(valid_perfect_paths)):
            distance = _distance_between_anchor_paths(
                valid_perfect_paths[left_idx],
                valid_perfect_paths[right_idx],
                n_mfcc=n_mfcc,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                sakoe_chiba_radius=sakoe_chiba_radius,
            )
            if np.isfinite(distance):
                distances_100.append(float(distance))

    if not distances_100:
        distances_100 = [0.0]

    distances_0: list[float] = []
    for perfect_path in valid_perfect_paths:
        for zero_path in valid_zero_paths:
            distance = _distance_between_anchor_paths(
                perfect_path,
                zero_path,
                n_mfcc=n_mfcc,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                sakoe_chiba_radius=sakoe_chiba_radius,
            )
            if np.isfinite(distance):
                distances_0.append(float(distance))

    if not distances_0:
        raise ValueError("invalid_zero_anchor_distances")

    calibration_params = fit_sigmoid_from_anchor_distances(
        distances_100=distances_100,
        distances_0=distances_0,
        epsilon=0.02,
    )

    return anchor_set, calibration_params, valid_perfect_paths, valid_zero_paths


def analyze(
    user_audio_path: str,
    transcript: str,
    n_mfcc: int = 20,
    frame_ms: int = 25,
    hop_ms: int = 10,
    sakoe_chiba_radius: int | None = None,
    use_vosk: bool = True,
    max_anchors_per_class: int = DEFAULT_MAX_ANCHORS_PER_CLASS,
    anchor_root: str | None = None,
) -> ScoringResult:
    normalized_word = normalize_word(transcript)
    if not normalized_word:
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float("inf"),
            status="invalid_reference",
            reason="empty_transcript",
        )

    user_audio = preprocess_audio(user_audio_path)
    speech_gate = validate_speech_signal(
        user_audio.samples,
        sample_rate=user_audio.sample_rate,
    )
    if not speech_gate.passed:
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float("inf"),
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
                distance=float("inf"),
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
                distance=float("inf"),
                status="wrong_word",
                reason=reason,
            )

    try:
        anchor_set, calibration_params, valid_perfect_paths, valid_zero_paths = _build_anchor_calibration(
            word=normalized_word,
            n_mfcc=n_mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            sakoe_chiba_radius=sakoe_chiba_radius,
            max_anchors_per_class=max_anchors_per_class,
            anchor_root=anchor_root,
        )
    except ValueError as exc:
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float("inf"),
            status="invalid_reference",
            reason=str(exc),
        )

    user_mfcc = extract_mfcc(
        user_audio.samples,
        user_audio.sample_rate,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    if user_mfcc.shape[1] == 0:
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float("inf"),
            status="empty_audio",
            reason="empty_features",
        )

    user_to_perfect_distances: list[float] = []
    for perfect_path in valid_perfect_paths:
        perfect_mfcc = _extract_anchor_mfcc(
            anchor_path=perfect_path,
            n_mfcc=n_mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )
        distance = dtw_distance(
            user_mfcc,
            perfect_mfcc,
            sakoe_chiba_radius=sakoe_chiba_radius,
        )
        if np.isfinite(distance):
            user_to_perfect_distances.append(float(distance))

    if not user_to_perfect_distances:
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float("inf"),
            status="invalid_reference",
            reason="invalid_perfect_anchor_distances",
        )

    user_to_zero_distances: list[float] = []
    for zero_path in valid_zero_paths:
        zero_mfcc = _extract_anchor_mfcc(
            anchor_path=zero_path,
            n_mfcc=n_mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )
        distance = dtw_distance(
            user_mfcc,
            zero_mfcc,
            sakoe_chiba_radius=sakoe_chiba_radius,
        )
        if np.isfinite(distance):
            user_to_zero_distances.append(float(distance))

    user_distance = median_or_default(user_to_perfect_distances, default=float("inf"))
    user_zero_distance = median_or_default(user_to_zero_distances, default=float("inf"))

    force_zero = False
    reason = ""
    if is_known_zero_anchor(user_audio_path, anchor_set):
        force_zero = True
        reason = "known_zero_anchor"
    elif np.isfinite(user_zero_distance) and np.isfinite(user_distance) and user_zero_distance <= user_distance:
        force_zero = True
        reason = "closer_to_zero_anchors"

    result = compute_calibrated_scoring_result(
        distance=user_distance,
        calibration_params=calibration_params,
        status="ok",
        reason=reason,
        force_zero=force_zero,
    )

    # Добавляем краткий свод по якорям в reason только при пустом reason.
    if not result.reason:
        anchor_stats = describe_anchor_set(anchor_set)
        result.reason = (
            f"anchors:perfect={anchor_stats['perfect']};"
            f"wrong={anchor_stats['wrong']};"
            f"moderate={anchor_stats['moderate']};"
            f"empty={anchor_stats['empty_word']}"
        )

    return result