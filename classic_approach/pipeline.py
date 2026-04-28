from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from asr.vosk import VoskError, check_expected_text_for_preprocessed_audio
from app.logging_config import get_logger
from scoring.anchor_calibration import (
    build_anchor_distance_profile_from_distances,
    build_anchor_distance_profiles,
    describe_anchor_set,
    fit_sigmoid_from_anchor_profiles,
    get_word_anchor_set,
    normalize_word,
)

from .dtw import dtw_distance
from .input_gate import validate_speech_signal
from .mfcc_extractor import extract_mfcc
from .preprocessing import preprocess_audio
from .scorer import (
    ScoringResult,
    ComputeScoringResult,
    compute_profile_calibrated_scoring_result,
)


DEFAULT_MAX_ANCHORS_PER_CLASS = 20
logger = get_logger(__name__)


@lru_cache(maxsize=4096)
def _extract_anchor_mfcc_cached(
    anchor_path: str,
    mtime_ns: int,
    file_size: int,
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
    use_deltas: bool,
    delta_width: int,
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
        use_deltas=use_deltas,
        delta_width=delta_width,
    )


def _extract_anchor_mfcc(
    anchor_path: str,
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
    use_deltas: bool,
    delta_width: int,
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
        bool(use_deltas),
        int(delta_width),
    )


def _distance_between_anchor_paths(
    anchor_path_a: str,
    anchor_path_b: str,
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
    use_deltas: bool,
    delta_width: int,
    sakoe_chiba_radius: int | None,
) -> float:
    mfcc_a = _extract_anchor_mfcc(
        anchor_path=anchor_path_a,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        use_deltas=use_deltas,
        delta_width=delta_width,
    )
    mfcc_b = _extract_anchor_mfcc(
        anchor_path=anchor_path_b,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        use_deltas=use_deltas,
        delta_width=delta_width,
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
    use_deltas: bool,
    delta_width: int,
) -> list[str]:
    valid_paths: list[str] = []
    for path in paths:
        try:
            mfcc = _extract_anchor_mfcc(
                anchor_path=path,
                n_mfcc=n_mfcc,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                use_deltas=use_deltas,
                delta_width=delta_width,
            )
        except Exception as exc:
            logger.warning("Skipping anchor %s: failed to extract MFCC (%s)", path, exc)
            continue

        if mfcc.shape[1] == 0:
            logger.warning("Skipping anchor %s: MFCC has no frames", path)
            continue
        valid_paths.append(path)

    return valid_paths


def _build_anchor_calibration(
    word: str,
    n_mfcc: int,
    frame_ms: int,
    hop_ms: int,
    use_deltas: bool,
    delta_width: int,
    sakoe_chiba_radius: int | None,
    max_anchors_per_class: int,
    anchor_root: str | None,
):
    logger.info(
        "Building classic anchor calibration for '%s' (max_per_class=%s)",
        word,
        max_anchors_per_class,
    )
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
        use_deltas=use_deltas,
        delta_width=delta_width,
    )
    valid_moderate_paths = _collect_valid_anchor_paths(
        paths=anchor_set.moderate_paths,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        use_deltas=use_deltas,
        delta_width=delta_width,
    )
    valid_fail_paths = _collect_valid_anchor_paths(
        paths=anchor_set.fail_paths,
        n_mfcc=n_mfcc,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        use_deltas=use_deltas,
        delta_width=delta_width,
    )

    if not valid_perfect_paths:
        raise ValueError("missing_perfect_anchors")
    if not valid_fail_paths:
        raise ValueError("missing_fail_anchors")

    distance_cache: dict[tuple[str, str], float] = {}

    def _cached_distance(left_path: str, right_path: str) -> float:
        key = (left_path, right_path) if left_path <= right_path else (right_path, left_path)
        cached = distance_cache.get(key)
        if cached is not None:
            return cached

        distance = _distance_between_anchor_paths(
            left_path,
            right_path,
            n_mfcc=n_mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            use_deltas=use_deltas,
            delta_width=delta_width,
            sakoe_chiba_radius=sakoe_chiba_radius,
        )
        distance_cache[key] = float(distance)
        return float(distance)

    perfect_profiles, moderate_profiles, fail_profiles = build_anchor_distance_profiles(
        perfect_items=valid_perfect_paths,
        moderate_items=valid_moderate_paths,
        fail_items=valid_fail_paths,
        distance_fn=_cached_distance,
    )

    calibration_params = fit_sigmoid_from_anchor_profiles(
        perfect_profiles=perfect_profiles,
        moderate_profiles=moderate_profiles,
        fail_profiles=fail_profiles,
    )

    logger.info(
        "Classic calibration ready for '%s': perfect=%s moderate=%s fail=%s d100=%.4f d0=%.4f",
        word,
        len(valid_perfect_paths),
        len(valid_moderate_paths),
        len(valid_fail_paths),
        calibration_params.d100,
        calibration_params.d0,
    )

    return (
        anchor_set,
        calibration_params,
        valid_perfect_paths,
        valid_moderate_paths,
        valid_fail_paths,
    )


def analyze(
    user_audio_path: str,
    transcript: str,
    n_mfcc: int = 20,
    frame_ms: int = 25,
    hop_ms: int = 10,
    use_deltas: bool = True,
    delta_width: int = 9,
    sakoe_chiba_radius: int | None = None,
    use_vosk: bool = True,
    max_anchors_per_class: int = DEFAULT_MAX_ANCHORS_PER_CLASS,
    anchor_root: str | None = None,
) -> ScoringResult:
    logger.info(
        "Classic analyze started: path=%s transcript='%s' use_vosk=%s",
        user_audio_path,
        transcript,
        use_vosk,
    )
    normalized_word = normalize_word(transcript)
    if not normalized_word:
        logger.warning("Classic analyze aborted: empty transcript")
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
        logger.warning("Classic analyze aborted: speech gate failed")
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
            logger.error("Classic analyze aborted: Vosk failure (%s)", exc)
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
            logger.warning("Classic analyze aborted: ASR mismatch (%s)", reason)
            return ComputeScoringResult(
                dtw_score=0.0,
                distance=float("inf"),
                status="wrong_word",
                reason=reason,
            )

    try:
        (
            anchor_set,
            calibration_params,
            valid_perfect_paths,
            valid_moderate_paths,
            valid_fail_paths,
        ) = _build_anchor_calibration(
            word=normalized_word,
            n_mfcc=n_mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            use_deltas=use_deltas,
            delta_width=delta_width,
            sakoe_chiba_radius=sakoe_chiba_radius,
            max_anchors_per_class=max_anchors_per_class,
            anchor_root=anchor_root,
        )
    except ValueError as exc:
        logger.warning("Classic analyze aborted: invalid anchors (%s)", exc)
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
        use_deltas=use_deltas,
        delta_width=delta_width,
    )

    if user_mfcc.shape[1] == 0:
        logger.warning("Classic analyze aborted: user MFCC is empty")
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
            use_deltas=use_deltas,
            delta_width=delta_width,
        )
        distance = dtw_distance(
            user_mfcc,
            perfect_mfcc,
            sakoe_chiba_radius=sakoe_chiba_radius,
        )
        if np.isfinite(distance):
            user_to_perfect_distances.append(float(distance))

    if not user_to_perfect_distances:
        logger.warning("Classic analyze aborted: no finite distances to perfect anchors")
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float("inf"),
            status="invalid_reference",
            reason="invalid_perfect_anchor_distances",
        )

    user_to_moderate_distances: list[float] = []
    for moderate_path in valid_moderate_paths:
        moderate_mfcc = _extract_anchor_mfcc(
            anchor_path=moderate_path,
            n_mfcc=n_mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            use_deltas=use_deltas,
            delta_width=delta_width,
        )
        distance = dtw_distance(
            user_mfcc,
            moderate_mfcc,
            sakoe_chiba_radius=sakoe_chiba_radius,
        )
        if np.isfinite(distance):
            user_to_moderate_distances.append(float(distance))

    user_to_fail_distances: list[float] = []
    for fail_path in valid_fail_paths:
        fail_mfcc = _extract_anchor_mfcc(
            anchor_path=fail_path,
            n_mfcc=n_mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            use_deltas=use_deltas,
            delta_width=delta_width,
        )
        distance = dtw_distance(
            user_mfcc,
            fail_mfcc,
            sakoe_chiba_radius=sakoe_chiba_radius,
        )
        if np.isfinite(distance):
            user_to_fail_distances.append(float(distance))

    if not user_to_fail_distances:
        logger.warning("Classic analyze aborted: no finite distances to fail anchors")
        return ComputeScoringResult(
            dtw_score=0.0,
            distance=float("inf"),
            status="invalid_reference",
            reason="invalid_fail_anchor_distances",
        )

    user_profile = build_anchor_distance_profile_from_distances(
        perfect_distances=user_to_perfect_distances,
        moderate_distances=user_to_moderate_distances,
        fail_distances=user_to_fail_distances,
    )

    result = compute_profile_calibrated_scoring_result(
        profile=user_profile,
        calibration_params=calibration_params,
        status="ok",
        reason="",
    )

    # Добавляем краткий свод по якорям в reason только при пустом reason.
    if not result.reason:
        anchor_stats = describe_anchor_set(anchor_set)
        result.reason = (
            f"anchors:perfect={anchor_stats['perfect']};"
            f"fail={anchor_stats['fail']};"
            f"moderate={anchor_stats['moderate']};"
            f"empty={anchor_stats['empty_word']}"
        )

    logger.info(
        "Classic analyze finished: status=%s score=%.2f distance=%s",
        result.status,
        result.dtw_score,
        result.distance,
    )

    return result
