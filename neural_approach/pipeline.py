from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import numpy as np
from asr.vosk import VoskError, check_expected_text_for_preprocessed_audio
from app.logging_config import get_logger
from scoring.anchor_calibration import (
    build_anchor_distance_profile_from_distances,
    build_anchor_distance_profiles,
    describe_anchor_set,
    fit_sigmoid_from_anchor_distances,
    fit_sigmoid_from_anchor_profiles,
    get_word_anchor_set,
    normalize_word,
    sigmoid_score,
)

from .embedding_comparator import compare_embeddings
from .input_gate import validate_speech_signal
from .preprocessing import preprocess_audio
from .scorer import (
    ScoringResult,
    blend_with_duration_score,
    compute_anchor_profile_calibrated_scoring_result,
    compute_raw_distance,
    compute_scoring_result,
)
from .wav2vec_extractor import DEFAULT_MODEL_NAME, Wav2VecEmbeddings, extract_wav2vec_embeddings


DEFAULT_MAX_ANCHORS_PER_CLASS = 20
DEFAULT_RAW_DISTANCE_ALPHA = 0.65
DEFAULT_DURATION_SCORE_WEIGHT = 0.1
logger = get_logger(__name__)


@lru_cache(maxsize=1024)
def _extract_anchor_embeddings_cached(
    anchor_path: str,
    mtime_ns: int,
    file_size: int,
    model_name: str,
    device: str | None,
    hf_token: str | None,
) -> Wav2VecEmbeddings:
    del mtime_ns
    del file_size

    anchor_audio = preprocess_audio(anchor_path)
    return extract_wav2vec_embeddings(
        anchor_audio.samples,
        anchor_audio.sample_rate,
        model_name=model_name,
        device=device,
        hf_token=hf_token,
    )


def _extract_anchor_embeddings(
    anchor_path: str,
    model_name: str,
    device: str | None,
    hf_token: str | None,
) -> Wav2VecEmbeddings:
    anchor_file = Path(anchor_path)
    stats = anchor_file.stat()
    return _extract_anchor_embeddings_cached(
        str(anchor_file.resolve()),
        int(stats.st_mtime_ns),
        int(stats.st_size),
        model_name,
        device,
        hf_token,
    )


def _collect_valid_anchor_paths(
    paths: tuple[str, ...],
    model_name: str,
    device: str | None,
    hf_token: str | None,
) -> list[str]:
    valid_paths: list[str] = []
    for path in paths:
        try:
            embeddings = _extract_anchor_embeddings(
                anchor_path=path,
                model_name=model_name,
                device=device,
                hf_token=hf_token,
            )
        except Exception as exc:
            logger.warning("Skipping anchor %s: failed to extract embeddings (%s)", path, exc)
            continue

        if int(embeddings.frame_embeddings.shape[0]) == 0:
            logger.warning("Skipping anchor %s: no embedding frames", path)
            continue
        valid_paths.append(path)

    return valid_paths


def _compare_raw_distance(
    embeddings_a: Wav2VecEmbeddings,
    embeddings_b: Wav2VecEmbeddings,
    similarity_metric: str,
    sakoe_chiba_radius: int | None,
    raw_distance_alpha: float,
):
    comparison = compare_embeddings(
        user_embeddings=embeddings_a,
        reference_embeddings=embeddings_b,
        metric=similarity_metric,
        sakoe_chiba_radius=sakoe_chiba_radius,
    )
    raw_distance = compute_raw_distance(
        similarity=comparison.similarity,
        temporal_distance=comparison.temporal_distance,
        metric=comparison.metric,
        alpha=raw_distance_alpha,
    )
    return comparison, raw_distance


def _frame_count(embeddings: Wav2VecEmbeddings) -> int:
    return int(embeddings.frame_embeddings.shape[0])


def _median_log_frame_distance(
    frame_count: int,
    reference_frame_counts: list[int],
    exclude_index: int | None = None,
) -> float:
    values = [
        abs(math.log(max(int(frame_count), 1) / max(int(reference_count), 1)))
        for index, reference_count in enumerate(reference_frame_counts)
        if exclude_index is None or index != exclude_index
    ]
    if not values:
        return 0.0
    return float(np.median(values))


def _duration_score_against_perfect_anchors(
    user_embeddings: Wav2VecEmbeddings,
    perfect_embeddings: list[Wav2VecEmbeddings],
    fail_embeddings: list[Wav2VecEmbeddings],
) -> tuple[float, float]:
    perfect_frame_counts = [_frame_count(embeddings) for embeddings in perfect_embeddings]
    fail_frame_counts = [_frame_count(embeddings) for embeddings in fail_embeddings]

    if not perfect_frame_counts or not fail_frame_counts:
        return float("inf"), 0.0

    perfect_duration_distances = [
        _median_log_frame_distance(frame_count, perfect_frame_counts, exclude_index=index)
        for index, frame_count in enumerate(perfect_frame_counts)
    ]
    fail_duration_distances = [
        _median_log_frame_distance(frame_count, perfect_frame_counts)
        for frame_count in fail_frame_counts
    ]
    duration_params = fit_sigmoid_from_anchor_distances(
        distances_100=perfect_duration_distances,
        distances_0=fail_duration_distances,
    )
    user_duration_distance = _median_log_frame_distance(
        _frame_count(user_embeddings),
        perfect_frame_counts,
    )
    return user_duration_distance, sigmoid_score(user_duration_distance, duration_params)


def _distance_between_anchor_paths(
    anchor_path_a: str,
    anchor_path_b: str,
    similarity_metric: str,
    model_name: str,
    device: str | None,
    hf_token: str | None,
    sakoe_chiba_radius: int | None,
    raw_distance_alpha: float,
) -> float:
    embeddings_a = _extract_anchor_embeddings(
        anchor_path=anchor_path_a,
        model_name=model_name,
        device=device,
        hf_token=hf_token,
    )
    embeddings_b = _extract_anchor_embeddings(
        anchor_path=anchor_path_b,
        model_name=model_name,
        device=device,
        hf_token=hf_token,
    )

    _, raw_distance = _compare_raw_distance(
        embeddings_a=embeddings_a,
        embeddings_b=embeddings_b,
        similarity_metric=similarity_metric,
        sakoe_chiba_radius=sakoe_chiba_radius,
        raw_distance_alpha=raw_distance_alpha,
    )
    return float(raw_distance)


def _build_anchor_calibration(
    word: str,
    similarity_metric: str,
    model_name: str,
    device: str | None,
    hf_token: str | None,
    sakoe_chiba_radius: int | None,
    raw_distance_alpha: float,
    max_anchors_per_class: int,
    anchor_root: str | None,
):
    logger.info(
        "Building neural anchor calibration for '%s' (model=%s, max_per_class=%s)",
        word,
        model_name,
        max_anchors_per_class,
    )
    anchor_set = get_word_anchor_set(
        word=word,
        anchor_root=anchor_root,
        max_anchors_per_class=max_anchors_per_class,
    )

    valid_perfect_paths = _collect_valid_anchor_paths(
        paths=anchor_set.perfect_paths,
        model_name=model_name,
        device=device,
        hf_token=hf_token,
    )
    valid_moderate_paths = _collect_valid_anchor_paths(
        paths=anchor_set.moderate_paths,
        model_name=model_name,
        device=device,
        hf_token=hf_token,
    )
    valid_fail_paths = _collect_valid_anchor_paths(
        paths=anchor_set.fail_paths,
        model_name=model_name,
        device=device,
        hf_token=hf_token,
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
            similarity_metric=similarity_metric,
            model_name=model_name,
            device=device,
            hf_token=hf_token,
            sakoe_chiba_radius=sakoe_chiba_radius,
            raw_distance_alpha=raw_distance_alpha,
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
        "Neural calibration ready for '%s': perfect=%s moderate=%s fail=%s d100=%.4f d0=%.4f",
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


def _resolve_similarity_metric(metric: str) -> str:
    metric_key = metric.strip().lower()
    if metric_key != "cosine":
        raise ValueError("Only 'cosine' metric is supported")
    return metric_key


def analyze(
    user_audio_path: str,
    transcript: str,
    similarity: str = "cosine",
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    hf_token: str | None = None,
    sakoe_chiba_radius: int | None = None,
    use_vosk: bool = True,
    raw_distance_alpha: float = DEFAULT_RAW_DISTANCE_ALPHA,
    max_anchors_per_class: int = DEFAULT_MAX_ANCHORS_PER_CLASS,
    anchor_root: str | None = None,
) -> ScoringResult:
    logger.info(
        "Neural analyze started: path=%s transcript='%s' model=%s use_vosk=%s",
        user_audio_path,
        transcript,
        model_name,
        use_vosk,
    )
    metric_key = _resolve_similarity_metric(similarity)
    normalized_word = normalize_word(transcript)
    if not normalized_word:
        logger.warning("Neural analyze aborted: empty transcript")
        return compute_scoring_result(
            similarity=0.0,
            temporal_distance=float("inf"),
            metric=metric_key,
            model_name=model_name,
            status="invalid_reference",
            reason="empty_transcript",
        )

    user_audio = preprocess_audio(user_audio_path)
    speech_gate = validate_speech_signal(
        user_audio.samples,
        sample_rate=user_audio.sample_rate,
    )
    if not speech_gate.passed:
        logger.warning("Neural analyze aborted: speech gate failed")
        return compute_scoring_result(
            similarity=0.0,
            temporal_distance=float("inf"),
            metric=metric_key,
            model_name=model_name,
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
            logger.error("Neural analyze aborted: Vosk failure (%s)", exc)
            return compute_scoring_result(
                similarity=0.0,
                temporal_distance=float("inf"),
                metric=metric_key,
                model_name=model_name,
                status="asr_error",
                reason=f"vosk_failure:{exc}",
            )

        if not transcript_check.is_match:
            reason = (
                f"recognized:{transcript_check.recognized_text};"
                f"expected:{transcript_check.expected_text}"
            )
            logger.warning("Neural analyze aborted: ASR mismatch (%s)", reason)
            return compute_scoring_result(
                similarity=0.0,
                temporal_distance=float("inf"),
                metric=metric_key,
                model_name=model_name,
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
            similarity_metric=metric_key,
            model_name=model_name,
            device=device,
            hf_token=hf_token,
            sakoe_chiba_radius=sakoe_chiba_radius,
            raw_distance_alpha=raw_distance_alpha,
            max_anchors_per_class=max_anchors_per_class,
            anchor_root=anchor_root,
        )
    except ValueError as exc:
        logger.warning("Neural analyze aborted: invalid anchors (%s)", exc)
        return compute_scoring_result(
            similarity=0.0,
            temporal_distance=float("inf"),
            metric=metric_key,
            model_name=model_name,
            status="invalid_reference",
            reason=str(exc),
        )

    user_embeddings = extract_wav2vec_embeddings(
        user_audio.samples,
        user_audio.sample_rate,
        model_name=model_name,
        device=device,
        hf_token=hf_token,
    )
    if int(user_embeddings.frame_embeddings.shape[0]) == 0:
        logger.warning("Neural analyze aborted: user embeddings are empty")
        return compute_scoring_result(
            similarity=0.0,
            temporal_distance=float("inf"),
            metric=metric_key,
            model_name=model_name,
            status="empty_audio",
            reason="empty_embeddings",
        )

    user_similarity_to_perfect: list[float] = []
    user_temporal_to_perfect: list[float] = []
    user_raw_to_perfect: list[float] = []
    perfect_anchor_embeddings: list[Wav2VecEmbeddings] = []

    for perfect_path in valid_perfect_paths:
        perfect_embeddings = _extract_anchor_embeddings(
            anchor_path=perfect_path,
            model_name=model_name,
            device=device,
            hf_token=hf_token,
        )
        perfect_anchor_embeddings.append(perfect_embeddings)
        comparison, raw_distance = _compare_raw_distance(
            embeddings_a=user_embeddings,
            embeddings_b=perfect_embeddings,
            similarity_metric=metric_key,
            sakoe_chiba_radius=sakoe_chiba_radius,
            raw_distance_alpha=raw_distance_alpha,
        )
        if np.isfinite(raw_distance):
            user_similarity_to_perfect.append(float(comparison.similarity))
            user_temporal_to_perfect.append(float(comparison.temporal_distance))
            user_raw_to_perfect.append(float(raw_distance))

    if not user_raw_to_perfect:
        logger.warning("Neural analyze aborted: no finite distances to perfect anchors")
        return compute_scoring_result(
            similarity=0.0,
            temporal_distance=float("inf"),
            metric=metric_key,
            model_name=model_name,
            status="invalid_reference",
            reason="invalid_perfect_anchor_distances",
        )

    user_raw_to_moderate: list[float] = []
    for moderate_path in valid_moderate_paths:
        moderate_embeddings = _extract_anchor_embeddings(
            anchor_path=moderate_path,
            model_name=model_name,
            device=device,
            hf_token=hf_token,
        )
        _, raw_distance = _compare_raw_distance(
            embeddings_a=user_embeddings,
            embeddings_b=moderate_embeddings,
            similarity_metric=metric_key,
            sakoe_chiba_radius=sakoe_chiba_radius,
            raw_distance_alpha=raw_distance_alpha,
        )
        if np.isfinite(raw_distance):
            user_raw_to_moderate.append(float(raw_distance))

    user_raw_to_fail: list[float] = []
    fail_anchor_embeddings: list[Wav2VecEmbeddings] = []
    for fail_path in valid_fail_paths:
        fail_embeddings = _extract_anchor_embeddings(
            anchor_path=fail_path,
            model_name=model_name,
            device=device,
            hf_token=hf_token,
        )
        fail_anchor_embeddings.append(fail_embeddings)
        _, raw_distance = _compare_raw_distance(
            embeddings_a=user_embeddings,
            embeddings_b=fail_embeddings,
            similarity_metric=metric_key,
            sakoe_chiba_radius=sakoe_chiba_radius,
            raw_distance_alpha=raw_distance_alpha,
        )
        if np.isfinite(raw_distance):
            user_raw_to_fail.append(float(raw_distance))

    if not user_raw_to_fail:
        logger.warning("Neural analyze aborted: no finite distances to fail anchors")
        return compute_scoring_result(
            similarity=0.0,
            temporal_distance=float("inf"),
            metric=metric_key,
            model_name=model_name,
            status="invalid_reference",
            reason="invalid_fail_anchor_distances",
        )

    aggregated_similarity = float(np.mean(user_similarity_to_perfect))
    aggregated_temporal_distance = float(np.mean(user_temporal_to_perfect))
    user_profile = build_anchor_distance_profile_from_distances(
        perfect_distances=user_raw_to_perfect,
        moderate_distances=user_raw_to_moderate,
        fail_distances=user_raw_to_fail,
    )

    result = compute_anchor_profile_calibrated_scoring_result(
        similarity=aggregated_similarity,
        temporal_distance=aggregated_temporal_distance,
        metric=metric_key,
        model_name=model_name,
        profile=user_profile,
        calibration_params=calibration_params,
        status="ok",
        reason="",
    )
    duration_distance, duration_score = _duration_score_against_perfect_anchors(
        user_embeddings=user_embeddings,
        perfect_embeddings=perfect_anchor_embeddings,
        fail_embeddings=fail_anchor_embeddings,
    )
    result = blend_with_duration_score(
        result,
        duration_distance=duration_distance,
        duration_score=duration_score,
        duration_weight=DEFAULT_DURATION_SCORE_WEIGHT,
    )

    if not result.reason:
        anchor_stats = describe_anchor_set(anchor_set)
        result.reason = (
            f"anchors:perfect={anchor_stats['perfect']};"
            f"fail={anchor_stats['fail']};"
            f"moderate={anchor_stats['moderate']};"
            f"empty={anchor_stats['empty_word']}"
        )

    logger.info(
        "Neural analyze finished: status=%s score=%.2f raw_distance=%s",
        result.status,
        result.pronunciation_score,
        result.raw_distance,
    )

    return result
