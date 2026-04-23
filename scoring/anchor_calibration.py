from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable, Sequence, TypeVar

import numpy as np
from app.logging_config import get_logger


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR_ROOT = WORKSPACE_ROOT / "data" / "ref"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
ANCHOR_QUALITIES = {"perfect", "fail", "moderate", "wrong"}
CALIBRATION_D100_QUANTILE = 0.80
CALIBRATION_D0_QUANTILE = 0.20
MIN_CALIBRATION_GAP = 0.03
MAX_CALIBRATION_SLOPE = 80.0
DEFAULT_ZERO_ANCHOR_MARGIN = 0.02
DEFAULT_PERFECT_TARGET_SCORE = 95.0
DEFAULT_MODERATE_TARGET_SCORE = 60.0
DEFAULT_FAIL_TARGET_SCORE = 5.0
DEFAULT_SCORE_FLOOR = 1.0
DEFAULT_SCORE_CEIL = 99.0
DEFAULT_PROFILE_REGULARIZATION = 0.03
logger = get_logger(__name__)


@dataclass(frozen=True)
class WordAnchorSet:
    word: str
    perfect_paths: tuple[str, ...]
    fail_paths: tuple[str, ...]
    moderate_paths: tuple[str, ...]
    empty_paths: tuple[str, ...]

    @property
    def wrong_paths(self) -> tuple[str, ...]:
        # Backward-compatible alias for legacy code/tests.
        return self.fail_paths

    @property
    def zero_paths(self) -> tuple[str, ...]:
        # Legacy alias. Empty anchors are handled by the speech gate, not by
        # pronunciation quality scoring.
        return self.fail_paths + self.moderate_paths

    @property
    def scoring_paths(self) -> tuple[str, ...]:
        return self.perfect_paths + self.moderate_paths + self.fail_paths

    @property
    def has_required_anchors(self) -> bool:
        return bool(self.perfect_paths) and bool(self.fail_paths)


@dataclass(frozen=True)
class SigmoidCalibrationParams:
    d100: float
    d0: float
    a: float
    b: float
    epsilon: float


@dataclass(frozen=True)
class AnchorDistanceProfile:
    perfect_distance: float
    moderate_distance: float
    fail_distance: float

    @property
    def is_valid(self) -> bool:
        return (
            np.isfinite(self.perfect_distance)
            and np.isfinite(self.fail_distance)
            and np.isfinite(self.moderate_distance)
        )

    def to_feature_vector(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.perfect_distance),
                float(self.moderate_distance),
                float(self.fail_distance),
                float(self.perfect_distance - self.fail_distance),
                float(self.moderate_distance - self.fail_distance),
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class MultiAnchorSigmoidParams:
    weights: tuple[float, ...]
    bias: float
    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    d100: float
    d0: float
    epsilon: float
    target_perfect_score: float
    target_moderate_score: float
    target_fail_score: float
    score_floor: float = DEFAULT_SCORE_FLOOR
    score_ceil: float = DEFAULT_SCORE_CEIL


_WORD_SEPARATOR_RE = re.compile(r"\s+")
_FOLDER_PATTERN_RE = re.compile(r"^(?P<word>[^_]+)_(?P<quality>.+)$")

T = TypeVar("T")


def normalize_word(word: str) -> str:
    return _WORD_SEPARATOR_RE.sub(" ", (word or "").strip().lower())


def _resolve_anchor_root(anchor_root: str | Path | None = None) -> Path:
    if anchor_root is None:
        return DEFAULT_ANCHOR_ROOT
    return Path(anchor_root).expanduser().resolve()


def _resolve_audio_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _deduplicate_paths(paths: Sequence[str]) -> tuple[str, ...]:
    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = _resolve_audio_path(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(normalized)
    return tuple(unique_paths)


def _iter_audio_files(directory: Path) -> list[str]:
    files: list[str] = []
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        files.append(str(file_path.resolve()))
    return files


def _parse_anchor_folder(folder_name: str) -> tuple[str | None, str | None]:
    stripped_name = (folder_name or "").strip().lower()
    if not stripped_name:
        return None, None

    if stripped_name == "_empty_word":
        return "", "empty_word"

    match = _FOLDER_PATTERN_RE.match(stripped_name)
    if match is None:
        return None, None

    word = normalize_word(match.group("word"))
    quality = match.group("quality").strip().lower()
    if quality == "wrong":
        quality = "fail"
    if not word or not quality:
        return None, None
    return word, quality


def _limited(paths: tuple[str, ...], max_anchors_per_class: int | None) -> tuple[str, ...]:
    if max_anchors_per_class is None:
        return paths

    limit = int(max_anchors_per_class)
    if limit <= 0:
        return tuple()
    return paths[:limit]


def _scan_anchor_groups(anchor_root: Path) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    grouped: dict[str, dict[str, list[str]]] = {}
    empty_word_paths: list[str] = []

    if not anchor_root.exists():
        return grouped, empty_word_paths

    for subdir in sorted(anchor_root.iterdir()):
        if not subdir.is_dir():
            continue

        word, quality = _parse_anchor_folder(subdir.name)
        if quality is None:
            continue

        audio_files = _iter_audio_files(subdir)
        if not audio_files:
            continue

        if quality == "empty_word":
            empty_word_paths.extend(audio_files)
            continue

        if quality not in ANCHOR_QUALITIES or word is None:
            continue

        entry = grouped.setdefault(
            word,
            {
                "perfect": [],
                "fail": [],
                "moderate": [],
            },
        )
        entry[quality].extend(audio_files)

    return grouped, empty_word_paths


def get_word_anchor_set(
    word: str,
    anchor_root: str | Path | None = None,
    max_anchors_per_class: int | None = None,
) -> WordAnchorSet:
    normalized_word = normalize_word(word)
    root = _resolve_anchor_root(anchor_root)
    grouped, empty_word_paths = _scan_anchor_groups(root)

    entry = grouped.get(
        normalized_word,
        {
            "perfect": [],
            "fail": [],
            "moderate": [],
        },
    )

    perfect_paths = _limited(_deduplicate_paths(entry["perfect"]), max_anchors_per_class)
    fail_paths = _limited(_deduplicate_paths(entry["fail"]), max_anchors_per_class)
    moderate_paths = _limited(_deduplicate_paths(entry["moderate"]), max_anchors_per_class)
    empty_paths = _limited(_deduplicate_paths(empty_word_paths), max_anchors_per_class)

    logger.info(
        "Resolved anchor set for '%s': perfect=%s fail=%s moderate=%s empty=%s",
        normalized_word,
        len(perfect_paths),
        len(fail_paths),
        len(moderate_paths),
        len(empty_paths),
    )

    return WordAnchorSet(
        word=normalized_word,
        perfect_paths=perfect_paths,
        fail_paths=fail_paths,
        moderate_paths=moderate_paths,
        empty_paths=empty_paths,
    )


def list_anchor_words(anchor_root: str | Path | None = None) -> list[str]:
    root = _resolve_anchor_root(anchor_root)
    grouped, _ = _scan_anchor_groups(root)
    words = [word for word, entry in grouped.items() if entry.get("perfect")]
    logger.debug("Listed anchor words from %s: %s words", root, len(words))
    return sorted(set(words))


def describe_anchor_set(anchor_set: WordAnchorSet) -> dict[str, int]:
    return {
        "perfect": len(anchor_set.perfect_paths),
        "fail": len(anchor_set.fail_paths),
        "moderate": len(anchor_set.moderate_paths),
        "empty_word": len(anchor_set.empty_paths),
        "zero_total": len(anchor_set.zero_paths),
    }


def is_known_zero_anchor(audio_path: str, anchor_set: WordAnchorSet) -> bool:
    resolved = _resolve_audio_path(audio_path)
    known = resolved in set(anchor_set.zero_paths)
    if known:
        logger.warning("Audio path is known zero-anchor: %s", resolved)
    return known


def pairwise_distances(items: Sequence[T], distance_fn: Callable[[T, T], float]) -> list[float]:
    distances: list[float] = []
    for left_idx in range(len(items)):
        for right_idx in range(left_idx + 1, len(items)):
            distance = float(distance_fn(items[left_idx], items[right_idx]))
            if np.isfinite(distance):
                distances.append(distance)
    return distances


def cross_distances(
    source_items: Sequence[T],
    target_items: Sequence[T],
    distance_fn: Callable[[T, T], float],
) -> list[float]:
    distances: list[float] = []
    for source_item in source_items:
        for target_item in target_items:
            distance = float(distance_fn(source_item, target_item))
            if np.isfinite(distance):
                distances.append(distance)
    return distances


def median_or_default(values: Sequence[float], default: float = float("inf")) -> float:
    finite_values = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=np.float64)
    if finite_values.size == 0:
        return float(default)
    return float(np.median(finite_values))


def build_anchor_distance_profile_from_distances(
    perfect_distances: Sequence[float],
    moderate_distances: Sequence[float],
    fail_distances: Sequence[float],
) -> AnchorDistanceProfile:
    perfect_distance = median_or_default(perfect_distances, default=float("inf"))
    fail_distance = median_or_default(fail_distances, default=float("inf"))
    moderate_distance = median_or_default(moderate_distances, default=float("inf"))

    if not np.isfinite(moderate_distance):
        if np.isfinite(perfect_distance) and np.isfinite(fail_distance):
            moderate_distance = float((perfect_distance + fail_distance) / 2.0)
        elif np.isfinite(perfect_distance):
            moderate_distance = float(perfect_distance)
        elif np.isfinite(fail_distance):
            moderate_distance = float(fail_distance)

    return AnchorDistanceProfile(
        perfect_distance=float(perfect_distance),
        moderate_distance=float(moderate_distance),
        fail_distance=float(fail_distance),
    )


def compute_anchor_distance_profile(
    item: T,
    perfect_items: Sequence[T],
    moderate_items: Sequence[T],
    fail_items: Sequence[T],
    distance_fn: Callable[[T, T], float],
    item_class: str | None = None,
) -> AnchorDistanceProfile:
    normalized_class = (item_class or "").strip().lower()

    def _distances(items: Sequence[T], class_name: str) -> list[float]:
        is_same_class = normalized_class == class_name
        distances: list[float] = []
        for candidate in items:
            if is_same_class and candidate == item:
                continue
            distance = float(distance_fn(item, candidate))
            if np.isfinite(distance):
                distances.append(distance)
        if not distances and is_same_class and len(items) == 1:
            distances.append(0.0)
        return distances

    return build_anchor_distance_profile_from_distances(
        perfect_distances=_distances(perfect_items, "perfect"),
        moderate_distances=_distances(moderate_items, "moderate"),
        fail_distances=_distances(fail_items, "fail"),
    )


def build_anchor_distance_profiles(
    perfect_items: Sequence[T],
    moderate_items: Sequence[T],
    fail_items: Sequence[T],
    distance_fn: Callable[[T, T], float],
) -> tuple[list[AnchorDistanceProfile], list[AnchorDistanceProfile], list[AnchorDistanceProfile]]:
    perfect_profiles = [
        compute_anchor_distance_profile(
            item=item,
            perfect_items=perfect_items,
            moderate_items=moderate_items,
            fail_items=fail_items,
            distance_fn=distance_fn,
            item_class="perfect",
        )
        for item in perfect_items
    ]
    moderate_profiles = [
        compute_anchor_distance_profile(
            item=item,
            perfect_items=perfect_items,
            moderate_items=moderate_items,
            fail_items=fail_items,
            distance_fn=distance_fn,
            item_class="moderate",
        )
        for item in moderate_items
    ]
    fail_profiles = [
        compute_anchor_distance_profile(
            item=item,
            perfect_items=perfect_items,
            moderate_items=moderate_items,
            fail_items=fail_items,
            distance_fn=distance_fn,
            item_class="fail",
        )
        for item in fail_items
    ]

    return perfect_profiles, moderate_profiles, fail_profiles


def fit_sigmoid_from_anchor_profiles(
    perfect_profiles: Sequence[AnchorDistanceProfile],
    moderate_profiles: Sequence[AnchorDistanceProfile],
    fail_profiles: Sequence[AnchorDistanceProfile],
    target_perfect_score: float = DEFAULT_PERFECT_TARGET_SCORE,
    target_moderate_score: float = DEFAULT_MODERATE_TARGET_SCORE,
    target_fail_score: float = DEFAULT_FAIL_TARGET_SCORE,
    epsilon: float = 0.02,
    regularization: float = DEFAULT_PROFILE_REGULARIZATION,
    score_floor: float = DEFAULT_SCORE_FLOOR,
    score_ceil: float = DEFAULT_SCORE_CEIL,
) -> MultiAnchorSigmoidParams:
    if not perfect_profiles:
        raise ValueError("perfect_profiles is empty")
    if not fail_profiles:
        raise ValueError("fail_profiles is empty")

    eps = float(epsilon)
    if not (0.0 < eps < 0.5):
        raise ValueError("epsilon must be in interval (0, 0.5)")

    feature_rows: list[np.ndarray] = []
    target_scores: list[float] = []

    for profile in perfect_profiles:
        if not profile.is_valid:
            continue
        feature_rows.append(profile.to_feature_vector())
        target_scores.append(float(target_perfect_score))

    for profile in moderate_profiles:
        if not profile.is_valid:
            continue
        feature_rows.append(profile.to_feature_vector())
        target_scores.append(float(target_moderate_score))

    for profile in fail_profiles:
        if not profile.is_valid:
            continue
        feature_rows.append(profile.to_feature_vector())
        target_scores.append(float(target_fail_score))

    if not feature_rows:
        raise ValueError("anchor_profiles contain no valid feature rows")

    features = np.vstack(feature_rows).astype(np.float64)
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    feature_stds = np.where(feature_stds > 1e-8, feature_stds, 1.0)
    normalized_features = (features - feature_means) / feature_stds

    target_probs = np.asarray(target_scores, dtype=np.float64) / 100.0
    target_probs = np.clip(target_probs, eps, 1.0 - eps)
    target_logits = np.log((1.0 - target_probs) / target_probs)

    reg = max(float(regularization), 0.0)
    feature_dim = int(normalized_features.shape[1])
    system_matrix = (
        normalized_features.T @ normalized_features
        + reg * np.eye(feature_dim, dtype=np.float64)
    )
    rhs = normalized_features.T @ target_logits
    try:
        weights = np.linalg.solve(system_matrix, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system_matrix) @ rhs

    bias = float(np.mean(target_logits - normalized_features @ weights))

    perfect_reference = [
        profile.perfect_distance
        for profile in perfect_profiles
        if np.isfinite(profile.perfect_distance)
    ]
    fail_reference = [
        profile.perfect_distance
        for profile in fail_profiles
        if np.isfinite(profile.perfect_distance)
    ]
    d100 = median_or_default(perfect_reference, default=float("inf"))
    d0 = median_or_default(fail_reference, default=float("inf"))

    logger.info(
        "Fitted multi-anchor sigmoid: perfect=%s moderate=%s fail=%s bias=%.4f",
        len(perfect_profiles),
        len(moderate_profiles),
        len(fail_profiles),
        bias,
    )

    return MultiAnchorSigmoidParams(
        weights=tuple(float(value) for value in weights.tolist()),
        bias=bias,
        feature_means=tuple(float(value) for value in feature_means.tolist()),
        feature_stds=tuple(float(value) for value in feature_stds.tolist()),
        d100=float(d100),
        d0=float(d0),
        epsilon=eps,
        target_perfect_score=float(target_perfect_score),
        target_moderate_score=float(target_moderate_score),
        target_fail_score=float(target_fail_score),
        score_floor=float(score_floor),
        score_ceil=float(score_ceil),
    )


def score_from_anchor_profile(
    profile: AnchorDistanceProfile,
    params: MultiAnchorSigmoidParams,
) -> float:
    if not profile.is_valid:
        return 0.0

    features = profile.to_feature_vector()
    means = np.asarray(params.feature_means, dtype=np.float64)
    stds = np.asarray(params.feature_stds, dtype=np.float64)
    weights = np.asarray(params.weights, dtype=np.float64)

    normalized = (features - means) / np.maximum(stds, 1e-8)
    logit = float(np.dot(normalized, weights) + float(params.bias))
    logit = float(np.clip(logit, -60.0, 60.0))
    score = 100.0 / (1.0 + math.exp(logit))
    return float(np.clip(score, params.score_floor, params.score_ceil))


def fit_sigmoid_from_anchor_distances(
    distances_100: Sequence[float],
    distances_0: Sequence[float],
    epsilon: float = 0.02,
) -> SigmoidCalibrationParams:
    eps = float(epsilon)
    if not (0.0 < eps < 0.5):
        raise ValueError("epsilon must be in interval (0, 0.5)")

    d100_candidates = np.asarray([float(value) for value in distances_100 if np.isfinite(value)], dtype=np.float64)
    d0_candidates = np.asarray([float(value) for value in distances_0 if np.isfinite(value)], dtype=np.float64)

    if d100_candidates.size == 0:
        raise ValueError("distances_100 is empty or contains no finite values")
    if d0_candidates.size == 0:
        raise ValueError("distances_0 is empty or contains no finite values")

    d100 = float(np.quantile(d100_candidates, CALIBRATION_D100_QUANTILE))
    d0 = float(np.quantile(d0_candidates, CALIBRATION_D0_QUANTILE))

    if d0 <= d100:
        separation = max(MIN_CALIBRATION_GAP, abs(d100) * 0.05)
        d0 = d100 + separation

    min_gap_from_slope = (2.0 * math.log((1.0 - eps) / eps)) / MAX_CALIBRATION_SLOPE
    enforced_gap = max(MIN_CALIBRATION_GAP, min_gap_from_slope)
    current_gap = d0 - d100
    if current_gap < enforced_gap:
        d0 = d100 + enforced_gap

    b = (d0 + d100) / 2.0
    a = (2.0 * math.log((1.0 - eps) / eps)) / (d0 - d100)

    logger.info("Fitted anchor sigmoid calibration: d100=%.4f d0=%.4f a=%.4f b=%.4f", d100, d0, a, b)

    return SigmoidCalibrationParams(
        d100=d100,
        d0=d0,
        a=float(a),
        b=float(b),
        epsilon=eps,
    )


def should_force_zero_by_zero_anchors(
    user_distance: float,
    user_zero_distance: float,
    calibration_params: SigmoidCalibrationParams,
    margin: float = DEFAULT_ZERO_ANCHOR_MARGIN,
) -> bool:
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    if not np.isfinite(user_distance) or not np.isfinite(user_zero_distance):
        return False

    if float(user_zero_distance) + float(margin) > float(user_distance):
        return False

    # Keep high-score region immune to aggressive zero-anchor ties.
    return float(user_distance) >= float(calibration_params.b)


def sigmoid_score(distance: float, params: SigmoidCalibrationParams) -> float:
    if not np.isfinite(distance):
        return 0.0

    exponent = float(params.a * (float(distance) - params.b))
    exponent = float(np.clip(exponent, -60.0, 60.0))
    score = 100.0 / (1.0 + math.exp(exponent))
    return float(np.clip(score, 0.0, 100.0))
