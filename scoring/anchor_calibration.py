from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable, Sequence, TypeVar

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR_ROOT = WORKSPACE_ROOT / "data" / "ref"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
ANCHOR_QUALITIES = {"perfect", "wrong", "moderate"}


@dataclass(frozen=True)
class WordAnchorSet:
    word: str
    perfect_paths: tuple[str, ...]
    wrong_paths: tuple[str, ...]
    moderate_paths: tuple[str, ...]
    empty_paths: tuple[str, ...]

    @property
    def zero_paths(self) -> tuple[str, ...]:
        return self.wrong_paths + self.moderate_paths + self.empty_paths

    @property
    def has_required_anchors(self) -> bool:
        return bool(self.perfect_paths) and bool(self.zero_paths)


@dataclass(frozen=True)
class SigmoidCalibrationParams:
    d100: float
    d0: float
    a: float
    b: float
    epsilon: float


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
                "wrong": [],
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
            "wrong": [],
            "moderate": [],
        },
    )

    perfect_paths = _limited(_deduplicate_paths(entry["perfect"]), max_anchors_per_class)
    wrong_paths = _limited(_deduplicate_paths(entry["wrong"]), max_anchors_per_class)
    moderate_paths = _limited(_deduplicate_paths(entry["moderate"]), max_anchors_per_class)
    empty_paths = _limited(_deduplicate_paths(empty_word_paths), max_anchors_per_class)

    return WordAnchorSet(
        word=normalized_word,
        perfect_paths=perfect_paths,
        wrong_paths=wrong_paths,
        moderate_paths=moderate_paths,
        empty_paths=empty_paths,
    )


def list_anchor_words(anchor_root: str | Path | None = None) -> list[str]:
    root = _resolve_anchor_root(anchor_root)
    grouped, _ = _scan_anchor_groups(root)
    words = [word for word, entry in grouped.items() if entry.get("perfect")]
    return sorted(set(words))


def describe_anchor_set(anchor_set: WordAnchorSet) -> dict[str, int]:
    return {
        "perfect": len(anchor_set.perfect_paths),
        "wrong": len(anchor_set.wrong_paths),
        "moderate": len(anchor_set.moderate_paths),
        "empty_word": len(anchor_set.empty_paths),
        "zero_total": len(anchor_set.zero_paths),
    }


def is_known_zero_anchor(audio_path: str, anchor_set: WordAnchorSet) -> bool:
    resolved = _resolve_audio_path(audio_path)
    return resolved in set(anchor_set.zero_paths)


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

    d100 = float(np.median(d100_candidates))
    d0 = float(np.median(d0_candidates))

    if d0 <= d100:
        separation = max(1e-3, abs(d100) * 0.05)
        d0 = d100 + separation

    b = (d0 + d100) / 2.0
    a = (2.0 * math.log((1.0 - eps) / eps)) / (d0 - d100)

    return SigmoidCalibrationParams(
        d100=d100,
        d0=d0,
        a=float(a),
        b=float(b),
        epsilon=eps,
    )


def sigmoid_score(distance: float, params: SigmoidCalibrationParams) -> float:
    if not np.isfinite(distance):
        return 0.0

    exponent = float(params.a * (float(distance) - params.b))
    exponent = float(np.clip(exponent, -60.0, 60.0))
    score = 100.0 / (1.0 + math.exp(exponent))
    return float(np.clip(score, 0.0, 100.0))