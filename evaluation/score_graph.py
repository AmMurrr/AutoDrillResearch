from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "diploma-matplotlib"))

import matplotlib  # noqa: E402


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classic_approach.pipeline import analyze as analyze_classic  # noqa: E402
from neural_approach.pipeline import analyze as analyze_neural  # noqa: E402


EVAL_DIR = PROJECT_ROOT / "data" / "eval"
VISUAL_DIR = Path(__file__).resolve().parent / "visual"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
CLASS_ORDER = ("perfect", "moderate", "fail")
APPROACH_ORDER = ("classic", "neural")
CLASS_COLORS = {
    "perfect": "#2e7d32",
    "moderate": "#f9a825",
    "fail": "#c62828",
}


@dataclass(frozen=True)
class EvaluationItem:
    item_id: int
    path: str
    expected_class: str
    word: str


@dataclass(frozen=True)
class RunResult:
    item_id: int
    path: str
    expected_class: str
    word: str
    approach: str
    score: float
    status: str
    verdict: str
    reason: str
    elapsed_seconds: float
    raw_distance: float | None = None
    moderate_distance: float | None = None
    fail_distance: float | None = None
    duration_distance: float | None = None
    duration_score: float | None = None
    embedding_score: float | None = None


RunnerOutput = tuple[float, str, str, str, dict[str, float | None]]


def _resolve_audio_path(path_value: str) -> Path:
    audio_path = Path(path_value).expanduser()
    if audio_path.is_absolute():
        return audio_path
    return PROJECT_ROOT / audio_path


def _path_for_csv(audio_path: Path) -> str:
    try:
        return audio_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return audio_path.as_posix()


def _is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def load_items(eval_dir: Path) -> dict[str, list[EvaluationItem]]:
    eval_dir = eval_dir.expanduser()
    if not eval_dir.is_absolute():
        eval_dir = PROJECT_ROOT / eval_dir
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory does not exist: {eval_dir}")
    if not eval_dir.is_dir():
        raise NotADirectoryError(f"Evaluation path is not a directory: {eval_dir}")

    items_by_word: dict[str, list[EvaluationItem]] = {}
    word_dirs = sorted(
        (path for path in eval_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    for word_dir in word_dirs:
        word = word_dir.name
        items: list[EvaluationItem] = []
        next_item_id = 1

        for class_name in CLASS_ORDER:
            class_dir = word_dir / class_name
            if not class_dir.exists():
                continue
            if not class_dir.is_dir():
                raise NotADirectoryError(
                    f"Evaluation class path is not a directory: {class_dir}"
                )

            audio_files = sorted(
                (path for path in class_dir.iterdir() if _is_audio_file(path)),
                key=lambda path: path.name.lower(),
            )
            for audio_path in audio_files:
                items.append(
                    EvaluationItem(
                        item_id=next_item_id,
                        path=_path_for_csv(audio_path),
                        expected_class=class_name,
                        word=word,
                    )
                )
                next_item_id += 1

        if items:
            items_by_word[word] = items

    return items_by_word


def _run_classic(item: EvaluationItem) -> RunnerOutput:
    result = analyze_classic(
        user_audio_path=str(_resolve_audio_path(item.path)),
        transcript=item.word,
        use_vosk=False,
    )
    return (
        float(result.dtw_score),
        result.status,
        result.verdict,
        result.reason,
        {
            "raw_distance": float(result.distance),
            "moderate_distance": result.moderate_distance,
            "fail_distance": result.fail_distance,
            "duration_distance": None,
            "duration_score": None,
            "embedding_score": None,
        },
    )


def _run_neural(item: EvaluationItem) -> RunnerOutput:
    result = analyze_neural(
        user_audio_path=str(_resolve_audio_path(item.path)),
        transcript=item.word,
        use_vosk=False,
    )
    return (
        float(result.pronunciation_score),
        result.status,
        result.verdict,
        result.reason,
        {
            "raw_distance": result.raw_distance,
            "moderate_distance": result.moderate_raw_distance,
            "fail_distance": result.fail_raw_distance,
            "duration_distance": result.duration_distance,
            "duration_score": result.duration_score,
            "embedding_score": getattr(result, "embedding_score", None),
        },
    )


def run_evaluation(items: Iterable[EvaluationItem]) -> list[RunResult]:
    results: list[RunResult] = []
    runners = {
        "classic": _run_classic,
        "neural": _run_neural,
    }

    for item in items:
        for approach in APPROACH_ORDER:
            started_at = perf_counter()
            try:
                score, status, verdict, reason, diagnostics = runners[approach](item)
            except Exception as exc:
                score = float("nan")
                status = "error"
                verdict = "error"
                reason = f"{type(exc).__name__}: {exc}"
                diagnostics = {
                    "raw_distance": None,
                    "moderate_distance": None,
                    "fail_distance": None,
                    "duration_distance": None,
                    "duration_score": None,
                    "embedding_score": None,
                }

            elapsed_seconds = perf_counter() - started_at
            results.append(
                RunResult(
                    item_id=item.item_id,
                    path=item.path,
                    expected_class=item.expected_class,
                    word=item.word,
                    approach=approach,
                    score=score,
                    status=status,
                    verdict=verdict,
                    reason=reason,
                    elapsed_seconds=elapsed_seconds,
                    raw_distance=diagnostics["raw_distance"],
                    moderate_distance=diagnostics["moderate_distance"],
                    fail_distance=diagnostics["fail_distance"],
                    duration_distance=diagnostics["duration_distance"],
                    duration_score=diagnostics["duration_score"],
                    embedding_score=diagnostics["embedding_score"],
                )
            )
            print(
                f"{approach:7s} id={item.item_id:02d} "
                f"class={item.expected_class:8s} score={_format_score_for_log(score)} "
                f"status={status}"
            )

    return results


def save_results(results: list[RunResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "path",
                "word",
                "class",
                "approach",
                "score",
                "status",
                "verdict",
                "reason",
                "elapsed_seconds",
                "raw_distance",
                "moderate_distance",
                "fail_distance",
                "duration_distance",
                "duration_score",
                "embedding_score",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "id": result.item_id,
                    "path": result.path,
                    "word": result.word,
                    "class": result.expected_class,
                    "approach": result.approach,
                    "score": _format_optional_float(result.score),
                    "status": result.status,
                    "verdict": result.verdict,
                    "reason": result.reason,
                    "elapsed_seconds": f"{result.elapsed_seconds:.6f}",
                    "raw_distance": _format_optional_float(result.raw_distance),
                    "moderate_distance": _format_optional_float(result.moderate_distance),
                    "fail_distance": _format_optional_float(result.fail_distance),
                    "duration_distance": _format_optional_float(result.duration_distance),
                    "duration_score": _format_optional_float(result.duration_score),
                    "embedding_score": _format_optional_float(result.embedding_score),
                }
            )


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def _format_score_for_log(score: float) -> str:
    if not math.isfinite(float(score)):
        return "   n/a"
    return f"{score:6.2f}"


def _safe_filename(value: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() or char in "-_" else "_"
        for char in value.strip()
    )
    return normalized.strip("_") or "word"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def save_score_plot(results: list[RunResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    word = results[0].word if results else output_path.stem

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x_positions: dict[tuple[str, str], float] = {}
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for approach_index, approach in enumerate(APPROACH_ORDER):
        group_offset = approach_index * (len(CLASS_ORDER) + 1)
        for class_index, class_name in enumerate(CLASS_ORDER):
            x = group_offset + class_index
            x_positions[(approach, class_name)] = float(x)
            tick_positions.append(float(x))
            tick_labels.append(f"{approach}\n{class_name}")

    for approach in APPROACH_ORDER:
        for class_name in CLASS_ORDER:
            group_results = [
                result
                for result in results
                if result.approach == approach and result.expected_class == class_name
            ]
            scores = [
                result.score for result in group_results if math.isfinite(result.score)
            ]
            if not scores:
                continue

            center_x = x_positions[(approach, class_name)]
            if len(scores) == 1:
                jitter = [0.0]
            else:
                step = 0.28 / max(len(scores) - 1, 1)
                jitter = [-0.14 + step * index for index in range(len(scores))]

            ax.scatter(
                [center_x + offset for offset in jitter],
                scores,
                s=52,
                color=CLASS_COLORS[class_name],
                alpha=0.82,
                edgecolors="#1f1f1f",
                linewidths=0.45,
                label=class_name,
            )

            mean_score = _mean(scores)
            if mean_score is not None:
                ax.hlines(
                    mean_score,
                    center_x - 0.26,
                    center_x + 0.26,
                    color="#111111",
                    linewidth=2.0,
                )
                ax.text(
                    center_x,
                    mean_score + 2.2,
                    f"{mean_score:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#111111",
                )

    separator_x = len(CLASS_ORDER) - 0.5
    ax.axvline(separator_x, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(f"Pronunciation scores for '{word}'")
    ax.set_ylabel("Score")
    ax.set_ylim(-2, 104)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.grid(axis="y", linestyle=":", alpha=0.55)

    handles_by_label = {}
    for handle, label in zip(*ax.get_legend_handles_labels(), strict=False):
        handles_by_label.setdefault(label, handle)
    ax.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        title="Class",
        loc="upper right",
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run classic and neural pronunciation scoring for audio files in data/eval."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=EVAL_DIR,
        help="Directory with <word>/<perfect|moderate|fail> audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUAL_DIR,
        help="Directory where per-word CSV files and plots are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items_by_word = load_items(args.eval_dir)
    if not items_by_word:
        raise ValueError(
            f"No audio files found in evaluation directory: {args.eval_dir}"
        )

    for word, items in items_by_word.items():
        safe_word = _safe_filename(word)
        results_csv = args.output_dir / f"score_results_{safe_word}.csv"
        plot_path = args.output_dir / f"score_graph_{safe_word}.png"

        print(f"\nWord '{word}': {len(items)} audio files")
        results = run_evaluation(items)
        save_results(results, results_csv)
        save_score_plot(results, plot_path)

        print(f"Saved results: {results_csv}")
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
