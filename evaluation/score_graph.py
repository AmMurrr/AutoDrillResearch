from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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


DATA_CSV = Path(__file__).resolve().parent / "data.csv"
VISUAL_DIR = Path(__file__).resolve().parent / "visual"
DEFAULT_RESULTS_CSV = VISUAL_DIR / "score_results.csv"
DEFAULT_PLOT_PATH = VISUAL_DIR / "score_graph.png"
DEFAULT_WORD = "there"
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


def _resolve_audio_path(path_value: str) -> Path:
    audio_path = Path(path_value).expanduser()
    if audio_path.is_absolute():
        return audio_path
    return PROJECT_ROOT / audio_path


def _infer_word(path_value: str, fallback: str) -> str:
    parts = Path(path_value).parts
    if "eval" in parts:
        eval_index = parts.index("eval")
        if len(parts) > eval_index + 1:
            return parts[eval_index + 1]
    return fallback


def load_items(csv_path: Path, default_word: str) -> list[EvaluationItem]:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        required_fields = {"id", "path", "class"}
        missing_fields = required_fields - fieldnames
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        items: list[EvaluationItem] = []
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip()
            raw_class = (row.get("class") or "").strip().lower()
            raw_id = (row.get("id") or "").strip()

            if not raw_path:
                raise ValueError(f"{csv_path}:{row_number} has empty path")
            if raw_class not in CLASS_ORDER:
                raise ValueError(
                    f"{csv_path}:{row_number} has unsupported class '{raw_class}'"
                )

            try:
                item_id = int(raw_id)
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{row_number} has invalid id '{raw_id}'") from exc

            audio_path = _resolve_audio_path(raw_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

            items.append(
                EvaluationItem(
                    item_id=item_id,
                    path=raw_path,
                    expected_class=raw_class,
                    word=_infer_word(raw_path, default_word),
                )
            )

    return items


def _run_classic(item: EvaluationItem) -> tuple[float, str, str, str]:
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
    )


def _run_neural(item: EvaluationItem) -> tuple[float, str, str, str]:
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
                score, status, verdict, reason = runners[approach](item)
            except Exception as exc:
                score = 0.0
                status = "error"
                verdict = "error"
                reason = f"{type(exc).__name__}: {exc}"

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
                )
            )
            print(
                f"{approach:7s} id={item.item_id:02d} "
                f"class={item.expected_class:8s} score={score:6.2f} status={status}"
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
                    "score": f"{result.score:.6f}",
                    "status": result.status,
                    "verdict": result.verdict,
                    "reason": result.reason,
                    "elapsed_seconds": f"{result.elapsed_seconds:.6f}",
                }
            )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def save_score_plot(results: list[RunResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            scores = [result.score for result in group_results]
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
    ax.set_title("Pronunciation scores for 'there'")
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
        description="Run classic and neural pronunciation scoring for evaluation/data.csv."
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=DATA_CSV,
        help="CSV with id,path,class columns.",
    )
    parser.add_argument(
        "--word",
        default=DEFAULT_WORD,
        help="Fallback transcript word when it cannot be inferred from path.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Where to save per-run scores.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Where to save the matplotlib graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_items(args.data_csv, default_word=args.word)
    results = run_evaluation(items)
    save_results(results, args.results_csv)
    save_score_plot(results, args.plot)

    print(f"Saved results: {args.results_csv}")
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
