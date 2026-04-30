from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DIPLOMA_LOG_LEVEL", "WARNING")

import pandas as pd  # noqa: E402
import psutil  # noqa: E402


EVAL_DIR = PROJECT_ROOT / "data" / "eval"
RESOURCES_DIR = PROJECT_ROOT / "evaluation" / "resources"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
CLASS_ORDER = ("perfect", "moderate", "fail")
APPROACH_ORDER = ("classic", "neural")
DEFAULT_REPEATS = 3
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.01
_CALIBRATION_CACHED_APPROACHES: set[str] = set()


@dataclass(frozen=True)
class EvaluationItem:
    item_id: int
    path: str
    word: str
    expected_class: str


@dataclass(frozen=True)
class MeasuredCall:
    result: object | None
    wall_seconds: float
    cpu_seconds: float
    ram_baseline_mb: float
    ram_peak_mb: float
    ram_delta_peak_mb: float
    error: str


@dataclass(frozen=True)
class ResourceResult:
    run_id: int
    repeat_index: int
    item_id: int
    word: str
    expected_class: str
    path: str
    approach: str
    status: str
    score: float | None
    wall_seconds: float
    cpu_seconds: float
    ram_baseline_mb: float
    ram_peak_mb: float
    ram_delta_peak_mb: float
    error: str


@dataclass(frozen=True)
class ColdRamResult:
    approach: str
    cold_start_item_id: int
    cold_start_word: str
    cold_start_path: str
    cold_start_ram_mb: float
    cold_error: str


def _is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def _path_for_csv(audio_path: Path) -> str:
    try:
        return audio_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return audio_path.as_posix()


def _resolve_audio_path(path_value: str) -> Path:
    audio_path = Path(path_value).expanduser()
    if audio_path.is_absolute():
        return audio_path
    return PROJECT_ROOT / audio_path


def load_items(eval_dir: Path, words: Iterable[str] | None = None) -> list[EvaluationItem]:
    eval_dir = eval_dir.expanduser()
    if not eval_dir.is_absolute():
        eval_dir = PROJECT_ROOT / eval_dir
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory does not exist: {eval_dir}")
    if not eval_dir.is_dir():
        raise NotADirectoryError(f"Evaluation path is not a directory: {eval_dir}")

    selected_words = {word.strip() for word in words or () if word.strip()}
    word_dirs = [
        path
        for path in sorted(eval_dir.iterdir(), key=lambda value: value.name.lower())
        if path.is_dir() and (not selected_words or path.name in selected_words)
    ]

    items: list[EvaluationItem] = []
    next_item_id = 1
    for word_dir in word_dirs:
        for class_name in CLASS_ORDER:
            class_dir = word_dir / class_name
            if not class_dir.exists():
                continue
            if not class_dir.is_dir():
                raise NotADirectoryError(f"Evaluation class path is not a directory: {class_dir}")

            audio_files = sorted(
                (path for path in class_dir.iterdir() if _is_audio_file(path)),
                key=lambda path: path.name.lower(),
            )
            for audio_path in audio_files:
                items.append(
                    EvaluationItem(
                        item_id=next_item_id,
                        path=_path_for_csv(audio_path),
                        word=word_dir.name,
                        expected_class=class_name,
                    )
                )
                next_item_id += 1

    if selected_words:
        found_words = {item.word for item in items}
        missing_words = sorted(selected_words.difference(found_words))
        if missing_words:
            raise ValueError(f"No evaluation audio found for words: {', '.join(missing_words)}")

    return items


def _run_classic(item: EvaluationItem):
    classic_pipeline = _pipeline_for_approach("classic")
    return classic_pipeline.analyze(
        user_audio_path=str(_resolve_audio_path(item.path)),
        transcript=item.word,
        use_vosk=False,
    )


def _run_neural(item: EvaluationItem):
    neural_pipeline = _pipeline_for_approach("neural")
    return neural_pipeline.analyze(
        user_audio_path=str(_resolve_audio_path(item.path)),
        transcript=item.word,
        use_vosk=False,
    )


def _runner_for_approach(approach: str) -> Callable[[EvaluationItem], object]:
    if approach == "classic":
        return _run_classic
    if approach == "neural":
        return _run_neural
    raise ValueError(f"Unknown approach: {approach}")


def _score_from_result(approach: str, result: object | None) -> float | None:
    if result is None:
        return None
    if approach == "classic":
        return float(getattr(result, "dtw_score"))
    if approach == "neural":
        return float(getattr(result, "pronunciation_score"))
    return None


def _status_from_result(result: object | None, error: str) -> str:
    if error:
        return "error"
    if result is None:
        return "error"
    return str(getattr(result, "status", "ok"))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def _pipeline_for_approach(approach: str):
    if approach == "classic":
        import classic_approach.pipeline as classic_pipeline

        return classic_pipeline
    if approach == "neural":
        import neural_approach.pipeline as neural_pipeline

        return neural_pipeline
    raise ValueError(f"Unknown approach: {approach}")


def enable_benchmark_calibration_caches(approaches: Iterable[str] = APPROACH_ORDER) -> None:
    """Cache first per-word calibration during warm-up without editing approach modules."""
    selected_approaches = tuple(dict.fromkeys(approaches))
    for approach in selected_approaches:
        if approach in _CALIBRATION_CACHED_APPROACHES:
            continue
        pipeline = _pipeline_for_approach(approach)
        pipeline._build_anchor_calibration = lru_cache(maxsize=None)(  # noqa: SLF001
            pipeline._build_anchor_calibration  # noqa: SLF001
        )
        _CALIBRATION_CACHED_APPROACHES.add(approach)


def measure_call(
    fn: Callable[[], object],
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> MeasuredCall:
    process = psutil.Process()
    baseline_bytes = int(process.memory_info().rss)
    peak_bytes = baseline_bytes
    stop_event = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_bytes
        while not stop_event.is_set():
            peak_bytes = max(peak_bytes, int(process.memory_info().rss))
            stop_event.wait(sample_interval_seconds)

    sampler_thread = threading.Thread(target=sample_memory, daemon=True)
    sampler_thread.start()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result: object | None = None
    error = ""
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - benchmark must record failed runs.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        peak_bytes = max(peak_bytes, int(process.memory_info().rss))
        stop_event.set()
        sampler_thread.join()

    return MeasuredCall(
        result=result,
        wall_seconds=float(wall_seconds),
        cpu_seconds=float(cpu_seconds),
        ram_baseline_mb=baseline_bytes / 1024 / 1024,
        ram_peak_mb=peak_bytes / 1024 / 1024,
        ram_delta_peak_mb=(peak_bytes - baseline_bytes) / 1024 / 1024,
        error=error,
    )


def _warmup_items(items: list[EvaluationItem]) -> list[EvaluationItem]:
    by_word: dict[str, EvaluationItem] = {}
    for class_name in CLASS_ORDER:
        for item in items:
            if item.expected_class == class_name and item.word not in by_word:
                by_word[item.word] = item
    return [by_word[word] for word in sorted(by_word)]


def warm_up(items: list[EvaluationItem], approaches: Iterable[str] = APPROACH_ORDER) -> None:
    warmup_items = _warmup_items(items)
    if not warmup_items:
        raise ValueError("Cannot warm up benchmark: no evaluation items")

    selected_approaches = tuple(dict.fromkeys(approaches))
    print("Warm-up started; these runs are not written to CSV.")
    for item in warmup_items:
        for approach in selected_approaches:
            runner = _runner_for_approach(approach)
            try:
                runner(item)
                print(f"warmup {approach:7s} word={item.word} path={item.path}")
            except Exception as exc:  # noqa: BLE001 - warm-up should continue across approaches.
                print(
                    f"warmup {approach:7s} word={item.word} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
    print("Warm-up finished.")


def measure_cold_ram(
    items: list[EvaluationItem],
    approaches: Iterable[str],
    sample_interval_seconds: float,
) -> list[ColdRamResult]:
    cold_items = _warmup_items(items)
    if not cold_items:
        raise ValueError("Cannot measure cold RAM: no evaluation items")

    cold_item = cold_items[0]
    results: list[ColdRamResult] = []
    for approach in tuple(dict.fromkeys(approaches)):
        print(
            f"Cold-start RAM measurement: {approach:7s} "
            f"word={cold_item.word} path={cold_item.path}"
        )

        def cold_start_call(approach=approach, item=cold_item):
            enable_benchmark_calibration_caches((approach,))
            runner = _runner_for_approach(approach)
            return runner(item)

        measured = measure_call(
            cold_start_call,
            sample_interval_seconds=sample_interval_seconds,
        )
        results.append(
            ColdRamResult(
                approach=approach,
                cold_start_item_id=cold_item.item_id,
                cold_start_word=cold_item.word,
                cold_start_path=cold_item.path,
                cold_start_ram_mb=measured.ram_delta_peak_mb,
                cold_error=measured.error,
            )
        )
        print(
            f"cold {approach:7s} wall={measured.wall_seconds:.3f}s "
            f"cpu={measured.cpu_seconds:.3f}s "
            f"peak_ram={measured.ram_peak_mb:.1f}MB "
            f"delta_ram={measured.ram_delta_peak_mb:.1f}MB"
        )
    return results


def run_benchmark(
    items: list[EvaluationItem],
    repeats: int,
    sample_interval_seconds: float,
    approaches: Iterable[str] = APPROACH_ORDER,
    run_id_start: int = 1,
) -> list[ResourceResult]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    selected_approaches = tuple(dict.fromkeys(approaches))
    results: list[ResourceResult] = []
    run_id = run_id_start
    total_runs = len(items) * len(selected_approaches) * repeats
    for repeat_index in range(1, repeats + 1):
        for item in items:
            for approach in selected_approaches:
                runner = _runner_for_approach(approach)
                measured = measure_call(
                    lambda runner=runner, item=item: runner(item),
                    sample_interval_seconds=sample_interval_seconds,
                )
                score = _score_from_result(approach, measured.result)
                status = _status_from_result(measured.result, measured.error)
                results.append(
                    ResourceResult(
                        run_id=run_id,
                        repeat_index=repeat_index,
                        item_id=item.item_id,
                        word=item.word,
                        expected_class=item.expected_class,
                        path=item.path,
                        approach=approach,
                        status=status,
                        score=score,
                        wall_seconds=measured.wall_seconds,
                        cpu_seconds=measured.cpu_seconds,
                        ram_baseline_mb=measured.ram_baseline_mb,
                        ram_peak_mb=measured.ram_peak_mb,
                        ram_delta_peak_mb=measured.ram_delta_peak_mb,
                        error=measured.error,
                    )
                )
                print(
                    f"[{run_id}/{total_runs}] {approach:7s} "
                    f"repeat={repeat_index} word={item.word} "
                    f"class={item.expected_class} status={status} "
                    f"wall={measured.wall_seconds:.3f}s "
                    f"cpu={measured.cpu_seconds:.3f}s "
                    f"peak_ram={measured.ram_peak_mb:.1f}MB"
                )
                run_id += 1
    return results


def save_resource_results(results: list[ResourceResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "repeat_index",
        "item_id",
        "word",
        "class",
        "path",
        "approach",
        "status",
        "score",
        "wall_seconds",
        "cpu_seconds",
        "ram_baseline_mb",
        "ram_peak_mb",
        "ram_delta_peak_mb",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "run_id": result.run_id,
                    "repeat_index": result.repeat_index,
                    "item_id": result.item_id,
                    "word": result.word,
                    "class": result.expected_class,
                    "path": result.path,
                    "approach": result.approach,
                    "status": result.status,
                    "score": _format_optional_float(result.score),
                    "wall_seconds": f"{result.wall_seconds:.6f}",
                    "cpu_seconds": f"{result.cpu_seconds:.6f}",
                    "ram_baseline_mb": f"{result.ram_baseline_mb:.6f}",
                    "ram_peak_mb": f"{result.ram_peak_mb:.6f}",
                    "ram_delta_peak_mb": f"{result.ram_delta_peak_mb:.6f}",
                    "error": result.error,
                }
            )


def save_resource_summary(results_path: Path, output_path: Path) -> None:
    data = pd.read_csv(results_path)
    if data.empty:
        raise ValueError("Cannot build resource summary from empty results")

    aggregations = {
        "measured_runs": ("run_id", "count"),
        "unique_files_count": ("path", "nunique"),
        "wall_median_seconds": ("wall_seconds", "median"),
        "wall_p95_seconds": ("wall_seconds", lambda values: float(values.quantile(0.95))),
        "cpu_median_seconds": ("cpu_seconds", "median"),
        "cpu_p95_seconds": ("cpu_seconds", lambda values: float(values.quantile(0.95))),
    }
    summary = (
        data.groupby("approach", as_index=False)
        .agg(**aggregations)
        .sort_values("approach")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def save_resource_ram_summary(
    results_path: Path,
    cold_results: list[ColdRamResult],
    output_path: Path,
) -> None:
    data = pd.read_csv(results_path)
    if data.empty:
        raise ValueError("Cannot build RAM summary from empty results")

    warm_summary = (
        data.groupby("approach", as_index=False)
        .agg(
            measured_runs=("run_id", "count"),
            warm_start_ram_mb=("ram_baseline_mb", "median"),
        )
        .sort_values("approach")
    )
    cold_summary = pd.DataFrame(
        [
            {
                "approach": result.approach,
                "cold_start_item_id": result.cold_start_item_id,
                "cold_start_word": result.cold_start_word,
                "cold_start_path": result.cold_start_path,
                "cold_start_ram_mb": result.cold_start_ram_mb,
                "cold_error": result.cold_error,
            }
            for result in cold_results
        ]
    )
    summary = cold_summary.merge(warm_summary, on="approach", how="outer").sort_values("approach")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def _load_benchmark_items(args: argparse.Namespace) -> tuple[Path, list[EvaluationItem]]:
    eval_dir = args.eval_dir.expanduser()
    if not eval_dir.is_absolute():
        eval_dir = PROJECT_ROOT / eval_dir

    items = load_items(eval_dir=eval_dir, words=args.words)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        items = items[: args.limit]
    if not items:
        raise ValueError(f"No audio files found in evaluation directory: {eval_dir}")
    return eval_dir, items


def _run_current_process_benchmark(
    items: list[EvaluationItem],
    output_dir: Path,
    repeats: int,
    sample_interval_seconds: float,
    approaches: Iterable[str] = APPROACH_ORDER,
    results_path: Path | None = None,
    summary_path: Path | None = None,
    ram_summary_path: Path | None = None,
    write_summary: bool = True,
    write_ram_summary: bool = True,
) -> None:
    selected_approaches = tuple(dict.fromkeys(approaches))
    print(f"Loaded {len(items)} evaluation files.")
    cold_results = measure_cold_ram(
        items=items,
        approaches=selected_approaches,
        sample_interval_seconds=sample_interval_seconds,
    )
    enable_benchmark_calibration_caches(selected_approaches)
    warm_up(items, selected_approaches)
    results = run_benchmark(
        items=items,
        repeats=repeats,
        sample_interval_seconds=sample_interval_seconds,
        approaches=selected_approaches,
    )

    if results_path is None:
        results_path = output_dir / "resource_results.csv"
    if summary_path is None:
        summary_path = output_dir / "resource_summary.csv"
    if ram_summary_path is None:
        ram_summary_path = output_dir / "resource_ram_summary.csv"
    save_resource_results(results, results_path)
    if write_summary:
        save_resource_summary(results_path, summary_path)
    if write_ram_summary:
        save_resource_ram_summary(results_path, cold_results, ram_summary_path)
    print(f"Saved resource results: {results_path}")
    if write_summary:
        print(f"Saved resource summary: {summary_path}")
    if write_ram_summary:
        print(f"Saved RAM summary: {ram_summary_path}")


def _run_isolated_benchmark(args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_paths: list[Path] = []
    partial_ram_paths: list[Path] = []
    for approach in APPROACH_ORDER:
        partial_path = output_dir / f".resource_results.{approach}.csv"
        partial_ram_path = output_dir / f".resource_ram_summary.{approach}.csv"
        partial_paths.append(partial_path)
        partial_ram_paths.append(partial_ram_path)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--eval-dir",
            str(args.eval_dir),
            "--output-dir",
            str(output_dir),
            "--repeats",
            str(args.repeats),
            "--sample-interval",
            str(args.sample_interval),
            "--worker-approach",
            approach,
            "--worker-results-path",
            str(partial_path),
            "--worker-ram-summary-path",
            str(partial_ram_path),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.words:
            command.append("--words")
            command.extend(args.words)

        print(f"Starting isolated {approach} benchmark process.", flush=True)
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    frames = [pd.read_csv(path) for path in partial_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined["run_id"] = range(1, len(combined) + 1)
    ram_frames = [pd.read_csv(path) for path in partial_ram_paths]
    combined_ram = pd.concat(ram_frames, ignore_index=True)

    results_path = output_dir / "resource_results.csv"
    summary_path = output_dir / "resource_summary.csv"
    ram_summary_path = output_dir / "resource_ram_summary.csv"
    combined.to_csv(results_path, index=False)
    save_resource_summary(results_path, summary_path)
    combined_ram.to_csv(ram_summary_path, index=False)

    for partial_path in partial_paths:
        partial_path.unlink(missing_ok=True)
    for partial_ram_path in partial_ram_paths:
        partial_ram_path.unlink(missing_ok=True)

    print(f"Saved resource results: {results_path}")
    print(f"Saved resource summary: {summary_path}")
    print(f"Saved RAM summary: {ram_summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure steady-state CPU time, wall time and RAM for pronunciation scoring. "
            "Warm-up runs are executed before measured runs and are not written to CSV."
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
        default=RESOURCES_DIR,
        help="Directory where resource CSV files are saved.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Measured repeats per file and approach after warm-up.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help="RAM sampling interval in seconds.",
    )
    parser.add_argument(
        "--words",
        nargs="*",
        default=None,
        help="Optional subset of words to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests. Full research runs should omit this.",
    )
    parser.add_argument(
        "--no-isolated-processes",
        action="store_true",
        help=(
            "Run all approaches in this Python process. This is faster, but RAM metrics "
            "include memory shared by all imported approaches and are not suitable for "
            "clean per-approach resource comparison."
        ),
    )
    parser.add_argument(
        "--worker-approach",
        choices=APPROACH_ORDER,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-results-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-ram-summary-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    _, items = _load_benchmark_items(args)

    if args.worker_approach is not None:
        if args.worker_results_path is None:
            raise ValueError("--worker-results-path is required with --worker-approach")
        if args.worker_ram_summary_path is None:
            raise ValueError("--worker-ram-summary-path is required with --worker-approach")
        worker_results_path = args.worker_results_path.expanduser()
        if not worker_results_path.is_absolute():
            worker_results_path = PROJECT_ROOT / worker_results_path
        worker_ram_summary_path = args.worker_ram_summary_path.expanduser()
        if not worker_ram_summary_path.is_absolute():
            worker_ram_summary_path = PROJECT_ROOT / worker_ram_summary_path
        _run_current_process_benchmark(
            items=items,
            output_dir=worker_results_path.parent,
            repeats=args.repeats,
            sample_interval_seconds=args.sample_interval,
            approaches=(args.worker_approach,),
            results_path=worker_results_path,
            ram_summary_path=worker_ram_summary_path,
            write_summary=False,
            write_ram_summary=True,
        )
        return

    if args.no_isolated_processes:
        print(
            "WARNING: --no-isolated-processes makes RAM metrics process-wide; "
            "use the default isolated mode for research comparison."
        )
        _run_current_process_benchmark(
            items=items,
            output_dir=output_dir,
            repeats=args.repeats,
            sample_interval_seconds=args.sample_interval,
            approaches=APPROACH_ORDER,
        )
        return

    _run_isolated_benchmark(args, output_dir)


if __name__ == "__main__":
    main()
