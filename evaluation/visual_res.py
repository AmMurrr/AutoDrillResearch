from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "diploma-matplotlib"))

import matplotlib  # noqa: E402


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
APPROACH_ORDER = ("classic", "neural")
APPROACH_COLORS = {
    "classic": "#2f6f73",
    "neural": "#9b4d3f",
}
TIME_METRICS = (
    ("wall_median_seconds", "Wall median"),
    ("wall_p95_seconds", "Wall p95"),
    ("cpu_median_seconds", "CPU median"),
    ("cpu_p95_seconds", "CPU p95"),
)
RAM_METRICS = (
    ("cold_start_ram_mb", "Cold start"),
    ("warm_start_ram_mb", "Warm start"),
)


def _approaches_in_order(data: pd.DataFrame) -> list[str]:
    approaches = list(dict.fromkeys(APPROACH_ORDER + tuple(sorted(data["approach"].unique()))))
    return [approach for approach in approaches if approach in set(data["approach"])]


def _require_columns(data: pd.DataFrame, columns: set[str], csv_path: Path) -> None:
    missing = sorted(columns.difference(data.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")


def _clean_results(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in (
        "run_id",
        "repeat_index",
        "item_id",
        "score",
        "wall_seconds",
        "cpu_seconds",
        "ram_baseline_mb",
        "ram_peak_mb",
        "ram_delta_peak_mb",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in ("approach", "word", "class", "path", "status"):
        if column in data.columns:
            data[column] = data[column].fillna("").astype(str).str.strip()
    return data


def load_resource_data(resources_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_path = resources_dir / "resource_results.csv"
    summary_path = resources_dir / "resource_summary.csv"
    ram_summary_path = resources_dir / "resource_ram_summary.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Resource results CSV not found: {results_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Resource summary CSV not found: {summary_path}")
    if not ram_summary_path.is_file():
        raise FileNotFoundError(f"RAM summary CSV not found: {ram_summary_path}")

    results = _clean_results(pd.read_csv(results_path))
    summary = pd.read_csv(summary_path)
    ram_summary = pd.read_csv(ram_summary_path)
    _require_columns(
        results,
        {
            "run_id",
            "repeat_index",
            "item_id",
            "word",
            "class",
            "path",
            "approach",
            "wall_seconds",
            "cpu_seconds",
            "ram_baseline_mb",
            "ram_peak_mb",
            "ram_delta_peak_mb",
        },
        results_path,
    )
    _require_columns(
        summary,
        {
            "approach",
            "measured_runs",
            "unique_files_count",
            "wall_median_seconds",
            "wall_p95_seconds",
            "cpu_median_seconds",
            "cpu_p95_seconds",
        },
        summary_path,
    )
    _require_columns(
        ram_summary,
        {
            "approach",
            "cold_start_word",
            "cold_start_path",
            "cold_start_ram_mb",
            "warm_start_ram_mb",
        },
        ram_summary_path,
    )

    for column in summary.columns:
        if column != "approach":
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    summary["approach"] = summary["approach"].fillna("").astype(str).str.strip()
    for column in ram_summary.columns:
        if column not in {"approach", "cold_start_word", "cold_start_path", "cold_error"}:
            ram_summary[column] = pd.to_numeric(ram_summary[column], errors="coerce")
    ram_summary["approach"] = ram_summary["approach"].fillna("").astype(str).str.strip()
    return results, summary, ram_summary


def build_time_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for approach in _approaches_in_order(summary):
        row = summary[summary["approach"] == approach].iloc[0]
        for metric, label in TIME_METRICS:
            rows.append(
                {
                    "approach": approach,
                    "metric": metric,
                    "label": label,
                    "seconds": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)


def build_ram_summary(ram_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for approach in _approaches_in_order(ram_summary):
        row = ram_summary[ram_summary["approach"] == approach].iloc[0]
        for metric, label in RAM_METRICS:
            rows.append(
                {
                    "approach": approach,
                    "metric": metric,
                    "label": label,
                    "mb": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)


def build_time_by_word(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["approach", "word"], as_index=False)
        .agg(
            measured_runs=("run_id", "count"),
            unique_files_count=("path", "nunique"),
            wall_median_seconds=("wall_seconds", "median"),
            wall_p95_seconds=("wall_seconds", lambda values: float(values.quantile(0.95))),
            cpu_median_seconds=("cpu_seconds", "median"),
            cpu_p95_seconds=("cpu_seconds", lambda values: float(values.quantile(0.95))),
        )
        .sort_values(["approach", "word"])
    )
    return grouped


def build_time_distribution(results: pd.DataFrame) -> pd.DataFrame:
    return results[
        [
            "run_id",
            "repeat_index",
            "item_id",
            "word",
            "class",
            "approach",
            "wall_seconds",
            "cpu_seconds",
        ]
    ].copy()


def _save_csv(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def _bar_label(ax, bars, fmt: str) -> None:
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=8)


def plot_time_summary(data: pd.DataFrame, output_path: Path) -> None:
    approaches = _approaches_in_order(data)
    labels = [label for _, label in TIME_METRICS]
    x = np.arange(len(labels))
    width = 0.78 / max(len(approaches), 1)

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    for index, approach in enumerate(approaches):
        values = [
            float(
                data[(data["approach"] == approach) & (data["label"] == label)]["seconds"].iloc[0]
            )
            for label in labels
        ]
        offset = (index - (len(approaches) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=approach,
            color=APPROACH_COLORS.get(approach),
        )
        _bar_label(ax, bars, "%.3f")

    ax.set_title("Processing time by approach")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.legend(title="Approach")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ram_summary(data: pd.DataFrame, output_path: Path) -> None:
    approaches = _approaches_in_order(data)
    labels = [label for _, label in RAM_METRICS]
    x = np.arange(len(labels))
    width = 0.78 / max(len(approaches), 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for index, approach in enumerate(approaches):
        values = [
            float(
                data[
                    (data["approach"] == approach) & (data["label"] == label)
                ]["mb"].iloc[0]
            )
            for label in labels
        ]
        offset = (index - (len(approaches) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=approach,
            color=APPROACH_COLORS.get(approach),
        )
        _bar_label(ax, bars, "%.1f")

    ax.set_title("RAM usage: cold start vs warm start")
    ax.set_ylabel("MB")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.legend(title="Approach")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_time_by_word(data: pd.DataFrame, output_path: Path) -> None:
    approaches = _approaches_in_order(data)
    words = sorted(data["word"].dropna().unique())
    x = np.arange(len(words))
    width = 0.78 / max(len(approaches), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=False, constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "wall_median_seconds", "Wall median by word"),
        (axes[1], "cpu_median_seconds", "CPU median by word"),
    ):
        for index, approach in enumerate(approaches):
            values = []
            for word in words:
                subset = data[(data["approach"] == approach) & (data["word"] == word)]
                values.append(float(subset[metric].iloc[0]) if not subset.empty else 0.0)
            offset = (index - (len(approaches) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=approach,
                color=APPROACH_COLORS.get(approach),
            )
            _bar_label(ax, bars, "%.3f")

        ax.set_title(title)
        ax.set_ylabel("Seconds")
        ax.set_xticks(x)
        ax.set_xticklabels(words)
        ax.grid(axis="y", linestyle=":", alpha=0.55)
    axes[1].legend(title="Approach")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_time_distribution(data: pd.DataFrame, output_path: Path) -> None:
    approaches = _approaches_in_order(data)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "wall_seconds", "Wall time distribution"),
        (axes[1], "cpu_seconds", "CPU time distribution"),
    ):
        values = [
            data[data["approach"] == approach][metric].dropna().to_numpy()
            for approach in approaches
        ]
        box = ax.boxplot(values, tick_labels=approaches, patch_artist=True, showfliers=False)
        for patch, approach in zip(box["boxes"], approaches, strict=False):
            patch.set_facecolor(APPROACH_COLORS.get(approach, "#777777"))
            patch.set_alpha(0.85)
        ax.set_title(title)
        ax.set_ylabel("Seconds")
        ax.grid(axis="y", linestyle=":", alpha=0.55)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _approach_value(summary: pd.DataFrame, approach: str, column: str) -> float:
    return float(summary[summary["approach"] == approach][column].iloc[0])


def _ratio(left: float, right: float) -> float:
    if right == 0:
        return float("inf")
    return left / right


def write_notes(
    summary: pd.DataFrame,
    ram_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    classic_cold_ram = _approach_value(ram_summary, "classic", "cold_start_ram_mb")
    neural_cold_ram = _approach_value(ram_summary, "neural", "cold_start_ram_mb")
    classic_warm_ram = _approach_value(ram_summary, "classic", "warm_start_ram_mb")
    neural_warm_ram = _approach_value(ram_summary, "neural", "warm_start_ram_mb")
    classic_wall = _approach_value(summary, "classic", "wall_median_seconds")
    neural_wall = _approach_value(summary, "neural", "wall_median_seconds")
    classic_cpu = _approach_value(summary, "classic", "cpu_median_seconds")
    neural_cpu = _approach_value(summary, "neural", "cpu_median_seconds")

    content = f"""# Resource Graph Notes

## resource_time_summary.png
Сравнивает медиану и 95-й перцентиль wall time и CPU time по двум подходам. Neural быстрее по медианному wall time: {neural_wall:.3f} с против {classic_wall:.3f} с у classic. CPU time близок: {neural_cpu:.3f} с у neural и {classic_cpu:.3f} с у classic.

## resource_ram_summary.png
Сравнивает RAM для холодного и теплого запуска. На холодном запуске neural требует {neural_cold_ram:.1f} MB против {classic_cold_ram:.1f} MB у classic. В теплом режиме neural занимает {neural_warm_ram:.1f} MB против {classic_warm_ram:.1f} MB у classic, то есть примерно в {_ratio(neural_warm_ram, classic_warm_ram):.2f} раза больше.

## resource_time_by_word.png
Показывает медианное время обработки отдельно по словам. Этот график нужен, чтобы увидеть, зависит ли преимущество подхода от конкретного слова, а не только от общей агрегированной медианы.

## resource_time_distribution.png
Показывает распределение wall time и CPU time по всем измеренным запускам без выбросов на boxplot. График дополняет median/p95: он показывает стабильность времени и разброс внутри каждого подхода.
"""
    output_path.write_text(content, encoding="utf-8")


def save_outputs(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    ram_source_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    time_summary = build_time_summary(summary)
    ram_summary = build_ram_summary(ram_source_summary)
    time_by_word = build_time_by_word(results)
    time_distribution = build_time_distribution(results)

    _save_csv(time_summary, output_dir / "resource_time_summary_plot.csv")
    _save_csv(ram_summary, output_dir / "resource_ram_summary_plot.csv")
    _save_csv(time_by_word, output_dir / "resource_time_by_word_plot.csv")
    _save_csv(time_distribution, output_dir / "resource_time_distribution_plot.csv")

    plot_time_summary(time_summary, output_dir / "resource_time_summary.png")
    plot_ram_summary(ram_summary, output_dir / "resource_ram_summary.png")
    plot_time_by_word(time_by_word, output_dir / "resource_time_by_word.png")
    plot_time_distribution(time_distribution, output_dir / "resource_time_distribution.png")
    write_notes(summary, ram_source_summary, output_dir / "resource_graph_notes.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build resource usage graphs from evaluation/resources CSV files."
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=DEFAULT_RESOURCES_DIR,
        help="Directory with resource_results.csv and resource_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESOURCES_DIR,
        help="Directory where resource plots and plot CSV files are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resources_dir = args.resources_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not resources_dir.is_absolute():
        resources_dir = PROJECT_ROOT / resources_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    results, summary, ram_summary = load_resource_data(resources_dir)
    save_outputs(results, summary, ram_summary, output_dir)
    print(f"Saved resource plots, plot CSV files, and notes to {output_dir}")


if __name__ == "__main__":
    main()
