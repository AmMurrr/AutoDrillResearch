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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "visual"
CLASS_ORDER = ("fail", "moderate", "perfect")
APPROACH_ORDER = ("classic", "neural")
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def predicted_class(score: float) -> str:
    numeric_score = float(score)
    if numeric_score > 70.0:
        return "perfect"
    if numeric_score >= 30.0:
        return "moderate"
    return "fail"


def load_results(results_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(results_dir.glob("score_results_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No score_results_*.csv files found in {results_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame["source_file"] = csv_path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    required_columns = {"word", "class", "approach", "score", "status"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    data = data.copy()
    data["word"] = data["word"].astype(str).str.strip()
    data["class"] = data["class"].astype(str).str.strip()
    data["approach"] = data["approach"].astype(str).str.strip()
    data["status"] = data["status"].fillna("").astype(str).str.strip()
    data["score"] = pd.to_numeric(data["score"], errors="coerce").fillna(0.0)
    data = data[data["class"].isin(CLASS_ORDER)].copy()
    data["predicted_class"] = data["score"].map(predicted_class)
    return data


def _approaches_in_order(data: pd.DataFrame) -> list[str]:
    approaches = list(dict.fromkeys(APPROACH_ORDER + tuple(sorted(data["approach"].unique()))))
    return [approach for approach in approaches if approach in set(data["approach"])]


def _words_in_order(data: pd.DataFrame) -> list[str]:
    return sorted(data["word"].dropna().unique())


def _quality_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    total = int(len(group))
    if total == 0:
        return {
            "items_count": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "status_not_ok_count": 0,
            "status_not_ok_rate": 0.0,
            "perfect_to_fail_count": 0,
            "perfect_to_fail_rate": 0.0,
            "fail_to_perfect_count": 0,
            "fail_to_perfect_rate": 0.0,
            "score_mean": 0.0,
            "score_median": 0.0,
            "score_std": 0.0,
            "gap_perfect_moderate": 0.0,
            "gap_moderate_fail": 0.0,
        }

    y_true = group["class"]
    y_pred = group["predicted_class"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASS_ORDER),
        average=None,
        zero_division=0,
    )

    status_not_ok = group["status"] != "ok"
    perfect_items = group[group["class"] == "perfect"]
    fail_items = group[group["class"] == "fail"]
    perfect_to_fail_count = int(
        ((group["class"] == "perfect") & (group["predicted_class"] == "fail")).sum()
    )
    fail_to_perfect_count = int(
        ((group["class"] == "fail") & (group["predicted_class"] == "perfect")).sum()
    )
    score_by_class = group.groupby("class")["score"].mean()

    return {
        "items_count": total,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "status_not_ok_count": int(status_not_ok.sum()),
        "status_not_ok_rate": float(status_not_ok.mean()),
        "perfect_to_fail_count": perfect_to_fail_count,
        "perfect_to_fail_rate": _safe_rate(perfect_to_fail_count, len(perfect_items)),
        "fail_to_perfect_count": fail_to_perfect_count,
        "fail_to_perfect_rate": _safe_rate(fail_to_perfect_count, len(fail_items)),
        "score_mean": float(group["score"].mean()),
        "score_median": float(group["score"].median()),
        "score_std": float(group["score"].std(ddof=0)),
        "gap_perfect_moderate": _class_gap(score_by_class, "perfect", "moderate"),
        "gap_moderate_fail": _class_gap(score_by_class, "moderate", "fail"),
    }


def _safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count / total)


def _class_gap(score_by_class: pd.Series, left: str, right: str) -> float:
    if left not in score_by_class or right not in score_by_class:
        return 0.0
    return float(score_by_class[left] - score_by_class[right])


def build_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for approach in _approaches_in_order(data):
        group = data[data["approach"] == approach]
        rows.append({"approach": approach, **_quality_metrics(group)})
    return pd.DataFrame(rows)


def build_quality_by_word(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for approach in _approaches_in_order(data):
        for word in _words_in_order(data):
            group = data[(data["approach"] == approach) & (data["word"] == word)]
            if group.empty:
                continue
            rows.append({"approach": approach, "word": word, **_quality_metrics(group)})
    return pd.DataFrame(rows)


def build_score_by_class(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregations = {
        "score_count": ("score", "count"),
        "score_mean": ("score", "mean"),
        "score_median": ("score", "median"),
        "score_std": ("score", lambda values: float(values.std(ddof=0))),
        "score_min": ("score", "min"),
        "score_max": ("score", "max"),
    }
    by_word = (
        data.groupby(["approach", "word", "class"], as_index=False)
        .agg(**aggregations)
        .sort_values(["approach", "word", "class"])
    )
    overall = (
        data.groupby(["approach", "class"], as_index=False)
        .agg(**aggregations)
        .sort_values(["approach", "class"])
    )
    return by_word, overall


def build_confusion_matrices(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for approach in _approaches_in_order(data):
        group = data[data["approach"] == approach]
        matrix = confusion_matrix(
            group["class"],
            group["predicted_class"],
            labels=list(CLASS_ORDER),
        )
        row_totals = matrix.sum(axis=1)
        for true_index, true_class in enumerate(CLASS_ORDER):
            for predicted_index, predicted in enumerate(CLASS_ORDER):
                count = int(matrix[true_index, predicted_index])
                total = int(row_totals[true_index])
                rows.append(
                    {
                        "approach": approach,
                        "true_class": true_class,
                        "predicted_class": predicted,
                        "count": count,
                        "row_percent": _safe_rate(count, total),
                    }
                )
    return pd.DataFrame(rows)


def save_csv_outputs(
    data: pd.DataFrame,
    quality_summary: pd.DataFrame,
    quality_by_word: pd.DataFrame,
    score_by_class: pd.DataFrame,
    score_by_class_overall: pd.DataFrame,
    confusion_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "quality_rows.csv", index=False)
    quality_summary.to_csv(output_dir / "quality_summary.csv", index=False)
    quality_by_word.to_csv(output_dir / "quality_by_word.csv", index=False)
    score_by_class.to_csv(output_dir / "score_by_class.csv", index=False)
    score_by_class_overall.to_csv(output_dir / "score_by_class_overall.csv", index=False)
    confusion_data.to_csv(output_dir / "confusion_matrix.csv", index=False)


def plot_quality_summary(summary: pd.DataFrame, output_path: Path) -> None:
    approaches = summary["approach"].tolist()
    x = np.arange(len(SUMMARY_METRICS))
    width = 0.78 / max(len(approaches), 1)

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    for index, approach in enumerate(approaches):
        row = summary[summary["approach"] == approach].iloc[0]
        values = [float(row[metric]) * 100.0 for metric in SUMMARY_METRICS]
        offset = (index - (len(approaches) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=approach)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    ax.set_title("Quality metrics by approach")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([metric.replace("_", "\n") for metric in SUMMARY_METRICS])
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.legend(title="Approach")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_quality_by_word(quality_by_word: pd.DataFrame, output_path: Path) -> None:
    words = sorted(quality_by_word["word"].unique())
    approaches = list(dict.fromkeys(quality_by_word["approach"]))
    x = np.arange(len(words))
    width = 0.78 / max(len(approaches), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, constrained_layout=True)
    for ax, metric in zip(axes, ("accuracy", "macro_f1"), strict=True):
        for index, approach in enumerate(approaches):
            values = []
            for word in words:
                subset = quality_by_word[
                    (quality_by_word["approach"] == approach) & (quality_by_word["word"] == word)
                ]
                values.append(float(subset.iloc[0][metric]) * 100.0 if not subset.empty else 0.0)
            offset = (index - (len(approaches) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width=width, label=approach)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(words)
        ax.grid(axis="y", linestyle=":", alpha=0.55)
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("Percent")
    axes[1].legend(title="Approach", loc="upper right")
    fig.suptitle("Quality metrics by word")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_score_by_class(score_by_class_overall: pd.DataFrame, output_path: Path) -> None:
    approaches = list(dict.fromkeys(score_by_class_overall["approach"]))
    x = np.arange(len(CLASS_ORDER))
    width = 0.78 / max(len(approaches), 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for index, approach in enumerate(approaches):
        values = []
        errors = []
        for class_name in CLASS_ORDER:
            subset = score_by_class_overall[
                (score_by_class_overall["approach"] == approach)
                & (score_by_class_overall["class"] == class_name)
            ]
            values.append(float(subset.iloc[0]["score_mean"]) if not subset.empty else 0.0)
            errors.append(float(subset.iloc[0]["score_std"]) if not subset.empty else 0.0)

        offset = (index - (len(approaches) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, yerr=errors, capsize=4, label=approach)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    ax.axhline(30.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.axhline(70.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_title("Average score by expected class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.legend(title="Approach")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(confusion_data: pd.DataFrame, output_path: Path) -> None:
    approaches = list(dict.fromkeys(confusion_data["approach"]))
    fig, axes = plt.subplots(
        1,
        len(approaches),
        figsize=(5.8 * len(approaches), 5.4),
        constrained_layout=True,
        squeeze=False,
    )
    max_count = max(int(confusion_data["count"].max()), 1)

    for ax, approach in zip(axes[0], approaches, strict=True):
        subset = confusion_data[confusion_data["approach"] == approach]
        matrix = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)
        percent = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=float)
        for true_index, true_class in enumerate(CLASS_ORDER):
            for predicted_index, predicted in enumerate(CLASS_ORDER):
                row = subset[
                    (subset["true_class"] == true_class)
                    & (subset["predicted_class"] == predicted)
                ].iloc[0]
                matrix[true_index, predicted_index] = int(row["count"])
                percent[true_index, predicted_index] = float(row["row_percent"])

        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_count)
        for true_index in range(len(CLASS_ORDER)):
            for predicted_index in range(len(CLASS_ORDER)):
                value = matrix[true_index, predicted_index]
                text_color = "white" if value > max_count * 0.55 else "black"
                ax.text(
                    predicted_index,
                    true_index,
                    f"{value}\n{percent[true_index, predicted_index] * 100:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

        ax.set_title(approach)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(np.arange(len(CLASS_ORDER)))
        ax.set_yticks(np.arange(len(CLASS_ORDER)))
        ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_ORDER)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, label="Count")
    fig.suptitle("Confusion matrix by approach")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_plots(
    quality_summary: pd.DataFrame,
    quality_by_word: pd.DataFrame,
    score_by_class_overall: pd.DataFrame,
    confusion_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_quality_summary(quality_summary, output_dir / "quality_summary.png")
    plot_quality_by_word(quality_by_word, output_dir / "quality_by_word.png")
    plot_score_by_class(score_by_class_overall, output_dir / "score_by_class.png")
    plot_confusion_matrix(confusion_data, output_dir / "confusion_matrix.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate quality metrics from evaluation/visual/score_results_*.csv."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory with score_results_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where metric CSV files and plots are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    data = load_results(results_dir)
    quality_summary = build_quality_summary(data)
    quality_by_word = build_quality_by_word(data)
    score_by_class, score_by_class_overall = build_score_by_class(data)
    confusion_data = build_confusion_matrices(data)

    save_csv_outputs(
        data=data,
        quality_summary=quality_summary,
        quality_by_word=quality_by_word,
        score_by_class=score_by_class,
        score_by_class_overall=score_by_class_overall,
        confusion_data=confusion_data,
        output_dir=output_dir,
    )
    save_plots(
        quality_summary=quality_summary,
        quality_by_word=quality_by_word,
        score_by_class_overall=score_by_class_overall,
        confusion_data=confusion_data,
        output_dir=output_dir,
    )

    print(f"Saved quality metrics and plots to {output_dir}")


if __name__ == "__main__":
    main()
