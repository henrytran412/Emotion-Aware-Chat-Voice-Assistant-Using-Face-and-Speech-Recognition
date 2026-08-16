from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_confusion_matrix(report_path: Path) -> np.ndarray:
    text = report_path.read_text(encoding="utf-8")
    rows = []
    in_matrix_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not in_matrix_block:
            if line.startswith("Confusion matrix:"):
                in_matrix_block = True
            continue

        if not line:
            if rows:
                break
            continue

        if not line.startswith("["):
            if rows:
                break
            continue

        values = [int(x) for x in re.findall(r"-?\d+", line)]
        if values:
            rows.append(values)

    if not rows:
        raise ValueError(f"Could not find confusion matrix in {report_path}")

    cm = np.array(rows, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Parsed confusion matrix has invalid shape: {cm.shape}")
    return cm


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=240)
    heatmap = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0.0, vmax=1.0)

    ax.set_title("Confusion Matrix (Normalized)", fontsize=18, pad=16)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_norm[i, j] * 100
            text_color = "white" if cm_norm[i, j] > 0.45 else "black"
            ax.text(
                j,
                i,
                f"{pct:.1f}%\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized value", rotation=270, labelpad=18)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_overall_metrics(summary: dict, out_path: Path) -> None:
    metric_names = ["Accuracy", "Macro F1"]
    values = [summary["accuracy"] * 100.0, summary["macro_f1"] * 100.0]
    colors = ["#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=240)
    bars = ax.bar(metric_names, values, color=colors, width=0.6)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Best Test Metrics", fontsize=18, pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(summary: dict, out_path: Path) -> None:
    per_class = summary["per_class"]
    labels = [item["class"].title() for item in per_class]
    f1_scores = np.array([item["f1"] * 100.0 for item in per_class], dtype=np.float64)

    sort_idx = np.argsort(f1_scores)
    labels_sorted = [labels[i] for i in sort_idx]
    scores_sorted = f1_scores[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=240)
    bars = ax.barh(labels_sorted, scores_sorted, color="#2ca02c", alpha=0.9)

    ax.set_xlim(0, 100)
    ax.set_xlabel("F1 Score (%)", fontsize=12)
    ax.set_title("Per-Class F1 Scores", fontsize=18, pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, score in zip(bars, scores_sorted):
        ax.text(
            score + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create poster-ready confusion matrix and metric charts from dataset_CUDA outputs."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory containing eval_summary.json, summary_softmax_model_head.json, and test_report.txt",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save generated poster charts (default: <outputs-dir>/poster_graphs)",
    )
    args = parser.parse_args()

    outputs_dir = args.outputs_dir
    save_dir = args.save_dir if args.save_dir is not None else (outputs_dir / "poster_graphs")
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_path = outputs_dir / "summary_softmax_model_head.json"
    report_path = outputs_dir / "test_report.txt"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report file: {report_path}")

    summary = load_summary(summary_path)
    cm = parse_confusion_matrix(report_path)

    labels = [item["class"] for item in summary["per_class"]]

    confusion_out = save_dir / "poster_confusion_matrix.png"
    overall_out = save_dir / "poster_overall_metrics.png"
    per_class_out = save_dir / "poster_per_class_f1.png"

    plot_confusion_matrix(cm, labels, confusion_out)
    plot_overall_metrics(summary, overall_out)
    plot_per_class_f1(summary, per_class_out)

    print("Created poster charts:")
    print(f" - {confusion_out}")
    print(f" - {overall_out}")
    print(f" - {per_class_out}")


if __name__ == "__main__":
    main()
