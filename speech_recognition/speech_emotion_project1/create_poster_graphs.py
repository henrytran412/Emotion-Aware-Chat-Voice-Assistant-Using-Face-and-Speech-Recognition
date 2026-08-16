from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        raise ValueError(f"Confusion matrix file is empty or invalid: {path}")

    labels = [cell.strip() for cell in rows[0][1:]]
    matrix_rows: list[list[int]] = []

    for row in rows[1:]:
        matrix_rows.append([int(float(cell.strip())) for cell in row[1:]])

    cm = np.array(matrix_rows, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Parsed confusion matrix has invalid shape: {cm.shape}")

    if len(labels) != cm.shape[0]:
        raise ValueError(
            f"Label count ({len(labels)}) does not match matrix size ({cm.shape[0]})"
        )
    return labels, cm


def read_per_class_f1(path: Path) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    scores: list[float] = []
    summary_rows = {"accuracy", "macro avg", "weighted avg"}

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            if not label or label in summary_rows:
                continue

            f1_value = float((row.get("f1-score") or "0").strip())
            labels.append(label)
            scores.append(f1_value * 100.0)

    if not labels:
        raise ValueError(f"No per-class rows found in {path}")

    return labels, np.array(scores, dtype=np.float64)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=240)
    heatmap = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0.0, vmax=1.0)

    ax.set_title("Speech Emotion Confusion Matrix (Normalized)", fontsize=17, pad=16)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_norm[i, j] * 100.0
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
    metric_names = ["Accuracy", "Macro F1", "Weighted F1"]
    values = [
        float(summary["accuracy"]) * 100.0,
        float(summary["f1_macro"]) * 100.0,
        float(summary["f1_weighted"]) * 100.0,
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=240)
    bars = ax.bar(metric_names, values, color=colors, width=0.6)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Best Speech Test Metrics", fontsize=17, pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(labels: list[str], scores: np.ndarray, out_path: Path) -> None:
    sort_idx = np.argsort(scores)
    labels_sorted = [labels[i] for i in sort_idx]
    scores_sorted = scores[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=240)
    bars = ax.barh(labels_sorted, scores_sorted, color="#17a2b8", alpha=0.9)

    ax.set_xlim(0, 100)
    ax.set_xlabel("F1 Score (%)", fontsize=12)
    ax.set_title("Per-Class F1 Scores", fontsize=17, pad=14)
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
        description="Create poster-ready confusion matrix and metric charts for speech emotion results."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "pretrained_outputs_safe",
        help="Model output directory containing eval_report and metrics files.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save generated poster charts (default: <model-dir>/eval_report/poster_graphs)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    eval_report_dir = model_dir / "eval_report"
    save_dir = args.save_dir if args.save_dir is not None else (eval_report_dir / "poster_graphs")
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_path = eval_report_dir / "evaluation_summary.json"
    class_report_path = eval_report_dir / "classification_report.csv"
    confusion_path = eval_report_dir / "confusion_matrix.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not class_report_path.exists():
        raise FileNotFoundError(f"Missing classification report: {class_report_path}")
    if not confusion_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix file: {confusion_path}")

    summary = read_json(summary_path)
    labels, cm = read_confusion_matrix(confusion_path)
    f1_labels, f1_scores = read_per_class_f1(class_report_path)

    confusion_out = save_dir / "poster_confusion_matrix.png"
    overall_out = save_dir / "poster_overall_metrics.png"
    per_class_out = save_dir / "poster_per_class_f1.png"

    plot_confusion_matrix(cm, labels, confusion_out)
    plot_overall_metrics(summary, overall_out)
    plot_per_class_f1(f1_labels, f1_scores, per_class_out)

    print("Created poster charts:")
    print(f" - {confusion_out}")
    print(f" - {overall_out}")
    print(f" - {per_class_out}")


if __name__ == "__main__":
    main()
