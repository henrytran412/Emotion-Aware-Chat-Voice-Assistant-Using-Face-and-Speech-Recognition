import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AffectNetCSVDataset, SEVEN_CLASSES

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "test")
CSV_PATH = os.path.join(ROOT, "labels.csv")
OUT_DIR = os.path.join(ROOT, "outputs")

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0
KNN_K = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

tfm = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def _extract_embeddings(model: torch.nn.Module, dataloader: DataLoader):
    emb_list = []
    labels = []
    preds = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            logits = model(x)
            preds.extend(logits.argmax(dim=1).cpu().tolist())

            feats = model.forward_features(x)
            emb = model.forward_head(feats, pre_logits=True)
            if emb.ndim > 2:
                emb = torch.flatten(emb, 1)
            emb_list.append(emb.cpu().numpy())
            labels.extend(y.tolist())

    embeddings = np.concatenate(emb_list, axis=0) if emb_list else np.empty((0, 0), dtype=np.float32)
    return embeddings, np.array(labels, dtype=np.int64), np.array(preds, dtype=np.int64)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, class_names):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    return cm, report


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    acc = float(accuracy_score(y_true, y_pred))

    per_class = []
    for idx, cls_name in enumerate(class_names):
        per_class.append(
            {
                "class": cls_name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
    }


def _save_confusion_matrix_figure(cm: np.ndarray, class_names, out_png: str, title: str, normalize: bool):
    plot_cm = cm.astype(np.float64)
    if normalize:
        row_sum = plot_cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(plot_cm, np.maximum(row_sum, 1.0))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(plot_cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = plot_cm.max() / 2.0 if plot_cm.size else 0.0
    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            if normalize:
                txt = f"{plot_cm[i, j]:.2f}"
            else:
                txt = str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center", color="white" if plot_cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_per_class_bar_chart(per_class_metrics, out_png: str, title: str):
    classes = [m["class"] for m in per_class_metrics]
    precision = np.array([m["precision"] for m in per_class_metrics])
    recall = np.array([m["recall"] for m in per_class_metrics])
    f1 = np.array([m["f1"] for m in per_class_metrics])

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, precision, width=w, label="Precision")
    ax.bar(x, recall, width=w, label="Recall")
    ax.bar(x + w, f1, width=w, label="F1-score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_per_class_csv(per_class_metrics, out_csv: str):
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for m in per_class_metrics:
            f.write(f"{m['class']},{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['support']}\n")


def _save_poster_artifacts(name: str, y_true: np.ndarray, y_pred: np.ndarray, class_names, out_dir: str, text_lines):
    cm, report = _evaluate(y_true, y_pred, class_names)
    metrics = _compute_metrics(y_true, y_pred, class_names)

    print(f"\n=== {name} ===")
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    text_lines.append(f"\n=== {name} ===")
    text_lines.append("Confusion matrix:\n" + str(cm))
    text_lines.append(report)
    text_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")

    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    _save_confusion_matrix_figure(
        cm=cm,
        class_names=class_names,
        out_png=os.path.join(out_dir, f"confusion_{safe_name}.png"),
        title=f"{name} Confusion Matrix (Count)",
        normalize=False,
    )
    _save_confusion_matrix_figure(
        cm=cm,
        class_names=class_names,
        out_png=os.path.join(out_dir, f"confusion_{safe_name}_norm.png"),
        title=f"{name} Confusion Matrix (Row-normalized)",
        normalize=True,
    )
    _save_per_class_bar_chart(
        per_class_metrics=metrics["per_class"],
        out_png=os.path.join(out_dir, f"per_class_{safe_name}.png"),
        title=f"{name} Per-class Precision / Recall / F1",
    )
    _write_per_class_csv(metrics["per_class"], os.path.join(out_dir, f"per_class_{safe_name}.csv"))
    with open(os.path.join(out_dir, f"summary_{safe_name}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["softmax", "knn", "both"], default="softmax")
    parser.add_argument("--knn-k", type=int, default=KNN_K)
    args = parser.parse_args()

    test_ds = AffectNetCSVDataset(TEST_DIR, CSV_PATH, transform=tfm)
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    ckpt = torch.load(os.path.join(OUT_DIR, "best.pt"), map_location=DEVICE)
    class_names = ckpt.get("classes", SEVEN_CLASSES)

    model = timm.create_model(
        ckpt["model_name"],
        pretrained=False,
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()

    test_emb, y_test, y_softmax = _extract_embeddings(model, test_dl)

    lines = []
    lines.append(f"Test samples: {len(test_ds)}")
    all_summary = {
        "device": str(DEVICE),
        "test_samples": int(len(test_ds)),
        "model_name": ckpt["model_name"],
        "num_classes": int(ckpt["num_classes"]),
    }

    if args.mode in ("softmax", "both"):
        metrics = _save_poster_artifacts(
            name="Softmax model head",
            y_true=y_test,
            y_pred=y_softmax,
            class_names=class_names,
            out_dir=OUT_DIR,
            text_lines=lines,
        )
        all_summary["softmax_model_head"] = metrics

    if args.mode in ("knn", "both"):
        train_ds = AffectNetCSVDataset(TRAIN_DIR, CSV_PATH, transform=tfm)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
        )
        train_emb, y_train, _ = _extract_embeddings(model, train_dl)

        if len(train_emb) == 0:
            raise RuntimeError("No train embeddings extracted for KNN.")

        knn = KNeighborsClassifier(n_neighbors=args.knn_k, weights="distance", metric="cosine")
        knn.fit(train_emb, y_train)
        y_knn = knn.predict(test_emb)

        metrics = _save_poster_artifacts(
            name=f"KNN embeddings k={args.knn_k}",
            y_true=y_test,
            y_pred=y_knn,
            class_names=class_names,
            out_dir=OUT_DIR,
            text_lines=lines,
        )
        all_summary[f"knn_embeddings_k_{args.knn_k}"] = metrics

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)

    print("\nSaved outputs/test_report.txt and poster figures/metrics in outputs/")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
