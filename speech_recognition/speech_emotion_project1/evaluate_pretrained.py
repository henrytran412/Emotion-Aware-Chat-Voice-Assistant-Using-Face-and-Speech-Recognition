import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


def parse_ravdess_file(file_path: Path) -> Dict:
    parts = file_path.name.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected file name format: {file_path.name}")
    emotion_code = int(parts[2])
    actor_id = int(parts[6].split(".")[0])
    label = EMOTION_MAP[emotion_code]
    return {"path": str(file_path), "actor_id": actor_id, "label": label}


def scan_ravdess(data_dir: Path) -> List[Dict]:
    items: List[Dict] = []
    for actor_dir in sorted(data_dir.glob("Actor_*")):
        if not actor_dir.is_dir():
            continue
        for wav_file in sorted(actor_dir.glob("*.wav")):
            items.append(parse_ravdess_file(wav_file))
    if not items:
        raise FileNotFoundError(f"No RAVDESS wav files found in: {data_dir}")
    return items


def split_items(
    items: List[Dict],
    split_mode: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    labels = np.array([x["label"] for x in items])
    indices = np.arange(len(items))

    if split_mode == "actor":
        groups = np.array([x["actor_id"] for x in items])
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(gss.split(indices, labels, groups=groups))
    elif split_mode == "random":
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    else:
        raise ValueError("SPLIT_MODE must be 'random' or 'actor'")

    train_val_labels = labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    test_items = [items[i] for i in test_idx]
    return train_items, val_items, test_items


def load_audio(audio_path: str, target_sr: int = 16000):
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    audio = waveform.squeeze(0).numpy()
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio, target_sr


def main():
    root = Path(__file__).resolve().parent
    model_dir = Path(os.environ.get("MODEL_DIR", str(root / "pretrained_outputs")))
    data_dir = Path(os.environ.get("RAVDESS_DIR", str(root / "data" / "RAVDESS")))
    output_dir = Path(os.environ.get("EVAL_OUT_DIR", str(model_dir / "eval_report")))

    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    config_path = model_dir / "train_config.json"
    cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    split_mode = os.environ.get("SPLIT_MODE", cfg.get("split_mode", "random")).strip().lower()
    seed = int(os.environ.get("SEED", str(cfg.get("seed", 42))))
    test_size = float(os.environ.get("TEST_SIZE", str(cfg.get("test_size", 0.2))))
    val_size = float(os.environ.get("VAL_SIZE", str(cfg.get("val_size", 0.15))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_dir))
    model = AutoModelForAudioClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()} if isinstance(next(iter(model.config.id2label.keys())), str) else model.config.id2label
    labels_order = [id2label[i] for i in sorted(id2label.keys())]
    label2id = {v: k for k, v in id2label.items()}

    items = scan_ravdess(data_dir)
    _, _, test_items = split_items(
        items=items,
        split_mode=split_mode,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    rows = []

    for sample in test_items:
        audio, sr = load_audio(sample["path"], target_sr=16000)
        inputs = feature_extractor(
            audio,
            sampling_rate=sr,
            return_attention_mask=False,
            return_tensors="pt",
            truncation=True,
            max_length=sr * 6,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_id = int(np.argmax(probs))
        true_id = int(label2id[sample["label"]])
        conf = float(probs[pred_id])
        y_true.append(true_id)
        y_pred.append(pred_id)
        rows.append(
            {
                "file": Path(sample["path"]).name,
                "actor_id": sample["actor_id"],
                "true_label": sample["label"],
                "pred_label": id2label[pred_id],
                "confidence": conf,
                "correct": int(pred_id == true_id),
            }
        )

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=sorted(id2label.keys()))
    report = classification_report(
        y_true,
        y_pred,
        labels=sorted(id2label.keys()),
        target_names=labels_order,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "actor_id", "true_label", "pred_label", "confidence", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label in labels_order:
            writer.writerow(
                [
                    label,
                    report[label]["precision"],
                    report[label]["recall"],
                    report[label]["f1-score"],
                    report[label]["support"],
                ]
            )
        writer.writerow(
            [
                "macro avg",
                report["macro avg"]["precision"],
                report["macro avg"]["recall"],
                report["macro avg"]["f1-score"],
                report["macro avg"]["support"],
            ]
        )
        writer.writerow(
            [
                "weighted avg",
                report["weighted avg"]["precision"],
                report["weighted avg"]["recall"],
                report["weighted avg"]["f1-score"],
                report["weighted avg"]["support"],
            ]
        )

    with open(output_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + labels_order)
        for i, label in enumerate(labels_order):
            writer.writerow([label] + cm[i].tolist())

    summary = {
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "split_mode": split_mode,
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "num_test_samples": len(test_items),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    poster_text = [
        "Speech Emotion Recognition Evaluation Report",
        "===========================================",
        f"Model directory: {model_dir}",
        f"Data directory: {data_dir}",
        f"Split mode: {split_mode}",
        f"Seed: {seed}",
        f"Test samples: {len(test_items)}",
        "",
        f"Accuracy:           {acc:.4f} ({acc * 100:.2f}%)",
        f"Balanced Accuracy:  {bal_acc:.4f} ({bal_acc * 100:.2f}%)",
        f"Macro F1:           {f1_macro:.4f}",
        f"Weighted F1:        {f1_weighted:.4f}",
        "",
        "Per-class metrics are in classification_report.csv",
        "Confusion matrix is in confusion_matrix.csv",
        "Sample-level predictions are in predictions.csv",
    ]
    with open(output_dir / "poster_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(poster_text) + "\n")

    print("\n".join(poster_text))
    print(f"\nSaved full evaluation artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
