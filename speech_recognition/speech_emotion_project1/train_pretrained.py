import json
import os
import inspect
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


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


class RavdessAudioDataset(Dataset):
    def __init__(
        self,
        items: List[Dict],
        label2id: Dict[str, int],
        feature_extractor: AutoFeatureExtractor,
        target_sr: int = 16000,
        use_attention_mask: bool = True,
    ):
        self.items = items
        self.label2id = label2id
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.use_attention_mask = use_attention_mask

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.items[idx]
        audio, sample_rate = sf.read(sample["path"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        waveform = torch.from_numpy(audio).unsqueeze(0)
        if sample_rate != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sr)
        audio = waveform.squeeze(0).numpy()
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        enc = self.feature_extractor(
            audio,
            sampling_rate=self.target_sr,
            return_attention_mask=self.use_attention_mask,
            truncation=True,
            max_length=self.target_sr * 6,
        )
        item = {
            "input_values": torch.tensor(enc["input_values"][0], dtype=torch.float32),
            "labels": torch.tensor(self.label2id[sample["label"]], dtype=torch.long),
            "sample_path": sample["path"],
        }
        if self.use_attention_mask and "attention_mask" in enc:
            item["attention_mask"] = torch.tensor(enc["attention_mask"][0], dtype=torch.long)
        return item


@dataclass
class DataCollatorAudioWithPadding:
    feature_extractor: AutoFeatureExtractor
    use_attention_mask: bool = False
    include_sample_paths: bool = False

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        if self.use_attention_mask and "attention_mask" in features[0]:
            for input_feature, feature in zip(input_features, features):
                input_feature["attention_mask"] = feature["attention_mask"]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        if self.include_sample_paths:
            batch["sample_paths"] = [f.get("sample_path") for f in features]
        if not self.use_attention_mask:
            batch.pop("attention_mask", None)
        return batch


class DebugAudioTrainer(Trainer):
    def __init__(self, *args, class_weights=None, balanced_sampling=False, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.balanced_sampling = balanced_sampling
        self.label_smoothing = float(label_smoothing)

    def get_train_dataloader(self):
        if not self.balanced_sampling:
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if not hasattr(self.train_dataset, "items") or not hasattr(self.train_dataset, "label2id"):
            return super().get_train_dataloader()

        label_ids = [self.train_dataset.label2id[item["label"]] for item in self.train_dataset.items]
        class_counts = Counter(label_ids)
        sample_weights = [1.0 / class_counts[label_id] for label_id in label_ids]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            drop_last=self.args.dataloader_drop_last,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_paths = inputs.pop("sample_paths", None)
        labels = inputs.get("labels")
        try:
            outputs = model(**inputs)
            logits = outputs.get("logits")
            weight = None
            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
            loss = F.cross_entropy(
                logits,
                labels,
                weight=weight,
                label_smoothing=self.label_smoothing,
            )
            return (loss, outputs) if return_outputs else loss
        except Exception:
            print("\n[debug] training batch failed")
            if sample_paths:
                print("[debug] sample_paths:")
                for path in sample_paths:
                    print(f"  - {path}")
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    print(f"[debug] {key}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
            raise


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    weighted_f1 = f1_score(labels, preds, average="weighted")
    macro_f1 = f1_score(labels, preds, average="macro")
    bal_acc = balanced_accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
    }


def build_class_weights(train_items: List[Dict], label2id: Dict[str, int]) -> torch.Tensor:
    counts = np.zeros(len(label2id), dtype=np.float64)
    for sample in train_items:
        counts[label2id[sample["label"]]] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def main():
    root = Path(__file__).resolve().parent
    data_dir = Path(os.environ.get("RAVDESS_DIR", str(root / "data" / "RAVDESS")))
    out_dir = Path(os.environ.get("OUTPUT_DIR", str(root / "pretrained_outputs")))

    model_name = os.environ.get("MODEL_NAME", "microsoft/wavlm-base-plus")
    split_mode = os.environ.get("SPLIT_MODE", "random").strip().lower()
    require_cuda = os.environ.get("REQUIRE_CUDA", "1") == "1"

    seed = int(os.environ.get("SEED", "42"))
    num_epochs = int(os.environ.get("EPOCHS", "30"))
    train_bs = int(os.environ.get("BATCH_SIZE", "8"))
    eval_bs = int(os.environ.get("EVAL_BATCH_SIZE", "8"))
    grad_acc = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
    lr = float(os.environ.get("LR", "2e-5"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.1"))
    dataloader_workers = int(os.environ.get("NUM_WORKERS", "4"))
    test_size = float(os.environ.get("TEST_SIZE", "0.2"))
    val_size = float(os.environ.get("VAL_SIZE", "0.15"))
    early_stop_patience = int(os.environ.get("EARLY_STOP_PATIENCE", "4"))
    debug_audio_batch = os.environ.get("DEBUG_AUDIO_BATCH", "0") == "1"
    use_class_weights = os.environ.get("USE_CLASS_WEIGHTS", "1") == "1"
    balanced_sampling = os.environ.get("BALANCED_SAMPLING", "1") == "1"
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", "0.05"))
    best_metric = os.environ.get("BEST_METRIC", "eval_weighted_f1")

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Install a CUDA-enabled PyTorch build.")

    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Device: CPU")

    items = scan_ravdess(data_dir)
    labels = sorted({x["label"] for x in items})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_items, val_items, test_items = split_items(
        items=items,
        split_mode=split_mode,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )
    print(f"Samples total={len(items)} train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    use_attention_mask = os.environ.get("USE_ATTENTION_MASK", "1") == "1"
    train_ds = RavdessAudioDataset(train_items, label2id, feature_extractor, use_attention_mask=use_attention_mask)
    val_ds = RavdessAudioDataset(val_items, label2id, feature_extractor, use_attention_mask=use_attention_mask)
    test_ds = RavdessAudioDataset(test_items, label2id, feature_extractor, use_attention_mask=use_attention_mask)
    collator = DataCollatorAudioWithPadding(
        feature_extractor,
        use_attention_mask=use_attention_mask,
        include_sample_paths=debug_audio_batch,
    )
    class_weights = build_class_weights(train_items, label2id) if use_class_weights else None

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    out_dir.mkdir(parents=True, exist_ok=True)

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "evaluation_strategy" if "evaluation_strategy" in ta_params else "eval_strategy"

    ta_kwargs = {
        "output_dir": str(out_dir),
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "per_device_train_batch_size": train_bs,
        "per_device_eval_batch_size": eval_bs,
        "gradient_accumulation_steps": grad_acc,
        "dataloader_num_workers": dataloader_workers,
        "num_train_epochs": num_epochs,
        strategy_key: "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": best_metric,
        "greater_is_better": True,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "report_to": "none",
    }
    args = TrainingArguments(**ta_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=early_stop_patience)],
        "class_weights": class_weights,
        "balanced_sampling": balanced_sampling,
        "label_smoothing": label_smoothing,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = feature_extractor
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = feature_extractor

    trainer = DebugAudioTrainer(**trainer_kwargs)

    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    trainer.save_model(str(out_dir))
    feature_extractor.save_pretrained(str(out_dir))
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "split_mode": split_mode,
                "seed": seed,
                "epochs": num_epochs,
                "batch_size": train_bs,
                "eval_batch_size": eval_bs,
                "grad_accum_steps": grad_acc,
                "lr": lr,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "num_workers": dataloader_workers,
                "test_size": test_size,
                "val_size": val_size,
                "early_stop_patience": early_stop_patience,
                "use_class_weights": use_class_weights,
                "balanced_sampling": balanced_sampling,
                "label_smoothing": label_smoothing,
                "best_metric": best_metric,
                "use_attention_mask": use_attention_mask,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            f,
            indent=2,
        )

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print(f"Saved model and metrics to: {out_dir}")


if __name__ == "__main__":
    main()
