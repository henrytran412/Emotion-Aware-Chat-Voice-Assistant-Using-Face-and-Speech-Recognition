import json
import os
import random
import tempfile
import time
from multiprocessing import freeze_support

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SEVEN_CLASSES
from dataset_cached import CachedNpyDataset


ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR = os.path.join(ROOT, "val")
CSV_PATH = os.path.join(ROOT, "labels.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
CACHE_DIR = os.path.join(ROOT, "cache_224_uint8")

os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 7
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Defaults tuned for desktop GPU (RTX 4070). Override with env vars.
MODEL_NAME = os.environ.get("MODEL_NAME", "mobilenetv3_large_100")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
LR = float(os.environ.get("LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.05"))
GRAD_CLIP_NORM = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "6"))
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS", "8"))
SEED = int(os.environ.get("SEED", "42"))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_save(obj, final_path: str, retries: int = 8, delay_s: float = 0.5):
    out_dir = os.path.dirname(final_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    last_err = None

    for attempt in range(1, retries + 1):
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".pt", dir=out_dir)
            os.close(fd)
            torch.save(obj, tmp_path)
            os.replace(tmp_path, final_path)
            return
        except OSError as e:
            last_err = e
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < retries and getattr(e, "winerror", None) == 1224:
                time.sleep(delay_s * attempt)
                continue
            raise
        except RuntimeError as e:
            last_err = e
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < retries and "error code: 1224" in str(e):
                time.sleep(delay_s * attempt)
                continue
            raise

    raise RuntimeError(f"Failed saving checkpoint to {final_path}: {last_err}")


def main():
    set_seed(SEED)

    require_cuda = os.environ.get("REQUIRE_CUDA", "1") == "1"
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is required but not available. Install a CUDA-enabled PyTorch build "
            "(not +cpu) for your NVIDIA GPU, then rerun training. "
            f"Detected torch={torch.__version__}, torch.cuda={torch.version.cuda}."
        )

    device = torch.device("cuda" if cuda_available else "cpu")
    amp = device.type == "cuda"

    print("ROOT:", ROOT)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    for p in [TRAIN_DIR, VAL_DIR, CSV_PATH, os.path.join(CACHE_DIR, "train"), os.path.join(CACHE_DIR, "val")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required path: {p}")

    print("Loading cached datasets...")
    train_ds = CachedNpyDataset(os.path.join(CACHE_DIR, "train"), mean=MEAN, std=STD, augment=True)
    val_ds = CachedNpyDataset(os.path.join(CACHE_DIR, "val"), mean=MEAN, std=STD, augment=False)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    class_counts = np.bincount(np.array(train_ds.labels, dtype=np.int64), minlength=NUM_CLASSES)
    class_counts = np.maximum(class_counts, 1)
    class_weights = class_counts.sum() / (NUM_CLASSES * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print("Class counts:", class_counts.tolist())
    print("Class weights:", [round(float(w), 4) for w in class_weights.cpu().tolist()])

    pin = device.type == "cuda"
    dl_kwargs = {}
    if NUM_WORKERS > 0:
        dl_kwargs["prefetch_factor"] = 2
        dl_kwargs["persistent_workers"] = True
    else:
        dl_kwargs["persistent_workers"] = False

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        **dl_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        **dl_kwargs,
    )

    print("Building model:", MODEL_NAME)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def _to_device(x: torch.Tensor) -> torch.Tensor:
        if device.type == "cuda":
            return x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        return x.to(device)

    def evaluate():
        model.eval()
        correct, total = 0, 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x = _to_device(x)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return loss_sum / max(total, 1), correct / max(total, 1)

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = []

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}")
        for x, y in pbar:
            x = _to_device(x)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                print("Non-finite loss detected, skipping step")
                continue

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
            pbar.set_postfix(train_loss=running_loss / max(seen, 1), lr=optimizer.param_groups[0]["lr"])

        val_loss, val_acc = evaluate()
        scheduler.step()
        epoch_train_loss = running_loss / max(seen, 1)

        record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch}: train_loss={epoch_train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            no_improve = 0

            safe_torch_save(model.state_dict(), os.path.join(OUT_DIR, "best.pth"))
            safe_torch_save(
                {
                    "model_name": MODEL_NAME,
                    "num_classes": NUM_CLASSES,
                    "state_dict": model.state_dict(),
                    "classes": SEVEN_CLASSES,
                    "img_size": IMG_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "color_order": "RGB",
                },
                os.path.join(OUT_DIR, "best.pt"),
            )
            print(f"Saved best.pth and best.pt (val_acc={best_acc:.4f})")
        else:
            no_improve += 1

        if epoch >= MIN_EPOCHS and no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    with open(os.path.join(OUT_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(OUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        for c in SEVEN_CLASSES:
            f.write(c + "\n")

    with open(os.path.join(OUT_DIR, "preprocess.json"), "w", encoding="utf-8") as f:
        json.dump({"img_size": IMG_SIZE, "mean": MEAN, "std": STD, "color_order": "RGB"}, f, indent=2)

    with open(os.path.join(OUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "num_workers": NUM_WORKERS,
                "label_smoothing": LABEL_SMOOTHING,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "min_epochs": MIN_EPOCHS,
                "seed": SEED,
            },
            f,
            indent=2,
        )

    print("Training finished.")
    print("Best val acc:", best_acc, "at epoch", best_epoch)
    print("Outputs in:", OUT_DIR)


if __name__ == "__main__":
    freeze_support()
    main()
