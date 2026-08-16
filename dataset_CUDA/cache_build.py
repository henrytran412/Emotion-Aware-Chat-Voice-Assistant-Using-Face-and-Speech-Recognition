import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SEVEN_CLASSES = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL2ID = {name: i for i, name in enumerate(SEVEN_CLASSES)}

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR   = os.path.join(ROOT, "val")
CSV_PATH  = os.path.join(ROOT, "labels.csv")
CACHE_DIR = os.path.join(ROOT, "cache_224_uint8")

IMG_SIZE = 224

def load_df(split_dir):
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(LABEL2ID)].copy()

    df["pth"] = df["pth"].astype(str)
    df["full_path"] = df["pth"].apply(lambda p: os.path.join(split_dir, p))
    df = df[df["full_path"].apply(os.path.isfile)].reset_index(drop=True)
    return df

def build_split(split_name, split_dir):
    df = load_df(split_dir)
    out_dir = os.path.join(CACHE_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)

    meta = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Caching {split_name}"):
        path = row["full_path"]
        y = LABEL2ID[row["label"]]

        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.asarray(img, dtype=np.uint8)  # HWC uint8
        except Exception as e:
            # skip corrupt file
            continue

        # save as .npy (fast to load)
        fname = f"{i:07d}.npy"
        np.save(os.path.join(out_dir, fname), arr)
        meta.append((fname, y))

    # save labels file
    np.save(os.path.join(out_dir, "labels.npy"), np.array(meta, dtype=object))
    print(f"{split_name}: saved {len(meta)} items to {out_dir}")

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    build_split("train", TRAIN_DIR)
    build_split("val", VAL_DIR)
    print("Done. Cache dir:", CACHE_DIR)