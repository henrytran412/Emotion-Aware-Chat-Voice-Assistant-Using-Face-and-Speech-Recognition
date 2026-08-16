import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

SEVEN_CLASSES = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL2ID = {name: i for i, name in enumerate(SEVEN_CLASSES)}

class AffectNetCSVDataset(Dataset):
    """
    Works with either:
      - one global labels.csv (pth relative to class folder or split folder), or
      - per-split csv.

    Strategy:
      For a given split_dir, build full_path = split_dir / pth
      Then keep only rows where full_path exists.
    """

    def __init__(self, split_dir, csv_path, transform=None, return_relFCs=False):
        self.split_dir = split_dir
        self.transform = transform
        self.return_relFCs = return_relFCs

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        if "pth" not in df.columns or "label" not in df.columns:
            raise ValueError(f"CSV must contain 'pth' and 'label'. Found: {df.columns.tolist()}")

        df["label"] = df["label"].astype(str).str.lower().str.strip()

        # keep only 7-class labels
        df = df[df["label"].isin(LABEL2ID)].copy()

        # full_path for THIS split
        df["full_path"] = df["pth"].astype(str).apply(lambda p: os.path.join(split_dir, p))

        # keep rows that actually exist in this split
        df = df[df["full_path"].apply(os.path.isfile)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No samples found for split_dir={split_dir}. "
                f"Check that CSV 'pth' matches files under this folder."
            )

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["full_path"]).convert("RGB")
        y = LABEL2ID[row["label"]]

        if self.transform:
            img = self.transform(img)

        # CrossEntropyLoss expects class indices as torch.long
        return img, torch.tensor(y, dtype=torch.long)
