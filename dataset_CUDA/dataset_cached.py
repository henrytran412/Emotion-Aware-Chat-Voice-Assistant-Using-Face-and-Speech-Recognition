import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CachedNpyDataset(Dataset):
    """
    Loads cached HWC uint8 arrays from .npy and applies normalization on the fly.
    """
    def __init__(self, cache_split_dir, mean, std, augment=False):
        self.cache_split_dir = cache_split_dir
        self.augment = augment

        labels_path = os.path.join(cache_split_dir, "labels.npy")
        meta = np.load(labels_path, allow_pickle=True)
        self.files = [m[0] for m in meta]
        self.labels = [int(m[1]) for m in meta]

        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        y = self.labels[idx]

        arr = np.load(os.path.join(self.cache_split_dir, fname))  # HWC uint8
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # CHW float

        if self.augment:
            # Horizontal flip
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])

            # Brightness jitter
            if torch.rand(1).item() < 0.7:
                b = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)  # [0.8, 1.2]
                x = torch.clamp(x * b, 0.0, 1.0)

            # Contrast jitter
            if torch.rand(1).item() < 0.7:
                c = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)  # [0.8, 1.2]
                x_mean = x.mean(dim=(1, 2), keepdim=True)
                x = torch.clamp((x - x_mean) * c + x_mean, 0.0, 1.0)

            # Random occlusion block to improve robustness to partial faces
            if torch.rand(1).item() < 0.3:
                _, h, w = x.shape
                block_h = max(8, int(h * (0.1 + torch.rand(1).item() * 0.2)))
                block_w = max(8, int(w * (0.1 + torch.rand(1).item() * 0.2)))
                y0 = int(torch.rand(1).item() * max(1, h - block_h))
                x0 = int(torch.rand(1).item() * max(1, w - block_w))
                x[:, y0:y0 + block_h, x0:x0 + block_w] = 0.0

        x = (x - self.mean) / self.std
        return x, torch.tensor(y, dtype=torch.long)
