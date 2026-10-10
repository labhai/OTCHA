"""
MURA Dataset

Returns (images[2, C, H, W], target, sample_id) per study view pair.

Input format:
- Grayscale images, resized and normalized with mean/std 0.5
- The loader expands to three channels when expand_channels=3

Data source:
- mura_train.csv / mura_valid.csv / mura_test.csv
  Columns: patient_id, study, part, images, target
  images column: string representation of list with 2 image paths
- Images: {data_dir}/{image_path} (e.g., data_dir/MURA-v1.1/train/XR_SHOULDER/...)
"""

import os
import ast
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# =============================================================================
# Transforms
# =============================================================================
def get_mura_transform(train=True, image_size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),  # [1, H, W], range [0, 1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [1, H, W], range [0, 1]
        ])


# =============================================================================
# Dataset
# =============================================================================
class MURADataset(Dataset):
    def __init__(self, data_dir: str, csv_file: str, split: str = "train",
                 image_size: int = 224, expand_channels: int = 1, use_clahe: bool = False):
        self.data_dir = data_dir
        self.split = split
        self.expand_channels = expand_channels
        self.use_clahe = use_clahe
        self.transform = get_mura_transform(train=(split == "train"), image_size=image_size)

        # ---- Load CSV ----
        csv_path = os.path.join(data_dir, csv_file) if not os.path.isabs(csv_file) else csv_file
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # ---- Build samples ----
        self.samples = []
        for _, row in df.iterrows():
            image_paths = ast.literal_eval(row["images"])  # str → list
            self.samples.append({
                "patient_id": int(row["patient_id"]),
                "study": int(row["study"]),
                "part": str(row["part"]),
                "image_paths": image_paths,
                "label": int(row["target"]),
                "sample_id": f"patient{row['patient_id']:05d}_study{row['study']}_{row['part']}",
            })

        self.labels = np.array([s["label"] for s in self.samples])

        # ---- Print summary ----
        pos = (self.labels == 1).sum()
        neg = (self.labels == 0).sum()
        n_patients = df["patient_id"].nunique()
        print(f"[MURADataset] {split} ({expand_channels}ch): {len(self.samples)} samples | "
              f"Pos: {pos}, Neg: {neg} | Patients: {n_patients} | Image: {image_size}x{image_size}")

    # -----------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------
    def _load_and_transform(self, image_path: str):
        full_path = os.path.join(self.data_dir, image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        img = Image.open(full_path).convert("L")  # Grayscale (1ch)

        if self.use_clahe:
            img_np = np.array(img, dtype=np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_np = clahe.apply(img_np)
            img = Image.fromarray(img_np, mode="L")

        img = self.transform(img)  # [1, H, W], range [0, 1]

        # Channel expansion: [1, H, W] -> [3, H, W]
        if self.expand_channels == 3:
            img = img.expand(3, -1, -1)
            img = transforms.functional.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            img = transforms.functional.normalize(img, mean=[0.5], std=[0.5])

        return img

    # -----------------------------------------------------------------
    # __getitem__
    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        images = []
        for img_path in s["image_paths"]:
            img = self._load_and_transform(img_path)
            images.append(img)

        images = torch.stack(images, dim=0)  # [2, C, H, W]
        target = torch.tensor(s["label"], dtype=torch.long)

        return images, target, s["sample_id"]
