import os
import ast
import json
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# =============================================================================
# Constants
# =============================================================================
CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

NUM_LABELS = len(CHEXPERT_LABELS)  # 13

# =============================================================================
# Label Mapping (U-Zeros)
# =============================================================================
def map_labels_uzeros(row: pd.Series, label_cols: List[str]) -> np.ndarray:
    targets = np.zeros(len(label_cols), dtype=np.float32)
    for i, col in enumerate(label_cols):
        targets[i] = 1.0 if row[col] == 1.0 else 0.0
    return targets


# =============================================================================
# Transforms
# =============================================================================
def get_chexpert_transform(train: bool = True, image_size: int = 224):
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
class CheXpertDataset(Dataset):
    def __init__(self, data_dir: str, csv_file: str, splits_json: str = "splits.json",
                 split: str = "train", image_size: int = 224, expand_channels: int = 1,
                 use_clahe: bool = False):
        self.data_dir = data_dir
        self.split = split
        self.expand_channels = expand_channels
        self.use_clahe = use_clahe
        self.transform = get_chexpert_transform(train=(split == "train"), image_size=image_size)

        # ---- Load CSV ----
        csv_path = os.path.join(data_dir, csv_file) if not os.path.isabs(csv_file) else csv_file
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # ---- Parse patient_id and study from ID ----
        df["patient_id"] = df["ID"].apply(lambda x: str(x).rsplit("_", 1)[0])
        df["study"] = df["ID"].apply(lambda x: str(x).rsplit("_", 1)[1])

        # ---- Parse image paths from 'images' column ----
        df["image_paths"] = df["images"].apply(self._parse_image_paths)

        # ---- Load splits ----
        splits_path = os.path.join(data_dir, splits_json) if not os.path.isabs(splits_json) else splits_json
        with open(splits_path, "r") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"split must be one of {list(splits.keys())}, got: {split}")

        # ---- Filter by split ----
        split_items = splits[split]
        if isinstance(split_items[0], dict):
            split_keys = set(
                (str(d["patient_id"]), str(d["study"]))
                for d in split_items
            )
            df = df[df.apply(
                lambda r: (str(r["patient_id"]), str(r["study"])) in split_keys, axis=1
            )].copy()
        else:
            split_ids = set(str(s) for s in split_items)
            df = df[df["ID"].astype(str).isin(split_ids)].copy()

        # ---- Build samples ----
        self.samples = self._build_pairs(df)
        self.targets = np.array([s["targets"] for s in self.samples])

        # ---- Print summary ----
        pos_counts = self.targets.sum(axis=0).astype(int)
        print(f"[CheXpertDataset] {split} (U-Zeros, {expand_channels}ch): "
              f"{len(self.samples)} samples | Image: {image_size}x{image_size}")
        print(f"  Positive counts: {dict(zip(CHEXPERT_LABELS, pos_counts))}")

    # -----------------------------------------------------------------
    # Parsing helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _parse_image_paths(images_str: str) -> List[str]:
        try:
            paths = ast.literal_eval(images_str)
            if isinstance(paths, list):
                return [str(p).strip() for p in paths]
        except (ValueError, SyntaxError):
            pass
        return [images_str.strip()]

    @staticmethod
    def _classify_view(path: str) -> str:
        lower = path.lower()
        if "frontal" in lower:
            return "frontal"
        elif "lateral" in lower:
            return "lateral"
        return "unknown"

    # -----------------------------------------------------------------
    # Pair building
    # -----------------------------------------------------------------
    def _build_pairs(self, df: pd.DataFrame) -> list:
        pairs = []
        skipped = 0

        for _, row in df.iterrows():
            paths = row["image_paths"]

            frontal_path = None
            lateral_path = None

            for p in paths:
                view = self._classify_view(p)
                if view == "frontal" and frontal_path is None:
                    frontal_path = p
                elif view == "lateral" and lateral_path is None:
                    lateral_path = p

            if frontal_path is None and lateral_path is None and len(paths) >= 2:
                frontal_path = paths[0]
                lateral_path = paths[1]

            if frontal_path is None or lateral_path is None:
                skipped += 1
                continue

            targets = map_labels_uzeros(row, CHEXPERT_LABELS)

            pairs.append({
                "sample_id": str(row["ID"]),
                "patient_id": str(row["patient_id"]),
                "study": str(row["study"]),
                "frontal_path": frontal_path,
                "lateral_path": lateral_path,
                "targets": targets,
            })

        if skipped > 0:
            print(f"  [Warning] Skipped {skipped} samples (missing frontal or lateral view)")

        return pairs

    # -----------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------
    def _load_and_transform(self, img_path: str) -> torch.Tensor:
        """Load grayscale image, optionally apply CLAHE, then transform."""
        if not os.path.isabs(img_path):
            full_path = os.path.join(self.data_dir, img_path)
        else:
            full_path = img_path

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
            img = img.expand(3, -1, -1)  # replicate grayscale to 3 channels
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
        frontal_img = self._load_and_transform(s["frontal_path"])
        images.append(frontal_img)
        lateral_img = self._load_and_transform(s["lateral_path"])
        images.append(lateral_img)

        images = torch.stack(images, dim=0)  # [2, C, H, W]
        targets = torch.tensor(s["targets"], dtype=torch.float32)  # [14]

        return images, targets, s["sample_id"]