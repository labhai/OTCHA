"""
VinDr-Mammo Dataset

Supports two modes:
- mode='2view': Returns (cc_img, mlo_img, target) per breast → for XFMamba
- mode='4view': Returns (images[4,1,H,W], target, study_id) per patient → for MV-HFMD

Input format (both modes):
- 1-channel grayscale, range [0, 1], no ImageNet normalization
- Each model handles channel expansion internally
  (e.g., XFMamba: x.expand(-1, 3, -1, -1))

Data source:
- breast-level_annotation.csv: study_id, image_id, laterality, view_position, breast_birads, ...
- splits.json: {"train": [...], "val": [...], "test": [...]}
- Images: {data_dir}/images/{study_id}/{image_id}.png (or .dicom converted)
"""

import os
import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# =============================================================================
# Labels
# =============================================================================
def birads_to_binary(birads_str) -> Optional[int]:
    if pd.isna(birads_str):
        return None
    try:
        b = int(str(birads_str).replace("BI-RADS", "").strip())
    except (ValueError, TypeError):
        return None

    if b in (1, 2):
        return 0
    if b in (3, 4, 5):
        return 1
    return None


# =============================================================================
# Transforms
# =============================================================================
def get_vindr_transform(train=True, image_size=224):
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
class VinDrDataset(Dataset):
    def __init__(self, data_dir: str, csv_file: str, splits_json: str = "splits.json",
                 split: str = "train", mode: str = "2view", image_size: int = 224, expand_channels: int = 1, use_clahe: bool = False):
        self.data_dir = data_dir
        self.split = split
        self.mode = mode
        self.expand_channels = expand_channels
        self.use_clahe = use_clahe
        self.transform = get_vindr_transform(train=(split == "train"), image_size=image_size)

        # ---- Load CSV ----
        csv_path = os.path.join(data_dir, csv_file) if not os.path.isabs(csv_file) else csv_file
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df["study_id"] = df["study_id"].astype(str)
        df["image_id"] = df["image_id"].astype(str)
        df["laterality"] = df["laterality"].astype(str)
        df["view_position"] = df["view_position"].astype(str).str.upper()

        # ---- Load splits ----
        splits_path = os.path.join(data_dir, splits_json) if not os.path.isabs(splits_json) else splits_json
        with open(splits_path, "r") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"split must be one of {list(splits.keys())}, got: {split}")

        # splits.json format: [{"patient_id": "...", "laterality": "L"}, ...]
        # CSV uses "study_id" which corresponds to "patient_id" in splits.json
        split_items = splits[split]
        if isinstance(split_items[0], dict):
            split_keys = set(
                (str(d["patient_id"]), str(d["laterality"]))
                for d in split_items
            )
            df = df[df.apply(
                lambda r: (str(r["study_id"]), str(r["laterality"])) in split_keys, axis=1
            )].copy()
        else:
            # Fallback: simple list of study_ids
            split_study_ids = set(str(s) for s in split_items)
            df = df[df["study_id"].isin(split_study_ids)].copy()

        # ---- Build label ----
        df["label"] = df["breast_birads"].apply(birads_to_binary)
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)

        # ---- Build samples ----
        if mode == "2view":
            self.samples = self._build_2view_pairs(df)
        elif mode == "4view":
            self.samples = self._build_4view_groups(df)
        else:
            raise ValueError(f"mode must be '2view' or '4view', got: {mode}")

        self.labels = np.array([s["label"] for s in self.samples])

        # ---- Print summary ----
        pos = (self.labels == 1).sum()
        neg = (self.labels == 0).sum()
        print(f"[VinDrDataset] {split} ({mode}, {expand_channels}ch): {len(self.samples)} samples | "
              f"Pos: {pos}, Neg: {neg} | Image: {image_size}x{image_size}")

    # -----------------------------------------------------------------
    # Pair building
    # -----------------------------------------------------------------
    def _build_2view_pairs(self, df: pd.DataFrame) -> list:
        pairs = []
        grouped = df.groupby(["study_id", "laterality"])

        for (study_id, lat), group in grouped:
            cc_rows = group[group["view_position"] == "CC"]
            mlo_rows = group[group["view_position"] == "MLO"]

            if cc_rows.empty or mlo_rows.empty:
                continue

            cc_row = cc_rows.iloc[0]
            mlo_row = mlo_rows.iloc[0]
            label = int(cc_row["label"])

            pairs.append({
                "study_id": study_id,
                "laterality": lat,
                "cc_image_id": cc_row["image_id"],
                "mlo_image_id": mlo_row["image_id"],
                "label": label,
                "sample_id": f"{study_id}_{lat}",
            })

        return pairs

    def _build_4view_groups(self, df: pd.DataFrame) -> list:
        groups = []
        grouped = df.groupby("study_id")

        for study_id, group in grouped:
            view_map = {}
            for _, row in group.iterrows():
                key = f"{row['laterality']}-{row['view_position']}"
                if key not in view_map:
                    view_map[key] = row

            required = ["L-CC", "L-MLO", "R-CC", "R-MLO"]
            if not all(k in view_map for k in required):
                continue

            label = max(int(view_map[k]["label"]) for k in required)

            groups.append({
                "study_id": study_id,
                "image_ids": {k: view_map[k]["image_id"] for k in required},
                "label": label,
                "sample_id": study_id,
            })

        return groups

    # -----------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------
    def _find_image_path(self, study_id: str, image_id: str) -> str:
        base = os.path.join(self.data_dir, "images", study_id, image_id)

        for ext in ["", ".png", ".jpg", ".jpeg", ".dicom"]:
            path = f"{base}{ext}"
            if os.path.exists(path):
                return path

        return f"{base}.png"

    def _load_and_transform(self, study_id: str, image_id: str):
        img_path = self._find_image_path(study_id, image_id)
        img = Image.open(img_path).convert("L")  # Grayscale (1ch)

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
        if self.mode == "2view":
            return self._getitem_2view(idx)
        else:
            return self._getitem_4view(idx)

    def _getitem_2view(self, idx):
        s = self.samples[idx]

        images = []
        cc_img = self._load_and_transform(s["study_id"], s["cc_image_id"])
        images.append(cc_img)
        mlo_img = self._load_and_transform(s["study_id"], s["mlo_image_id"])
        images.append(mlo_img)

        images = torch.stack(images, dim=0)  # [2, C, H, W]
        target = torch.tensor(s["label"], dtype=torch.long)

        return images, target, s["sample_id"]

    def _getitem_4view(self, idx):
        s = self.samples[idx]
        view_order = ["L-CC", "L-MLO", "R-CC", "R-MLO"]

        images = []
        for view_key in view_order:
            img = self._load_and_transform(s["study_id"], s["image_ids"][view_key])
            images.append(img)

        images = torch.stack(images, dim=0)  # [4, 1, H, W]
        target = torch.tensor(s["label"], dtype=torch.long)

        return images, target, s["sample_id"]