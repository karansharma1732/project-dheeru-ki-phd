"""Dataset + dataloaders for the centralized TB classifier.

Reads the demographics CSV produced by ``parse_demographics.py`` and yields
(image_tensor, label) pairs while retaining per-sample metadata (sex, age_band,
dataset) for subgroup fairness evaluation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Grayscale CXR replicated to 3 channels; ImageNet stats for pretrained backbones.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )


class TBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, train: bool):
        # Drop rows whose image could not be located.
        self.df = df[df["image_path"].astype(str).str.len() > 0].reset_index(drop=True)
        self.tf = build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("L")
        x = self.tf(img)
        y = int(row["label"])
        meta = {
            "sex": row.get("sex", "unknown"),
            "age_band": row.get("age_band", "unknown"),
            "dataset": row.get("dataset", "unknown"),
        }
        return x, y, meta


def stratified_splits(
    df: pd.DataFrame, val_frac: float, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratify by (dataset, label) so each split keeps source + class balance."""
    strat = df["dataset"].astype(str) + "_" + df["label"].astype(str)
    train_df, temp_df = train_test_split(
        df, test_size=val_frac + test_frac, stratify=strat, random_state=seed
    )
    temp_strat = temp_df["dataset"].astype(str) + "_" + temp_df["label"].astype(str)
    rel_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_strat, random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metas = [b[2] for b in batch]
    return xs, ys, metas


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    csv_path = Path(cfg["paths"]["demographics_csv"])
    df = pd.read_csv(csv_path)
    df = df[df["dataset"].isin(cfg["data"]["datasets"])].reset_index(drop=True)

    train_df, val_df, test_df = stratified_splits(
        df, cfg["data"]["val_frac"], cfg["data"]["test_frac"], cfg["seed"]
    )

    size = cfg["data"]["image_size"]
    bs = cfg["train"]["batch_size"]
    nw = cfg["data"]["num_workers"]

    def make(d, train):
        return DataLoader(
            TBDataset(d, size, train),
            batch_size=bs,
            shuffle=train,
            num_workers=nw,
            collate_fn=_collate,
            pin_memory=True,
        )

    # Class weights for imbalance (inverse frequency), computed on the training split.
    counts = train_df["label"].value_counts().sort_index()
    weights = torch.tensor(
        [len(train_df) / (2 * counts.get(c, 1)) for c in (0, 1)], dtype=torch.float
    )

    print(
        f"[data] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"| class weights={weights.tolist()}"
    )
    return make(train_df, True), make(val_df, False), make(test_df, False), weights
