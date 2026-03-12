"""
ShiftGuard10 Dataset & Augmentation Pipeline.
Handles class-balanced sampling and robust augmentation for distribution shift.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


# CIFAR-10 normalization stats (same distribution)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


class Cutout:
    """Randomly mask out a square patch from the image."""
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=img.dtype, device=img.device)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


def get_train_transforms():
    """Heavy augmentation pipeline for robust generalization."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(n_holes=1, length=8),
    ])


def get_val_transforms():
    """Clean validation transforms — no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def get_tta_transforms(n_views=5):
    """Test-Time Augmentation transforms — multiple stochastic views."""
    base = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    clean = get_val_transforms()
    return clean, base, n_views


class ShiftGuard10Dataset(Dataset):
    """
    Dataset for ShiftGuard10 competition.

    Args:
        root: Path to 'shift-guard-10-robust-image-classification-challenge/'
        split: 'train', 'val', or 'test'
        transform: Torchvision transforms pipeline
        val_ratio: Fraction of training data to use as validation (stratified)
        seed: Random seed for train/val split reproducibility
    """
    def __init__(self, root, split="train", transform=None,
                 val_ratio=0.1, seed=42):
        self.root = root
        self.split = split
        self.transform = transform

        if split in ("train", "val"):
            labels_df = pd.read_csv(os.path.join(root, "train_labels.csv"))
            # Clean whitespace from labels
            labels_df["label"] = labels_df["label"].str.strip()
            labels_df["id"] = labels_df["id"].astype(str).str.zfill(6)

            # Stratified train/val split
            rng = np.random.RandomState(seed)
            train_indices = []
            val_indices = []
            for cls in CLASS_NAMES:
                cls_indices = labels_df[labels_df["label"] == cls].index.tolist()
                rng.shuffle(cls_indices)
                n_val = max(1, int(len(cls_indices) * val_ratio))
                val_indices.extend(cls_indices[:n_val])
                train_indices.extend(cls_indices[n_val:])

            if split == "train":
                labels_df = labels_df.iloc[train_indices].reset_index(drop=True)
            else:
                labels_df = labels_df.iloc[val_indices].reset_index(drop=True)

            self.image_ids = labels_df["id"].tolist()
            self.labels = [CLASS_TO_IDX[lbl] for lbl in labels_df["label"]]
            self.image_dir = os.path.join(root, "train_images")

        elif split == "test":
            # Read test image IDs from sample_submission.csv
            sub_df = pd.read_csv(os.path.join(root, "sample_submission.csv"))
            sub_df["id"] = sub_df["id"].astype(str).str.zfill(6)
            self.image_ids = sub_df["id"].tolist()
            self.labels = None
            self.image_dir = os.path.join(root, "test_images")
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image, img_id  # Return id for test set

    def get_class_weights(self):
        """Compute inverse-frequency class weights for balanced loss."""
        counts = np.bincount(self.labels, minlength=len(CLASS_NAMES))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(CLASS_NAMES)
        return torch.FloatTensor(weights)

    def get_sampler(self):
        """Create WeightedRandomSampler for class-balanced batches."""
        counts = np.bincount(self.labels, minlength=len(CLASS_NAMES))
        class_weights = 1.0 / (counts + 1e-6)
        sample_weights = [class_weights[label] for label in self.labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.labels),
            replacement=True
        )
