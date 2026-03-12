"""
Utility functions: metrics, MixUp/CutMix, logging, seeding.
"""

import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report


def seed_everything(seed=42):
    """Deterministic seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_macro_f1(preds, targets, num_classes=10):
    """Compute macro F1 score (the competition metric)."""
    return f1_score(targets, preds, average="macro", zero_division=0)


def get_classification_report(preds, targets, class_names):
    """Full per-class precision/recall/F1 report."""
    labels = list(range(len(class_names)))
    return classification_report(
        targets, preds,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )


# ─── MixUp & CutMix ──────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=1.0):
    """MixUp: linear interpolation of input pairs."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: paste a random patch from one image onto another."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Generate random bounding box
    W, H = x.size(3), x.size(2)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    # Adjust lambda based on actual area ratio
    lam = 1 - (y2 - y1) * (x2 - x1) / (W * H)
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """Combined loss for MixUp/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(state, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"  [SAVED] Checkpoint: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint, returns epoch and best_f1."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_f1", 0.0)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
