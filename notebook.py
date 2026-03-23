"""
ShiftGuard10 — All-in-One Kaggle Notebook
==========================================
Trains from scratch and generates submission.csv.

Key ingredients:
  1. PyramidNet-272 + ShakeDrop (SOTA from-scratch CIFAR arch)
  2. WRN-28-10 as secondary ensemble member
  3. Balanced Softmax loss (unbiased for long-tailed distributions)
  4. RandAugment + Cutout(16) + MixUp/CutMix
  5. Cosine LR with warmup + SWA
  6. Multi-seed ensemble (3 seeds)
  7. 20-view TTA at inference

Usage:
  python notebook.py                 # Full training on GPU
  python notebook.py --debug         # Quick smoke test (2 epochs, tiny subset)
"""

import os
import sys
import csv
import time
import math
import copy
import random
import argparse
from collections import Counter

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# --- Paths (resolved in main()) ---
DATA_ROOT = "shift-guard-10-robust-image-classification-challenge"

OUTPUT_DIR = "."
CHECKPOINT_DIR = "checkpoints"

# --- Classes ---
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = 10

# --- Normalization ---
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

# --- Training Hyperparameters ---
SEEDS          = [42, 137, 2024]
EPOCHS         = 300
BATCH_SIZE     = 128
LR             = 0.1
MOMENTUM       = 0.9
WEIGHT_DECAY   = 5e-4
WARMUP_EPOCHS  = 5
LABEL_SMOOTH   = 0.1
GRAD_CLIP      = 5.0

# MixUp / CutMix
MIXUP_ALPHA    = 1.0
CUTMIX_ALPHA   = 1.0
MIX_PROB       = 0.5

# SWA
SWA_START      = 250
SWA_LR         = 0.005

# TTA
TTA_VIEWS      = 20

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  UTILITIES                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


def compute_macro_f1(preds, targets):
    from sklearn.metrics import f1_score
    return f1_score(targets, preds, average="macro", zero_division=0)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  RANDAUGMENT (Pure PIL Implementation)                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def _apply_op(img, op_name, magnitude):
    """Apply a single augmentation operation to a PIL image."""
    if op_name == "ShearX":
        img = img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
    elif op_name == "ShearY":
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
    elif op_name == "TranslateX":
        pixels = int(magnitude * img.size[0])
        img = img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))
    elif op_name == "TranslateY":
        pixels = int(magnitude * img.size[1])
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))
    elif op_name == "Rotate":
        img = img.rotate(magnitude)
    elif op_name == "Brightness":
        img = ImageEnhance.Brightness(img).enhance(1.0 + magnitude)
    elif op_name == "Color":
        img = ImageEnhance.Color(img).enhance(1.0 + magnitude)
    elif op_name == "Contrast":
        img = ImageEnhance.Contrast(img).enhance(1.0 + magnitude)
    elif op_name == "Sharpness":
        img = ImageEnhance.Sharpness(img).enhance(1.0 + magnitude)
    elif op_name == "Posterize":
        bits = max(1, int(magnitude))
        img = ImageOps.posterize(img, bits)
    elif op_name == "Solarize":
        threshold = int(magnitude)
        img = ImageOps.solarize(img, threshold)
    elif op_name == "AutoContrast":
        img = ImageOps.autocontrast(img)
    elif op_name == "Equalize":
        img = ImageOps.equalize(img)
    elif op_name == "Invert":
        img = ImageOps.invert(img)
    return img


# Operation names and magnitude ranges (max magnitude)
_AUGMENT_LIST = [
    ("ShearX",        0.3),
    ("ShearY",        0.3),
    ("TranslateX",    0.3),
    ("TranslateY",    0.3),
    ("Rotate",        30),
    ("Brightness",    0.9),
    ("Color",         0.9),
    ("Contrast",      0.9),
    ("Sharpness",     0.9),
    ("Posterize",     4),
    ("Solarize",      256),
    ("AutoContrast",  0),
    ("Equalize",      0),
]


class RandAugment:
    """RandAugment: Practical automated data augmentation with a reduced search space."""
    def __init__(self, n_ops=2, magnitude=14, max_magnitude=30):
        self.n_ops = n_ops
        self.magnitude = magnitude
        self.max_magnitude = max_magnitude

    def __call__(self, img):
        ops = random.choices(_AUGMENT_LIST, k=self.n_ops)
        for op_name, max_val in ops:
            # Scale magnitude
            if max_val > 0:
                mag = (self.magnitude / self.max_magnitude) * max_val
                # Random sign for geometric transforms
                if op_name in ("ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate",
                               "Brightness", "Color", "Contrast", "Sharpness"):
                    if random.random() < 0.5:
                        mag = -mag
            else:
                mag = 0
            img = _apply_op(img, op_name, mag)
        return img


class Cutout:
    """Randomly mask out a square patch from a tensor image."""
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=img.dtype, device=img.device)
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


class NumpyToTensor:
    """Convert PIL image to tensor and normalize."""
    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        # HWC -> CHW
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        # Normalize
        mean = torch.tensor(CIFAR_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(CIFAR_STD, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std


class TrainTransform:
    """Full training augmentation pipeline: RandomCrop + Flip + RandAugment + Normalize + Cutout."""
    def __init__(self):
        self.rand_augment = RandAugment(n_ops=2, magnitude=14)
        self.to_tensor = NumpyToTensor()
        self.cutout = Cutout(length=16)

    def __call__(self, img):
        # RandomCrop with padding
        img = ImageOps.expand(img, border=4, fill=(128, 128, 128))
        w, h = img.size
        x = random.randint(0, w - 32)
        y = random.randint(0, h - 32)
        img = img.crop((x, y, x + 32, y + 32))

        # Random horizontal flip
        if random.random() < 0.5:
            img = ImageOps.mirror(img)

        # RandAugment
        img = self.rand_augment(img)

        # To tensor + normalize
        tensor = self.to_tensor(img)

        # Cutout
        tensor = self.cutout(tensor)
        return tensor


class ValTransform:
    """Clean validation/test transform."""
    def __init__(self):
        self.to_tensor = NumpyToTensor()

    def __call__(self, img):
        return self.to_tensor(img)


class TTATransform:
    """Stochastic TTA view."""
    def __init__(self):
        self.to_tensor = NumpyToTensor()

    def __call__(self, img):
        # Random crop with padding
        img = ImageOps.expand(img, border=4, fill=(128, 128, 128))
        w, h = img.size
        x = random.randint(0, w - 32)
        y = random.randint(0, h - 32)
        img = img.crop((x, y, x + 32, y + 32))

        # Random horizontal flip
        if random.random() < 0.5:
            img = ImageOps.mirror(img)

        # Light color jitter
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))

        # Small rotation
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, fillcolor=(128, 128, 128))

        return self.to_tensor(img)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  DATASET                                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class ShiftGuard10Dataset(Dataset):
    def __init__(self, root, split="train", transform=None, val_ratio=0.1, seed=42):
        self.root = root
        self.split = split
        self.transform = transform

        if split in ("train", "val"):
            labels_path = os.path.join(root, "train_labels.csv")
            self.image_ids = []
            self.labels = []

            with open(labels_path) as f:
                reader = csv.DictReader(f)
                all_ids = []
                all_labels = []
                for row in reader:
                    img_id = row["id"].strip().zfill(6)
                    label = row["label"].strip()
                    all_ids.append(img_id)
                    all_labels.append(label)

            # Stratified split
            rng = np.random.RandomState(seed)
            class_indices = {cls: [] for cls in CLASS_NAMES}
            for i, lbl in enumerate(all_labels):
                class_indices[lbl].append(i)

            train_idx, val_idx = [], []
            for cls in CLASS_NAMES:
                idxs = class_indices[cls]
                rng.shuffle(idxs)
                n_val = max(1, int(len(idxs) * val_ratio))
                val_idx.extend(idxs[:n_val])
                train_idx.extend(idxs[n_val:])

            chosen = train_idx if split == "train" else val_idx
            self.image_ids = [all_ids[i] for i in chosen]
            self.labels = [CLASS_TO_IDX[all_labels[i]] for i in chosen]
            self.image_dir = os.path.join(root, "train_images")

        elif split == "test":
            sub_path = os.path.join(root, "sample_submission.csv")
            self.image_ids = []
            self.labels = None
            with open(sub_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.image_ids.append(row["id"].strip().zfill(6))
            self.image_dir = os.path.join(root, "test_images")

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
        return image, img_id

    def get_class_counts(self):
        counts = np.bincount(self.labels, minlength=NUM_CLASSES)
        return counts

    def get_sampler(self):
        counts = np.bincount(self.labels, minlength=NUM_CLASSES)
        class_weights = 1.0 / (counts + 1e-6)
        sample_weights = [class_weights[label] for label in self.labels]
        return WeightedRandomSampler(sample_weights, len(self.labels), replacement=True)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MODELS                                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ─── ShakeDrop ───────────────────────────────────────────────────────────────

class ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=(-1, 1)):
        if training:
            gate = torch.bernoulli(torch.tensor(1.0 - p_drop)).item()
            ctx.save_for_backward(torch.tensor(gate))
            ctx.alpha_range = alpha_range
            if gate == 0:
                alpha = torch.empty(1).uniform_(*alpha_range).item()
                return alpha * x
            return x
        else:
            return (1.0 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate, = ctx.saved_tensors
        if gate.item() == 0:
            beta = torch.empty(1).uniform_(*ctx.alpha_range).item()
            return beta * grad_output, None, None, None
        return grad_output, None, None, None


def shakedrop(x, training=True, p_drop=0.5):
    return ShakeDropFunction.apply(x, training, p_drop, (-1, 1))


# ─── PyramidNet Basic Block ────────────────────────────────────────────────

class PyramidBasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, in_ch, out_ch, stride, p_drop):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.AvgPool2d(2) if stride != 1 else nn.Identity()
        self.pad = out_ch - in_ch
        self.p_drop = p_drop

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)

        if self.training:
            out = shakedrop(out, True, self.p_drop)

        shortcut = self.shortcut(x)
        if self.pad > 0:
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad))

        return out + shortcut


# ─── PyramidNet Bottleneck Block ───────────────────────────────────────────

class PyramidBottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, in_ch, out_ch, stride, p_drop):
        super().__init__()
        bottleneck_ch = out_ch // 4

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, bottleneck_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_ch)
        self.conv2 = nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_ch)
        self.conv3 = nn.Conv2d(bottleneck_ch, out_ch, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.AvgPool2d(2) if stride != 1 else nn.Identity()
        self.pad = out_ch - in_ch
        self.p_drop = p_drop

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.bn4(out)

        if self.training:
            out = shakedrop(out, True, self.p_drop)

        shortcut = self.shortcut(x)
        if self.pad > 0:
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad))

        return out + shortcut


class PyramidNet(nn.Module):
    """
    PyramidNet with ShakeDrop.
    Default config: depth=272, alpha=200, bottleneck=True
    This is the SOTA from-scratch architecture for CIFAR-10.
    """
    def __init__(self, depth=272, alpha=200, num_classes=10, bottleneck=True):
        super().__init__()
        if bottleneck:
            block = PyramidBottleneck
            n = (depth - 2) // 9
        else:
            block = PyramidBasicBlock
            n = (depth - 2) // 6

        self.in_ch = 16
        self.addrate = alpha / (3 * n)
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Linear schedule for ShakeDrop survival probability
        total_blocks = 3 * n
        self.layer1 = self._make_layer(block, n, stride=1, total_blocks=total_blocks, start_block=0)
        self.layer2 = self._make_layer(block, n, stride=2, total_blocks=total_blocks, start_block=n)
        self.layer3 = self._make_layer(block, n, stride=2, total_blocks=total_blocks, start_block=2*n)

        self.final_ch = self.in_ch
        self.bn_final = nn.BatchNorm2d(self.final_ch)
        self.fc = nn.Linear(self.final_ch, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, n_blocks, stride, total_blocks, start_block):
        layers = []
        for i in range(n_blocks):
            block_idx = start_block + i
            # Linear survival probability: p_l = 1 - l/L * (1 - p_L), p_L=0.5
            p_drop = (block_idx + 1) / total_blocks * 0.5

            out_ch = int(round(self.in_ch + self.addrate))
            if block.outchannel_ratio == 4:
                out_ch = int(round(out_ch / 4)) * 4  # ensure divisible by 4
                if out_ch < 4:
                    out_ch = 4

            s = stride if i == 0 else 1
            layers.append(block(self.in_ch, out_ch, s, p_drop))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.flatten(1)
        return self.fc(out)


# ─── WideResNet-28-10 ─────────────────────────────────────────────────────

class WRNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout=0.3):
        super().__init__()
        n = (depth - 4) // 6
        ch = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, ch[0], 3, stride=1, padding=1, bias=False)
        self.group1 = self._make_group(ch[0], ch[1], n, 1, dropout)
        self.group2 = self._make_group(ch[1], ch[2], n, 2, dropout)
        self.group3 = self._make_group(ch[2], ch[3], n, 2, dropout)
        self.bn = nn.BatchNorm2d(ch[3])
        self.fc = nn.Linear(ch[3], num_classes)
        self._init_weights()

    def _make_group(self, in_p, out_p, n, stride, dropout):
        layers = [WRNBlock(in_p, out_p, stride, dropout)]
        for _ in range(1, n):
            layers.append(WRNBlock(out_p, out_p, 1, dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  BALANCED SOFTMAX LOSS                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax: adjusts logits by log(class_prior) to debias
    the loss under long-tailed distributions.
    Ref: Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition" (NeurIPS 2020)
    """
    def __init__(self, class_counts, label_smoothing=0.1):
        super().__init__()
        # Log of class frequencies as bias
        freq = torch.tensor(class_counts, dtype=torch.float32)
        freq = freq / freq.sum()
        self.register_buffer("log_freq", torch.log(freq + 1e-12))
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Adjust logits: add log(class_prior) to each logit
        adjusted_logits = logits + self.log_freq.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets, label_smoothing=self.label_smoothing)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MIXUP / CUTMIX                                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    x_clone = x.clone()
    x_clone[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (W * H)
    return x_clone, y, y[idx], lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING LOOP                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    correct = total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        use_mix = False

        if MIX_PROB > 0 and random.random() < MIX_PROB:
            if random.random() < 0.5:
                images, ya, yb, lam = mixup_data(images, targets, MIXUP_ALPHA)
            else:
                images, ya, yb, lam = cutmix_data(images, targets, CUTMIX_ALPHA)
            outputs = model(images)
            loss = mix_criterion(criterion, outputs, ya, yb, lam)
            use_mix = True
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        if not use_mix:
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return loss_meter.avg, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_targets = [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_meter.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    macro_f1 = compute_macro_f1(all_preds, all_targets)
    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_targets))
    return loss_meter.avg, acc, macro_f1


def train_single_model(model_factory, model_name, seed, device, train_dataset, val_dataset,
                       class_counts, epochs=EPOCHS, debug=False):
    """Train a single model with a given seed and return the best state dict."""
    print(f"\n{'='*60}")
    print(f"  Training {model_name} | Seed: {seed}")
    print(f"{'='*60}")

    seed_everything(seed)

    # Re-create model each time for fresh init
    model = model_factory().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Loss
    criterion = BalancedSoftmaxLoss(class_counts, label_smoothing=LABEL_SMOOTH).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, nesterov=True
    )

    # LR scheduler: cosine with warmup
    total_epochs = epochs
    warmup = WARMUP_EPOCHS

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (total_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # SWA
    swa_start = SWA_START if not debug else 9999
    use_swa = swa_start < total_epochs
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    # Dataloader with balanced sampling
    sampler = train_dataset.get_sampler() if not debug else None
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=4 if not debug else 0,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=4 if not debug else 0, pin_memory=True
    )

    best_f1 = 0.0
    best_state = None

    for epoch in range(total_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        swa_tag = " [SWA]" if (use_swa and epoch >= swa_start) else ""
        if (epoch + 1) % 10 == 0 or epoch < 3 or (epoch + 1) == total_epochs:
            print(f"  E{epoch+1:3d}/{total_epochs} | "
                  f"TrL:{train_loss:.3f} TrA:{train_acc:.1f}% | "
                  f"VL:{val_loss:.3f} VA:{val_acc:.1f}% F1:{val_f1:.4f} | "
                  f"LR:{lr:.5f} | {elapsed:.1f}s{swa_tag}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    # SWA BN update
    if use_swa:
        print("  Updating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        val_loss, val_acc, val_f1 = validate(swa_model, val_loader, criterion, device)
        print(f"  SWA Val — Acc: {val_acc:.1f}% F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(swa_model.module.state_dict())

    print(f"  ✓ Best F1: {best_f1:.4f}")
    return best_state, best_f1


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE WITH TTA                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def predict_with_tta(model, test_dataset, device, n_views=TTA_VIEWS, batch_size=256):
    """Predict with TTA: 1 clean view + n_views augmented views, averaged."""
    model.eval()

    # Clean prediction
    test_dataset.transform = ValTransform()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_probs = []
    all_ids = []
    for images, ids in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_ids.extend(ids)

    accumulated = torch.cat(all_probs, dim=0)
    print(f"    Clean view done")

    # TTA views
    tta_transform = TTATransform()
    for view_idx in range(n_views):
        test_dataset.transform = tta_transform
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        view_probs = []
        for images, ids in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            view_probs.append(probs.cpu())
        accumulated += torch.cat(view_probs, dim=0)
        if (view_idx + 1) % 5 == 0:
            print(f"    TTA view {view_idx + 1}/{n_views} done")

    avg_probs = accumulated / (n_views + 1)
    return avg_probs, all_ids


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="ShiftGuard10 — Beat 0.95 F1")
    parser.add_argument("--debug", action="store_true", help="Quick test: 2 epochs, small data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to dataset directory (auto-detected if not set)")
    parser.add_argument("--pyramid-depth", type=int, default=110,
                        help="PyramidNet depth (272=SOTA but slow, 110=fast+strong, 164=balanced)")
    parser.add_argument("--pyramid-alpha", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-wrn", action="store_true", help="Skip WRN training (only PyramidNet)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: 2 seeds, 10 TTA views")
    args = parser.parse_args()

    # --- Resolve data root ---
    global DATA_ROOT
    if args.data_root:
        DATA_ROOT = args.data_root
    else:
        # Auto-detect: Kaggle > local directory
        candidates = [
            "/kaggle/input/shift-guard-10-robust-image-classification-challenge",
            "shift-guard-10-robust-image-classification-challenge",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "shift-guard-10-robust-image-classification-challenge"),
        ]
        for p in candidates:
            if os.path.isdir(p):
                DATA_ROOT = p
                break

    if not os.path.isdir(DATA_ROOT):
        print(f"ERROR: Data directory not found: {DATA_ROOT}")
        print(f"  Download and extract the competition data, then either:")
        print(f"    1. Place it in ./shift-guard-10-robust-image-classification-challenge/")
        print(f"    2. Use --data-root /path/to/data")
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ShiftGuard10 — Beat 0.95 F1")
    print(f"  Device: {device}")
    print(f"  PyramidNet-{args.pyramid_depth} (alpha={args.pyramid_alpha})")
    print(f"{'='*60}\n")

    # --- Debug overrides ---
    global EPOCHS, BATCH_SIZE, SWA_START, TTA_VIEWS, SEEDS
    if args.debug:
        EPOCHS = 2
        BATCH_SIZE = 32
        SWA_START = 9999
        TTA_VIEWS = 2
        SEEDS = [42]
        print("  >> DEBUG MODE: 2 epochs, small batches, 1 seed, 2 TTA views\n")

    if hasattr(args, 'fast') and args.fast and not args.debug:
        SEEDS = [42, 137]
        TTA_VIEWS = 10
        print("  >> FAST MODE: 2 seeds, 10 TTA views\n")

    if args.epochs is not None:
        EPOCHS = args.epochs

    # --- Datasets ---
    train_dataset = ShiftGuard10Dataset(DATA_ROOT, split="train", transform=TrainTransform(), val_ratio=0.1)
    val_dataset = ShiftGuard10Dataset(DATA_ROOT, split="val", transform=ValTransform(), val_ratio=0.1)
    test_dataset = ShiftGuard10Dataset(DATA_ROOT, split="test", transform=ValTransform())

    class_counts = train_dataset.get_class_counts()
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"  Class counts: {dict(zip(CLASS_NAMES, class_counts))}")

    if args.debug:
        # Subset for debug
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(200, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(50, len(val_dataset))))

    # --- Train all models (multi-seed ensemble) ---
    all_models = []  # List of (model_factory, state_dict, name)

    # Model factory functions
    def make_pyramid():
        return PyramidNet(depth=args.pyramid_depth, alpha=args.pyramid_alpha, num_classes=NUM_CLASSES)

    def make_wrn():
        return WideResNet(depth=28, widen_factor=10, num_classes=NUM_CLASSES, dropout=0.3)

    # Train PyramidNet with multiple seeds
    for seed in SEEDS:
        state, f1 = train_single_model(
            make_pyramid, f"PyramidNet-{args.pyramid_depth}", seed, device,
            train_dataset if not args.debug else train_dataset,
            val_dataset, class_counts, epochs=EPOCHS, debug=args.debug
        )
        all_models.append((make_pyramid, state, f"pyramid_s{seed}"))

    # Train WRN-28-10 with multiple seeds (for ensemble diversity)
    if not args.skip_wrn:
        for seed in SEEDS:
            state, f1 = train_single_model(
                make_wrn, "WRN-28-10", seed, device,
                train_dataset if not args.debug else train_dataset,
                val_dataset, class_counts, epochs=EPOCHS, debug=args.debug
            )
            all_models.append((make_wrn, state, f"wrn_s{seed}"))

    # --- Ensemble Inference with TTA ---
    print(f"\n{'='*60}")
    print(f"  Ensemble Inference: {len(all_models)} models × {TTA_VIEWS}+1 TTA views")
    print(f"{'='*60}")

    # Reload test dataset (clean transform)
    test_dataset = ShiftGuard10Dataset(DATA_ROOT, split="test", transform=ValTransform())

    ensemble_probs = None
    for model_factory, state_dict, name in all_models:
        print(f"\n  Predicting with {name}...")
        model = model_factory().to(device)
        model.load_state_dict(state_dict)
        model.eval()

        probs, ids = predict_with_tta(model, test_dataset, device,
                                       n_views=TTA_VIEWS, batch_size=256)
        if ensemble_probs is None:
            ensemble_probs = probs
        else:
            ensemble_probs += probs

        del model
        torch.cuda.empty_cache()

    ensemble_probs /= len(all_models)

    # --- Generate Submission ---
    preds = ensemble_probs.argmax(dim=1).numpy()
    labels = [IDX_TO_CLASS[p] for p in preds]

    output_path = os.path.join(OUTPUT_DIR, "submission.csv")
    with open(output_path, "w", newline="") as f:
        f.write("id,label\n")
        for img_id, label in zip(ids, labels):
            f.write(f"{img_id},{label}\n")

    print(f"\n{'='*60}")
    print(f"  ✓ Submission saved: {output_path}")
    print(f"  Total predictions: {len(ids)}")
    dist = Counter(labels)
    print(f"  Distribution:")
    for cls in CLASS_NAMES:
        print(f"    {cls:12s}: {dist.get(cls, 0):5d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
