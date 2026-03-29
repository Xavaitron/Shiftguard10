"""
ShiftGuard10 — All-in-One Training + Inference (v3)
=====================================================
Trains from scratch, generates submission.csv.

Features:
  - 3-seed ensemble (train thrice, average predictions)
  - More aggressive oversampling of tail classes (sqrt-inverse weighting)
  - Use 95% of data for training (5% val) — more data for tail classes
  - Train 450 epochs with SWA from 360
  - 30-view TTA
  - Save/load checkpoints to avoid retraining

No External Checkpoint is used in this competition , All the training is done from scratch
The model used in this script is generated from scrtch from this script itself
If you remove the model path then the model will be trained from scratch
Training from scratch takes around 60 hrs for 2xT4 GPUS 
Since the limit for kaggle is 30hrs training has been done in batches and the checkpoints are saved
NO EXTERNAL CHECKPOINT OR DATASET IS USED IN THIS SCRIPT
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
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import transforms

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

DATA_ROOT = "shift-guard-10-robust-image-classification-challenge"
OUTPUT_DIR = "."

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = 10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


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
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


def compute_macro_f1(preds, targets):
    from sklearn.metrics import f1_score
    return f1_score(targets, preds, average="macro", zero_division=0)


def get_classification_report(preds, targets):
    from sklearn.metrics import classification_report
    return classification_report(targets, preds, target_names=CLASS_NAMES, zero_division=0)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  AUGMENTATION                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class Cutout:
    """Randomly mask out a square patch from a tensor image."""
    def __init__(self, length=16):
        self.length = length
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=img.dtype)
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
        x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


def get_train_transforms():
    """Strong augmentation: AutoAugment(CIFAR10) + Cutout."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(length=16),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def get_tta_transform():
    """Stochastic TTA view transform."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(10, fill=128),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  DATASET                                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class ShiftGuard10Dataset(Dataset):
    def __init__(self, root, split="train", transform=None, val_ratio=0.05, seed=42):
        self.root = root
        self.split = split
        self.transform = transform

        if split in ("train", "val"):
            labels_path = os.path.join(root, "train_labels.csv")
            all_ids, all_labels = [], []
            with open(labels_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_ids.append(row["id"].strip().zfill(6))
                    all_labels.append(row["label"].strip())

            rng = np.random.RandomState(seed)
            class_indices = {cls: [] for cls in CLASS_NAMES}
            for i, lbl in enumerate(all_labels):
                class_indices[lbl].append(i)

            train_idx, val_idx = [], []
            for cls in CLASS_NAMES:
                idxs = class_indices[cls][:]
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
        return np.bincount(self.labels, minlength=NUM_CLASSES)

    def get_sampler(self):
        """Sqrt-inverse frequency sampler: more aggressive than inverse frequency
        for extremely imbalanced datasets. Upsamples tail classes more heavily."""
        counts = np.bincount(self.labels, minlength=NUM_CLASSES)
        # sqrt-inverse: balances between uniform and fully inverse
        class_weights = 1.0 / (np.sqrt(counts) + 1e-6)
        sample_weights = [class_weights[label] for label in self.labels]
        return WeightedRandomSampler(sample_weights, len(self.labels), replacement=True)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL: WideResNet-28-10                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

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
    """WRN-28-10 for 32x32 images. ~36M params."""
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0
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
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  LOSS: Balanced Softmax                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class BalancedSoftmaxLoss(nn.Module):
    """Adjusts logits by log(class_prior) to debias long-tailed distributions."""
    def __init__(self, class_counts, label_smoothing=0.1):
        super().__init__()
        freq = torch.tensor(class_counts, dtype=torch.float32)
        freq = freq / freq.sum()
        self.register_buffer("log_freq", torch.log(freq + 1e-12))
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        adjusted = logits + self.log_freq.unsqueeze(0)
        return F.cross_entropy(adjusted, targets, label_smoothing=self.label_smoothing)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MIXUP / CUTMIX                                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = random.randint(0, W - 1), random.randint(0, H - 1)
    x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
    x2, y2 = min(W, cx + cut_w // 2), min(H, cy + cut_h // 2)
    x_out = x.clone()
    x_out[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (W * H)
    return x_out, y, y[idx], lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING                                                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def train_one_epoch(model, loader, criterion, optimizer, device, mix_prob=0.5):
    model.train()
    loss_meter = AverageMeter()
    correct = total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        use_mix = False

        if mix_prob > 0 and random.random() < mix_prob:
            if random.random() < 0.5:
                images, ya, yb, lam = mixup_data(images, targets, 1.0)
            else:
                images, ya, yb, lam = cutmix_data(images, targets, 1.0)
            outputs = model(images)
            loss = mix_criterion(criterion, outputs, ya, yb, lam)
            use_mix = True
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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

    f1 = compute_macro_f1(all_preds, all_targets)
    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_targets))
    return loss_meter.avg, acc, f1, all_preds, all_targets


def train_single_seed(seed, args, device, class_counts):
    """Train one WRN-28-10 with a given seed. Returns best state_dict and val F1.
    Saves full checkpoint after every epoch (one file per seed, overwritten).
    Resumes from checkpoint if training was interrupted."""
    seed_everything(seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"wrn_seed{seed}.pth")

    print(f"\n{'='*60}")
    print(f"  Training seed={seed} | epochs={args.epochs}")
    print(f"{'='*60}\n")

    # Data with this seed's split
    train_ds = ShiftGuard10Dataset(DATA_ROOT, "train", get_train_transforms(),
                                    val_ratio=args.val_ratio, seed=42)  # fixed split seed
    val_ds   = ShiftGuard10Dataset(DATA_ROOT, "val", get_val_transforms(),
                                    val_ratio=args.val_ratio, seed=42)  # fixed split seed

    if args.debug:
        train_ds = Subset(train_ds, range(min(200, len(train_ds))))
        val_ds   = Subset(val_ds, range(min(50, len(val_ds))))

    sampler = train_ds.get_sampler() if not args.debug else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=4 if not args.debug else 0,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=4 if not args.debug else 0, pin_memory=True
    )

    model = WideResNet(depth=28, widen_factor=10, num_classes=NUM_CLASSES, dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: WRN-28-10 | Params: {n_params:,}")

    criterion = BalancedSoftmaxLoss(class_counts, label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.wd, nesterov=True
    )

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    swa_start = args.swa_start
    use_swa = swa_start < args.epochs
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

    best_f1 = 0.0
    best_state = None
    start_epoch = 0

    # ── Resume from checkpoint if it exists ──
    if os.path.isfile(ckpt_path):
        print(f"  Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model_state' in ckpt:
            # New-format checkpoint: full training state
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            best_f1 = ckpt['best_f1']
            best_state = ckpt['best_state']
            start_epoch = ckpt['epoch'] + 1
            if use_swa and 'swa_state' in ckpt and ckpt['swa_state'] is not None:
                swa_model.load_state_dict(ckpt['swa_state'])
            if 'swa_scheduler_state' in ckpt and ckpt['swa_scheduler_state'] is not None:
                swa_scheduler.load_state_dict(ckpt['swa_scheduler_state'])
            print(f"  Resumed at epoch {start_epoch}/{args.epochs} | best F1 so far: {best_f1:.4f}")
        else:
            # Old-format checkpoint (plain state_dict) — load weights, train from scratch
            model.load_state_dict(ckpt)
            print(f"  Loaded old-format checkpoint weights, training from epoch 0")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.mix_prob
        )

        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        val_loss, val_acc, val_f1, preds, targets = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        swa_tag = " [SWA]" if (use_swa and epoch >= swa_start) else ""
        print(f"  E{epoch+1:3d}/{args.epochs} | "
              f"TrL:{train_loss:.3f} TrA:{train_acc:.1f}% | "
              f"VL:{val_loss:.3f} VA:{val_acc:.1f}% F1:{val_f1:.4f} | "
              f"LR:{lr:.6f} | {elapsed:.1f}s{swa_tag}")

        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epochs:
            print(get_classification_report(preds, targets))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

        # ── Save checkpoint after every epoch (overwrite) ──
        ckpt_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_f1': best_f1,
            'best_state': best_state,
            'swa_state': swa_model.state_dict() if (use_swa and epoch >= swa_start) else None,
            'swa_scheduler_state': swa_scheduler.state_dict() if (use_swa and epoch >= swa_start) else None,
            'completed': False,
        }
        torch.save(ckpt_data, ckpt_path)

    # SWA finalize
    if use_swa:
        print("\n  Updating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        _, val_acc, val_f1, preds, targets = validate(swa_model, val_loader, criterion, device)
        print(f"  SWA Val — Acc: {val_acc:.1f}% F1: {val_f1:.4f}")
        print(get_classification_report(preds, targets))
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(swa_model.module.state_dict())

    print(f"  Seed {seed} best F1: {best_f1:.4f}")

    # ── Mark checkpoint as completed ──
    ckpt_data = {
        'epoch': args.epochs - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_f1': best_f1,
        'best_state': best_state,
        'swa_state': swa_model.state_dict() if use_swa else None,
        'swa_scheduler_state': swa_scheduler.state_dict() if use_swa else None,
        'completed': True,
    }
    torch.save(ckpt_data, ckpt_path)
    print(f"  Saved (completed): {ckpt_path}")

    return best_state, best_f1


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE WITH TTA                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def predict_with_tta(model, test_dataset, device, n_views=30, batch_size=512):
    """1 clean view + n_views augmented views, softmax averaged."""
    model.eval()

    # Clean prediction
    test_dataset.transform = get_val_transforms()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_probs, all_ids = [], []
    for images, ids in loader:
        images = images.to(device)
        probs = F.softmax(model(images), dim=1)
        all_probs.append(probs.cpu())
        all_ids.extend(ids)
    accumulated = torch.cat(all_probs, dim=0)
    print(f"    Clean view done ({len(all_ids)} samples)")

    # TTA views
    for v in range(n_views):
        test_dataset.transform = get_tta_transform()
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        view_probs = []
        for images, ids in loader:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1)
            view_probs.append(probs.cpu())
        accumulated += torch.cat(view_probs, dim=0)
        if (v + 1) % 5 == 0 or (v + 1) == n_views:
            print(f"    TTA view {v+1}/{n_views} done")

    return accumulated / (n_views + 1), all_ids


def generate_submission(probs, ids, output_path):
    """Write submission.csv from averaged probabilities."""
    preds = probs.argmax(dim=1).numpy()
    labels = [IDX_TO_CLASS[p] for p in preds]
    with open(output_path, "w", newline="") as f:
        f.write("id,label\n")
        for img_id, label in zip(ids, labels):
            f.write(f"{img_id},{label}\n")

    dist = Counter(labels)
    print(f"\n  Submission saved: {output_path}")
    print(f"  Total: {len(ids)} predictions")
    print(f"  Distribution:")
    for cls in CLASS_NAMES:
        print(f"    {cls:12s}: {dist.get(cls, 0):5d}")
    return labels


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="ShiftGuard10")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-root", type=str, default="/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge")
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of data for validation (smaller = more training data)")
    parser.add_argument("--swa-start", type=int, default=360,
                        help="Epoch to start SWA")
    parser.add_argument("--swa-lr", type=float, default=0.005)
    parser.add_argument("--tta", type=int, default=30, help="Number of TTA views")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 7],
                        help="Seeds for ensemble (e.g. --seeds 42 137 7)")
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip training, load checkpoints and run inference only")
    parser.add_argument("--checkpoint-dir", type=str, default="/kaggle/input/models/xavaitron/wrn-3seed/pytorch/default/1",
                        help="Directory to save/load checkpoints")
    args, _ = parser.parse_known_args()

    # --- Resolve data root ---
    global DATA_ROOT
    if args.data_root:
        DATA_ROOT = args.data_root
    else:
        candidates = [
            "/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge",
            "/kaggle/input/shift-guard-10-robust-image-classification-challenge",
            "shift-guard-10-robust-image-classification-challenge",
            os.path.join(os.getcwd(),
                         "shift-guard-10-robust-image-classification-challenge"),
        ]
        for p in candidates:
            if os.path.isdir(p):
                DATA_ROOT = p
                break

    if not os.path.isdir(DATA_ROOT):
        print(f"ERROR: Data not found at: {DATA_ROOT}")
        print(f"  Use --data-root /path/to/data")
        sys.exit(1)

    # --- Debug overrides ---
    if args.debug:
        args.epochs = 2
        args.batch_size = 32
        args.swa_start = 9999
        args.tta = 2
        args.seeds = [42]
        print(">> DEBUG MODE: 2 epochs, 1 seed, 2 TTA views\n")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        print(f"  GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    print(f"{'='*60}")
    print(f"  ShiftGuard10 v3")
    print(f"  Device:    {device}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  LR:        {args.lr}")
    print(f"  Val ratio: {args.val_ratio}")
    print(f"  SWA:       epoch {args.swa_start} (lr={args.swa_lr})")
    print(f"  TTA:       {args.tta} views")
    print(f"  Data:      {DATA_ROOT}")
    print(f"{'='*60}\n")

    # Get class counts (from full training set with fixed split)
    tmp_ds = ShiftGuard10Dataset(DATA_ROOT, "train", val_ratio=args.val_ratio, seed=42)
    class_counts = tmp_ds.get_class_counts()
    print(f"  Train: {len(tmp_ds)} samples")
    print(f"  Class counts: {dict(zip(CLASS_NAMES, class_counts))}\n")
    del tmp_ds

    # ─── Train with each seed (or load checkpoints if they exist) ───
    all_states = []
    for seed in args.seeds:
        ckpt_path = os.path.join(args.checkpoint_dir, f"wrn_seed{seed}.pth")
        if os.path.isfile(ckpt_path):
            print(f"\n  Checkpoint found: {ckpt_path} — skipping training for seed={seed}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            best_state = ckpt.get('best_state', ckpt.get('model_state', ckpt))
            all_states.append(best_state)
        else:
            if args.inference_only:
                print(f"ERROR: Checkpoint not found: {ckpt_path} (inference-only mode)")
                sys.exit(1)
            state, f1 = train_single_seed(seed, args, device, class_counts)
            all_states.append(state)

    # ─── Ensemble Inference with TTA ─────────────────────────
    print(f"\n{'='*60}")
    print(f"  Ensemble Inference: {len(all_states)} model(s) × {args.tta}+1 TTA views")
    print(f"{'='*60}")

    test_dataset = ShiftGuard10Dataset(DATA_ROOT, "test", get_val_transforms())
    print(f"  Test samples: {len(test_dataset)}")

    ensemble_probs = None
    for i, state_dict in enumerate(all_states):
        print(f"\n  Model {i+1}/{len(all_states)} (seed={args.seeds[i]})...")
        model = WideResNet(depth=28, widen_factor=10, num_classes=NUM_CLASSES, dropout=0.3).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        probs, ids = predict_with_tta(model, test_dataset, device,
                                       n_views=args.tta, batch_size=512)
        if ensemble_probs is None:
            ensemble_probs = probs
        else:
            ensemble_probs += probs

        del model
        torch.cuda.empty_cache()

    ensemble_probs /= len(all_states)

    # ─── Generate Submission ─────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, args.output)
    generate_submission(ensemble_probs, ids, output_path)

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Ensemble: {len(all_states)} models")
    print(f"  Submission: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
