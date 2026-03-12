"""
ShiftGuard10 Training Script.

Supports CCT and WideResNet models with:
- Class-weighted cross-entropy + label smoothing
- MixUp / CutMix augmentation
- Cosine LR with warmup
- SWA (Stochastic Weight Averaging)
- Debug mode for CPU testing

Usage:
  python src/train.py --model cct --epochs 200
  python src/train.py --model wrn --epochs 200
  python src/train.py --model cct --debug        # Quick sanity check
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    ShiftGuard10Dataset, get_train_transforms, get_val_transforms, CLASS_NAMES
)
from src.models.cct import cct_7_3x1
from src.models.wideresnet import wrn_28_10
from src.utils import (
    seed_everything, compute_macro_f1, get_classification_report,
    mixup_data, cutmix_data, mixup_cutmix_criterion,
    save_checkpoint, load_checkpoint, AverageMeter
)


def build_model(cfg, device):
    """Build model based on config."""
    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]

    if model_name == "cct":
        cct_cfg = cfg["model"]["cct"]
        model = cct_7_3x1(
            num_classes=num_classes,
            embed_dim=cct_cfg["embed_dim"],
            num_heads=cct_cfg["num_heads"],
            num_layers=cct_cfg["num_layers"],
            mlp_ratio=cct_cfg["mlp_ratio"],
            n_conv_layers=cct_cfg["n_conv_layers"],
            kernel_size=cct_cfg["kernel_size"],
            dropout=cct_cfg["dropout"],
            attn_dropout=cct_cfg["attn_dropout"],
            stochastic_depth=cct_cfg["stochastic_depth"],
        )
    elif model_name == "wrn":
        wrn_cfg = cfg["model"]["wrn"]
        model = wrn_28_10(
            num_classes=num_classes,
            dropout=wrn_cfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_name} | Params: {n_params:,}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, epoch):
    """Train for one epoch with optional MixUp/CutMix."""
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    mix_prob = cfg["training"]["mix_prob"]
    mixup_alpha = cfg["training"]["mixup_alpha"]
    cutmix_alpha = cfg["training"]["cutmix_alpha"]

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        # Apply MixUp or CutMix with probability
        use_mix = np.random.rand() < mix_prob and epoch >= cfg["training"]["warmup_epochs"]
        if use_mix:
            if np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
            else:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)

        outputs = model(images)

        if use_mix:
            loss = mixup_cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
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
    """Validate and compute macro F1."""
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_targets = []

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
    return loss_meter.avg, acc, macro_f1, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="ShiftGuard10 Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, choices=["cct", "wrn"], default=None,
                        help="Override model from config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 2 epochs, 200 images, no SWA")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (e.g. --gpu 1 for cuda:1)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr

    # Debug mode adjustments
    if args.debug:
        cfg["training"]["epochs"] = 2
        cfg["training"]["batch_size"] = 32
        cfg["training"]["swa_start"] = 9999  # disable SWA
        cfg["data"]["num_workers"] = 0       # avoid multiprocessing issues
        print("=" * 60)
        print("  DEBUG MODE - 2 epochs, small batches, no SWA")
        print("=" * 60)

    seed_everything(cfg["seed"])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"  ShiftGuard10 Training")
    print(f"  Device: {device}")
    print(f"  Model:  {cfg['model']['name']}")
    print(f"  Epochs: {cfg['training']['epochs']}")
    print(f"  Batch:  {cfg['training']['batch_size']}")
    print(f"  LR:     {cfg['training']['lr']}")
    print(f"{'='*60}\n")

    # ─── Data ────────────────────────────────────────────────
    data_root = cfg["data"]["root"]
    train_dataset = ShiftGuard10Dataset(
        root=data_root, split="train",
        transform=get_train_transforms(),
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["seed"]
    )
    val_dataset = ShiftGuard10Dataset(
        root=data_root, split="val",
        transform=get_val_transforms(),
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["seed"]
    )

    # Class-balanced sampler for training (before any subsetting)
    sampler = None
    shuffle = True
    if cfg["training"]["use_balanced_sampler"] and not args.debug:
        sampler = train_dataset.get_sampler()
        shuffle = False  # sampler handles ordering

    if args.debug:
        # Subset for debug — after sampler creation since Subset lacks get_sampler()
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(200, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(50, len(val_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    # ─── Model & Optimizer ───────────────────────────────────
    model = build_model(cfg, device)

    # Class-weighted loss
    if cfg["training"]["use_class_weights"] and not args.debug:
        # Recompute from full training set (not subset)
        full_train = ShiftGuard10Dataset(
            root=data_root, split="train",
            val_ratio=cfg["data"]["val_ratio"],
            seed=cfg["seed"]
        )
        class_weights = full_train.get_class_weights().to(device)
        print(f"  Class weights: {class_weights.cpu().numpy().round(2)}")
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["training"]["label_smoothing"],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
        nesterov=True,
    )

    # Cosine annealing with warmup
    total_epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # SWA setup
    swa_start = cfg["training"]["swa_start"]
    use_swa = swa_start < total_epochs
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg["training"]["swa_lr"])
        print(f"  SWA enabled: starts at epoch {swa_start}")

    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        start_epoch, best_f1 = load_checkpoint(args.resume, model, optimizer)
        print(f"  Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # ─── Training Loop ───────────────────────────────────────
    checkpoint_dir = cfg["output"]["checkpoint_dir"]
    model_name = cfg["model"]["name"]
    print(f"\n  Starting training...\n")

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg, epoch
        )

        # LR schedule
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validate
        val_loss, val_acc, val_f1, preds, targets = validate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        swa_tag = " [SWA]" if (use_swa and epoch >= swa_start) else ""
        print(
            f"  Epoch {epoch+1:3d}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% F1: {val_f1:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s{swa_tag}"
        )

        # Save best model
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": cfg,
            }, os.path.join(checkpoint_dir, f"best_{model_name}.pth"))

        # Periodic checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": cfg,
            }, os.path.join(checkpoint_dir, f"{model_name}_epoch{epoch+1}.pth"))

    # ─── SWA BN Update ───────────────────────────────────────
    if use_swa:
        print("\n  Updating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Validate SWA model
        val_loss, val_acc, val_f1, preds, targets = validate(
            swa_model, val_loader, criterion, device
        )
        print(f"  SWA Val - Acc: {val_acc:.1f}% F1: {val_f1:.4f}")

        save_checkpoint({
            "epoch": total_epochs,
            "model_state_dict": swa_model.module.state_dict(),
            "best_f1": val_f1,
            "config": cfg,
        }, os.path.join(checkpoint_dir, f"swa_{model_name}.pth"))

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint({
                "epoch": total_epochs,
                "model_state_dict": swa_model.module.state_dict(),
                "best_f1": val_f1,
                "config": cfg,
            }, os.path.join(checkpoint_dir, f"best_{model_name}.pth"))

    # ─── Final Report ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Macro F1: {best_f1:.4f}")
    print(f"{'='*60}")

    # Print per-class breakdown
    print(f"\n  Per-class breakdown (final epoch):")
    print(get_classification_report(preds, targets, CLASS_NAMES))


if __name__ == "__main__":
    main()
