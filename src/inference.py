"""
ShiftGuard10 Inference — generates submission.csv with optional TTA.

Usage:
  python src/inference.py --checkpoint checkpoints/best_cct.pth
  python src/inference.py --checkpoint checkpoints/best_cct.pth --tta 10
  python src/inference.py --checkpoint checkpoints/best_cct.pth checkpoints/best_wrn.pth  # ensemble
"""

import os
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    ShiftGuard10Dataset, get_val_transforms, get_tta_transforms,
    IDX_TO_CLASS, CLASS_NAMES
)
from src.models.cct import cct_7_3x1
from src.models.wideresnet import wrn_28_10


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
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

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Loaded {model_name} from {checkpoint_path}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}, F1: {checkpoint.get('best_f1', '?'):.4f}")
    return model, cfg


@torch.no_grad()
def predict_clean(model, loader, device):
    """Standard prediction without TTA."""
    all_probs = []
    all_ids = []
    for images, ids in tqdm(loader, desc="Predicting"):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_ids.extend(ids)
    return torch.cat(all_probs, dim=0), all_ids


@torch.no_grad()
def predict_tta(model, dataset, device, n_views=5, batch_size=256):
    """
    Test-Time Augmentation: average predictions over multiple augmented views.
    First view is always the clean (un-augmented) version.
    """
    clean_transform, aug_transform, _ = get_tta_transforms(n_views)

    # Clean prediction
    dataset.transform = clean_transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    clean_probs, all_ids = predict_clean(model, loader, device)
    accumulated = clean_probs.clone()

    # Augmented predictions
    dataset.transform = aug_transform
    for view_idx in range(1, n_views):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        aug_probs, _ = predict_clean(model, loader, device)
        accumulated += aug_probs
        print(f"    TTA view {view_idx + 1}/{n_views} done")

    avg_probs = accumulated / n_views
    return avg_probs, all_ids


def generate_submission(probs, ids, output_path):
    """Generate submission CSV from probabilities."""
    preds = probs.argmax(dim=1).numpy()
    labels = [IDX_TO_CLASS[p] for p in preds]

    with open(output_path, "w") as f:
        f.write("id,label\n")
        for img_id, label in zip(ids, labels):
            # Ensure the id format matches expected (no leading zeros stripped)
            f.write(f"{img_id},{label}\n")

    print(f"\n  [SAVED] Submission: {output_path}")
    print(f"    Samples: {len(ids)}")

    # Print class distribution of predictions
    from collections import Counter
    dist = Counter(labels)
    print(f"    Prediction distribution:")
    for cls in CLASS_NAMES:
        print(f"      {cls:12s}: {dist.get(cls, 0):5d}")


def main():
    parser = argparse.ArgumentParser(description="ShiftGuard10 Inference")
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="Path(s) to model checkpoint(s). Multiple = ensemble.")
    parser.add_argument("--data-root", type=str,
                        default="shift-guard-10-robust-image-classification-challenge",
                        help="Path to data directory")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output submission CSV path")
    parser.add_argument("--tta", type=int, default=0,
                        help="Number of TTA views (0 = no TTA)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Inference batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ShiftGuard10 Inference")
    print(f"  Device: {device}")
    print(f"  Checkpoints: {args.checkpoint}")
    print(f"  TTA views: {args.tta if args.tta > 0 else 'disabled'}")
    print(f"{'='*60}\n")

    # Test dataset
    test_dataset = ShiftGuard10Dataset(
        root=args.data_root, split="test",
        transform=get_val_transforms()
    )
    print(f"  Test samples: {len(test_dataset)}")

    # Predict with each model
    ensemble_probs = None

    for ckpt_path in args.checkpoint:
        model, cfg = load_model(ckpt_path, device)

        if args.tta > 0:
            probs, ids = predict_tta(
                model, test_dataset, device,
                n_views=args.tta, batch_size=args.batch_size
            )
        else:
            loader = DataLoader(
                test_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=4
            )
            probs, ids = predict_clean(model, loader, device)

        if ensemble_probs is None:
            ensemble_probs = probs
        else:
            ensemble_probs += probs

    # Average ensemble probabilities
    ensemble_probs /= len(args.checkpoint)

    # Generate submission
    generate_submission(ensemble_probs, ids, args.output)


if __name__ == "__main__":
    main()
