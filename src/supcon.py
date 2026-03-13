"""
Supervised Contrastive Learning (SupCon) Pretraining
Trains a backbone (WRN, ResNet50, ConvNeXt) using SupCon Loss.
Unlike SimCLR, SupCon pulls all images of the *same class* together in the representation space,
making it extremely powerful for Long-Tail distributions.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import ShiftGuard10Dataset, CLASS_NAMES
from src.models.wideresnet import WideResNet
from src.models.torchvision_models import resnet50_cifar, convnext_tiny_cifar, effnet_v2_s_cifar
from src.utils import AverageMeter, seed_everything


class ContrastiveTransformations:
    """Generates two augmented views of the same image."""
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


def get_supcon_transforms(img_size=32):
    """Heavy augmentations typically used for Contrastive Learning on CIFAR."""
    s = 1.0
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss.
    Reference: https://arxiv.org/pdf/2004.11362.pdf
    """
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0] // 2
        labels = labels.contiguous().view(-1, 1) # [B, 1]
        
        # Since features are concatenated [view1(batch), view2(batch)]
        # The labels must also be appended
        labels = torch.cat([labels, labels], dim=0) # [2B, 1]
        
        # Create mask: 1 if same class, 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device) # [2B, 2B]

        # Compute cosine similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


class SupConNet(nn.Module):
    """Backbone + Projection Head for SupCon."""
    def __init__(self, backbone_model='wrn', projection_dim=128):
        super().__init__()
        if backbone_model == 'wrn':
            self.backbone = WideResNet(depth=28, widen_factor=10, num_classes=10, dropout=0.0)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_model == 'resnet50':
            self.backbone = resnet50_cifar(num_classes=10)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_model == 'convnext':
            self.backbone = convnext_tiny_cifar(num_classes=10)
            feat_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
        elif backbone_model == 'effnet':
            self.backbone = effnet_v2_s_cifar(num_classes=10)
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_model}")

        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        # We need normalized features for contrastive loss
        z = self.projection_head(features)
        return F.normalize(z, dim=1)


def train_supcon():
    parser = argparse.ArgumentParser(description="Supervised Contrastive Pretraining")
    parser.add_argument("--backbone", type=str, choices=["wrn", "resnet50", "convnext", "effnet"], default="wrn")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--balanced-sampling", action="store_true", help="Use balanced sampler during SupCon")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"SupCon Pretraining | Backbone: {args.backbone} | Device: {device}")

    transform = ContrastiveTransformations(get_supcon_transforms(), n_views=2)
    # Re-use dataset but without validation split (use all train data for SupCon)
    dataset = ShiftGuard10Dataset(
        root="shift-guard-10-robust-image-classification-challenge",
        split="train", transform=transform, val_ratio=0.0
    )

    sampler = None
    if args.balanced_sampling:
        sampler = dataset.get_sampler()
        print("  Using Balanced Sampling for SupCon batch composition.")

    if args.debug:
        dataset.image_ids = dataset.image_ids[:200]
        dataset.labels = dataset.labels[:200]
        args.epochs = 2

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=True
    )

    model = SupConNet(backbone_model=args.backbone, projection_dim=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = SupConLoss(temperature=args.temp)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()

        for idx, (images, targets) in enumerate(loader):
            img_i = images[0].to(device)
            img_j = images[1].to(device)
            targets = targets.to(device)

            z_i = model(img_i)
            z_j = model(img_j)

            features = torch.cat([z_i, z_j], dim=0)
            loss = criterion(features, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), img_i.size(0))

        scheduler.step()
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {loss_meter.avg:.4f} LR: {scheduler.get_last_lr()[0]:.4f}")

        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epochs:
            state_dict = model.backbone.state_dict()
            path = f"checkpoints/supcon_{args.backbone}_epoch{epoch+1}.pth"
            torch.save(state_dict, path)
            print(f"Saved backbone checkpoint to {path}")

if __name__ == "__main__":
    train_supcon()
