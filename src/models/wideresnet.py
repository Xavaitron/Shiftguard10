"""
WideResNet for 32x32 image classification (CIFAR-style).
WRN-28-10: 28 layers deep, widening factor 10.
Standard robust baseline for small-image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """WideResNet basic block with optional dropout."""
    def __init__(self, in_planes, out_planes, stride, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """
    WideResNet for 32×32 images.

    Default: WRN-28-10 (~36M params)
    - No initial 7×7 conv or maxpool (those are for ImageNet-scale)
    - Uses 3×3 conv stem appropriate for CIFAR-sized inputs
    """
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n = (depth - 4) // 6  # blocks per group

        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.group1 = self._make_group(channels[0], channels[1], n, stride=1, dropout=dropout)
        self.group2 = self._make_group(channels[1], channels[2], n, stride=2, dropout=dropout)
        self.group3 = self._make_group(channels[2], channels[3], n, stride=2, dropout=dropout)

        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)

        self._init_weights()

    def _make_group(self, in_planes, out_planes, num_blocks, stride, dropout):
        layers = [BasicBlock(in_planes, out_planes, stride, dropout)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_planes, out_planes, 1, dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)                      # (B, 16, 32, 32)
        out = self.group1(out)                   # (B, 160, 32, 32)
        out = self.group2(out)                   # (B, 320, 16, 16)
        out = self.group3(out)                   # (B, 640, 8, 8)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)      # (B, 640, 1, 1)
        out = out.flatten(1)                     # (B, 640)
        return self.fc(out)                      # (B, num_classes)


def wrn_28_10(num_classes=10, dropout=0.3):
    """WideResNet-28-10."""
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropout=dropout)
