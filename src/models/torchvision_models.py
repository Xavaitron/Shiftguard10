"""
Torchvision Models adapted for 32x32 (CIFAR-style) resolution.
Standard torchvision models are designed for 224x224 ImageNet inputs.
We swap out the stems (conv1/maxpool) to prevent aggressive downsampling.
These are trained 100% from scratch.
"""

import torch.nn as nn
import torchvision.models as models

def resnet50_cifar(num_classes=10):
    """ResNet-50 adapted for 32x32 images."""
    # weights=None is the default, ensuring train-from-scratch
    model = models.resnet50(num_classes=num_classes)
    
    # Replace the 7x7 stride-2 conv and the 3x3 maxpool with a simple 3x3 stride-1 conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    return model

def convnext_tiny_cifar(num_classes=10):
    """ConvNeXt-Tiny adapted for 32x32 images."""
    model = models.convnext_tiny(num_classes=num_classes)
    
    # ConvNeXt original stem downsamples by 4x. We change it to 1x for 32x32.
    # The stem is usually inside model.features[0] which is a Conv2dNormActivation
    # model.features[0][0] is the Conv2d
    in_channels = 3
    out_channels = model.features[0][0].out_channels
    model.features[0][0] = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    return model

def effnet_v2_s_cifar(num_classes=10):
    """EfficientNet-V2-Small adapted for 32x32 images."""
    model = models.efficientnet_v2_s(num_classes=num_classes)
    
    # Replace stem
    out_channels = model.features[0][0].out_channels
    model.features[0][0] = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    
    return model
