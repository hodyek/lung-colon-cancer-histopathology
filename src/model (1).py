"""
model.py
--------
Model definitions for all three architectures used in this project.

Functions:
    build_baseline_cnn    — Custom 3-block CNN trained from scratch.
    build_efficientnet_b0 — EfficientNet-B0 fine-tuned from ImageNet weights.
    build_resnet50        — ResNet-50 fine-tuned from ImageNet weights.
    build_model           — Dispatcher: returns a model by name string.
    count_parameters      — Count trainable parameters in a model.
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 5


# ── Baseline CNN ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Three-block CNN trained from scratch.

    Architecture:
        Block 1: 3 → 32 channels, 224×224 → 112×112
        Block 2: 32 → 64 channels, 112×112 → 56×56
        Block 3: 64 → 128 channels, 56×56 → 28×28
        Global average pooling → 128-dim feature vector
        Classifier: Linear(128 → num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_baseline_cnn(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Return a Baseline CNN with no pretrained weights."""
    return BaselineCNN(num_classes=num_classes)


# ── EfficientNet-B0 ───────────────────────────────────────────────────────────

def build_efficientnet_b0(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Return EfficientNet-B0 with pretrained ImageNet weights.

    The classifier head is replaced to output num_classes logits.

    Parameters
    ----------
    freeze_backbone : bool
        If True, freeze all feature layers and train only the head.
        Set to False for full fine-tuning (recommended after warmup phase).
    """
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


# ── ResNet-50 ─────────────────────────────────────────────────────────────────

def build_resnet50(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Return ResNet-50 with pretrained ImageNet weights.

    The fully connected layer is replaced to output num_classes logits.

    Parameters
    ----------
    freeze_backbone : bool
        If True, freeze all convolutional layers and train only the FC head.
        Set to False for full fine-tuning (recommended after warmup phase).
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ── Dispatcher ────────────────────────────────────────────────────────────────

def build_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Return a model by name string.

    Parameters
    ----------
    name : str
        One of 'baseline', 'efficientnet_b0', 'resnet50'.
    """
    name = name.lower().strip()
    if name == "baseline":
        return build_baseline_cnn(num_classes)
    elif name in ("efficientnet_b0", "efficientnet"):
        return build_efficientnet_b0(num_classes, freeze_backbone)
    elif name in ("resnet50", "resnet"):
        return build_resnet50(num_classes, freeze_backbone)
    else:
        raise ValueError(f"Unknown model name: '{name}'. "
                         f"Choose from 'baseline', 'efficientnet_b0', 'resnet50'.")


# ── Utility ───────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
