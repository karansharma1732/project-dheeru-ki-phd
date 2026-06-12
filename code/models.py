"""Model factory for the TB classifier.

DenseNet121 is the default per the implementation plan; ResNet50 is offered as an
ablation. Both expose ``model.features``-style access for Grad-CAM in gradcam.py.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(cfg: dict) -> nn.Module:
    arch = cfg["model"]["arch"].lower()
    pretrained = cfg["model"]["pretrained"]
    num_classes = cfg["model"]["num_classes"]

    if arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.densenet121(weights=weights)
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        return net

    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net

    raise ValueError(f"Unknown arch: {arch!r} (expected densenet121 | resnet50)")


def gradcam_target_layer(model: nn.Module, arch: str):
    """Return the last conv layer used as the Grad-CAM target."""
    arch = arch.lower()
    if arch == "densenet121":
        return model.features.norm5
    if arch == "resnet50":
        return model.layer4[-1]
    raise ValueError(f"Unknown arch: {arch!r}")
