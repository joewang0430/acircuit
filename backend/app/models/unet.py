from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch not available in environment") from e

try:  # segmentation_models_pytorch is optional
    import segmentation_models_pytorch as smp  # type: ignore
except Exception as e:  # pragma: no cover
    smp = None  # type: ignore


class UNetModel(nn.Module):
    """Wrapper around segmentation_models_pytorch Unet with ResNet encoder.

    Output logits shape: B x num_classes x H x W
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        num_classes: int = 12,
        channels: int = 3,
    ) -> None:
        super().__init__()
        if smp is None:
            raise RuntimeError("segmentation_models_pytorch not installed")
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=channels,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x):  # type: ignore
        return self.model(x)


def build_model(num_classes: int = 12) -> nn.Module:
    return UNetModel(num_classes=num_classes)
