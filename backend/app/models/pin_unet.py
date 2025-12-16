from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_transpose: bool = False) -> None:
        super().__init__()
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            in_after_up = out_ch
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            in_after_up = in_ch
        self.conv = conv_block(in_after_up + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            # pad/crop if needed due to odd dims; here we center-crop skip to x size
            sh = skip.shape[-2:]
            xh = x.shape[-2:]
            dh = sh[0] - xh[0]
            dw = sh[1] - xh[1]
            if dh != 0 or dw != 0:
                top = max(dh // 2, 0)
                left = max(dw // 2, 0)
                skip = skip[:, :, top:top + xh[0], left:left + xh[1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PinHeatmapUNet(nn.Module):
    """Lightweight U-Net for pin heatmap regression.

    Input: (B, 3, 64, 64)
    Output: (B, 1, 64, 64) probability map in [0,1]
    """

    def __init__(self, use_transpose: bool = False) -> None:
        super().__init__()
        # Encoder
        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck (at 8x8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up3 = UpBlock(256, 256, 128, use_transpose)
        self.up2 = UpBlock(128, 128, 64, use_transpose)
        self.up1 = UpBlock(64, 64, 64, use_transpose)

        # Final head
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)          # 64x64 -> 64 ch
        p1 = self.pool1(s1)        # 32x32
        s2 = self.enc2(p1)         # 32x32 -> 128 ch
        p2 = self.pool2(s2)        # 16x16
        s3 = self.enc3(p2)         # 16x16 -> 256 ch
        p3 = self.pool3(s3)        # 8x8

        # Bottleneck
        b = self.bottleneck(p3)    # 8x8 -> 256 ch

        # Decoder with skip connections
        d3 = self.up3(b, s3)       # 16x16 -> 128 ch
        d2 = self.up2(d3, s2)      # 32x32 -> 64 ch
        d1 = self.up1(d2, s1)      # 64x64 -> 64 ch

        logits = self.head(d1)     # (B,1,64,64)
        return torch.sigmoid(logits)


def build_model() -> nn.Module:
    return PinHeatmapUNet(use_transpose=False)


def gaussian_heatmap(coords: torch.Tensor, size: int = 64, sigma: float = 1.5) -> torch.Tensor:
    """Generate Gaussian heatmaps for batch of coordinates.

    coords: (B, 2) with (y, x) in pixel units [0, size)
    returns: (B, 1, size, size)
    """
    B = coords.shape[0]
    yy, xx = torch.meshgrid(torch.arange(size, device=coords.device), torch.arange(size, device=coords.device), indexing='ij')
    yy = yy.float(); xx = xx.float()
    cy = coords[:, 0].view(B, 1, 1)
    cx = coords[:, 1].view(B, 1, 1)
    dist2 = (yy.view(1, size, size) - cy) ** 2 + (xx.view(1, size, size) - cx) ** 2
    heat = torch.exp(-dist2 / (2 * (sigma ** 2)))
    return heat.unsqueeze(1)


def predict_coords(prob_map: torch.Tensor) -> torch.Tensor:
    """Argmax to (y,x) coordinates.

    prob_map: (B, 1, H, W) in [0,1]
    returns: (B, 2) (y, x)
    """
    B, _, H, W = prob_map.shape
    flat_idx = prob_map.view(B, -1).argmax(dim=1)
    y = (flat_idx // W).float()
    x = (flat_idx % W).float()
    return torch.stack([y, x], dim=1)
