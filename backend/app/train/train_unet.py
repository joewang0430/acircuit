"""Minimal training script for the 12-class U-Net on tiles.

Usage (after installing dependencies):
  python backend/app/train/train_unet.py \
    --manifest data/processed/tiles/annotations/tiles_manifest.json \
    --epochs 5 --batch-size 16 --lr 3e-4 --out-dir runs/unet_baseline
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple

import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch not installed; activate venv and install torch.") from e

try:
    from backend.app.datasets.tiles_dataset import TilesDataset, collate_fn
    from backend.app.models.unet import build_model
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Local imports failed: {e}") from e


def focal_ce_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, weight=None) -> torch.Tensor:
    """Focal loss built on top of softmax cross entropy."""
    ce = nn.functional.cross_entropy(logits, targets, weight=weight, reduction="none")
    # pt = exp(-ce)
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    return focal.mean()


def compute_class_weights(manifest_path: Path, num_classes: int = 12) -> torch.Tensor:
    # Simple frequency-based weights (inverse sqrt of pixel count proportion)
    # This is a placeholder; a more accurate version would read masks; here we just keep uniform.
    return torch.ones(num_classes, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Tuple[float, List[float]]:
    model.eval()
    iou_sums = torch.zeros(num_classes, dtype=torch.float64)
    denom = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for c in range(num_classes):
                pred_c = preds == c
                mask_c = masks == c
                inter = (pred_c & mask_c).sum().item()
                union = (pred_c | mask_c).sum().item()
                if union > 0:
                    iou_sums[c] += inter / union
                    denom[c] += 1
    per_class_iou = [(iou_sums[c] / denom[c]).item() if denom[c] > 0 else 0.0 for c in range(num_classes)]
    mean_iou = sum(per_class_iou) / num_classes
    return mean_iou, per_class_iou


def train(args: argparse.Namespace) -> None:
    # Choose device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    device = torch.device(device_type)
    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    ds_train = TilesDataset(manifest, split="train", augment=True)
    ds_val = TilesDataset(manifest, split="val", augment=False)
    ds_test = TilesDataset(manifest, split="test", augment=False)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    num_classes = 12
    model = build_model(num_classes=num_classes).to(device)

    class_weights = compute_class_weights(manifest, num_classes=num_classes)
    class_weights = class_weights.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # AMP: enable GradScaler only for CUDA; use torch.autocast for MPS if requested
    use_amp = bool(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_type == "cuda")

    best_val = -math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        step = 0
        for images, masks, _ in loader_train:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Autocast context per device
            if use_amp and device_type == "cuda":
                amp_ctx = torch.cuda.amp.autocast()
            elif use_amp and device_type == "mps":
                amp_ctx = torch.autocast(device_type="mps", dtype=torch.float16)
            else:
                from contextlib import nullcontext
                amp_ctx = nullcontext()

            with amp_ctx:
                logits = model(images)
                if args.focal:
                    loss = focal_ce_loss(logits, masks, gamma=2.0, weight=class_weights)
                else:
                    loss = nn.functional.cross_entropy(logits, masks, weight=class_weights)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            step += 1
            if args.train_steps and step >= args.train_steps:
                break

        train_loss = running_loss / len(ds_train)
        # Optionally limit val steps for quick smoke tests
        if args.val_steps:
            # run a limited evaluation loop
            model.eval()
            iou_sums = torch.zeros(num_classes, dtype=torch.float64)
            denom = torch.zeros(num_classes, dtype=torch.float64)
            with torch.no_grad():
                vstep = 0
                for images, masks, _ in loader_val:
                    images = images.to(device)
                    masks = masks.to(device)
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1)
                    for c in range(num_classes):
                        pred_c = preds == c
                        mask_c = masks == c
                        inter = (pred_c & mask_c).sum().item()
                        union = (pred_c | mask_c).sum().item()
                        if union > 0:
                            iou_sums[c] += inter / union
                            denom[c] += 1
                    vstep += 1
                    if vstep >= args.val_steps:
                        break
            val_per_class = [(iou_sums[c] / denom[c]).item() if denom[c] > 0 else 0.0 for c in range(num_classes)]
            val_miou = sum(val_per_class) / num_classes
        else:
            val_miou, val_per_class = evaluate(model, loader_val, device, num_classes)

        # Save checkpoint if improved
        ckpt_path = out_dir / "checkpoint.pt"
        if val_miou > best_val:
            best_val = val_miou
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_miou": val_miou,
            }, ckpt_path)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_miou": val_miou,
            "val_per_class": val_per_class,
        })
        print(f"Epoch {epoch}/{args.epochs} loss={train_loss:.4f} val_mIoU={val_miou:.4f}")

    # Final test evaluation with best model
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
    test_miou, test_per_class = evaluate(model, loader_test, device, num_classes)
    (out_dir / "metrics.json").write_text(json.dumps({
        "history": history,
        "best_val_miou": best_val,
        "test_miou": test_miou,
        "test_per_class": test_per_class,
    }, indent=2))
    print(f"Test mIoU={test_miou:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 12-class U-Net on tiles")
    p.add_argument("--manifest", type=str, default="data/processed/tiles/annotations/tiles_manifest.json")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", type=str, default="runs/unet_baseline")
    p.add_argument("--focal", action="store_true", help="Use focal loss instead of plain CE")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest across platforms)")
    p.add_argument("--train-steps", dest="train_steps", type=int, default=0, help="Max training steps per epoch (0=full)")
    p.add_argument("--val-steps", dest="val_steps", type=int, default=0, help="Max validation steps (0=full)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
