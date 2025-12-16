"""Train lightweight U-Net for pin heatmap regression on HCD data.

Data layout (per component type under data/hcd/Component Port Location Data/):
- <COMPONENT>/Input Images/            # 64x64 RGB input patches
- <COMPONENT>/XY Coordinates/          # text files with (y, x) per line

This script reads images and their pin coordinates, generates Gaussian heatmaps
as ground truth, and trains the model to regress probability maps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from backend.app.models.pin_unet import build_model, gaussian_heatmap, predict_coords


def parse_xy_file(fp: Path) -> List[Tuple[float, float]]:
    """Parse a coordinate txt file: each line 'y x' (float or int)."""
    coords: List[Tuple[float, float]] = []
    for line in fp.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            y = float(parts[0]); x = float(parts[1])
            coords.append((y, x))
    return coords


class PinHcdDataset(Dataset):
    def __init__(self, root: Path, components: List[str], size: int = 64, sigma: float = 1.8) -> None:
        self.items: List[Tuple[Path, Tuple[float, float]]] = []
        self.size = size
        self.sigma = sigma
        for comp in components:
            base = root / comp
            img_dir = base / "Input Images"
            xy_dir = base / "XY Coordinates"
            if not img_dir.exists() or not xy_dir.exists():
                continue
            # Match files by common stem: assume xy files map to images by index in name
            # We'll pair by reading all xy files, and for each coordinate, look up a same-stem image in Input Images
            for xy_fp in sorted(xy_dir.glob("*.txt")):
                stem = xy_fp.stem
                # Try common image extensions
                img_fp = None
                for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                    cand = img_dir / f"{stem}{ext}"
                    if cand.exists():
                        img_fp = cand
                        break
                if img_fp is None:
                    continue
                coords = parse_xy_file(xy_fp)
                # Some components may have multiple pins per image; we treat one target per sample.
                # If multiple coords, we can either create multiple samples or average heatmaps.
                # Here we create multiple samples (img reused).
                for (y, x) in coords:
                    self.items.append((img_fp, (y, x)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_fp, (y, x) = self.items[idx]
        img = Image.open(img_fp).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(img).astype(np.float32) / 255.0
        # Normalize by ImageNet stats for stability
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        # HWC->CHW
        img_t = torch.from_numpy(arr.transpose(2, 0, 1))
        # Clamp coordinates to valid range
        y = max(0.0, min(float(y), float(self.size - 1)))
        x = max(0.0, min(float(x), float(self.size - 1)))
        coord_t = torch.tensor([y, x], dtype=torch.float32)
        heat_t = gaussian_heatmap(coord_t.view(1, 2), size=self.size, sigma=self.sigma)[0]
        return img_t, heat_t, {"image_path": str(img_fp), "coord": (y, x)}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    root = Path(args.root)
    components = [c.strip() for c in args.components.split(",") if c.strip()]

    ds = PinHcdDataset(root, components=components, size=args.size, sigma=args.sigma)
    n = len(ds)
    if n == 0:
        raise RuntimeError(f"No samples found under {root} for components {components}")

    # split train/val
    val_ratio = args.val_ratio
    val_n = max(1, int(n * val_ratio))
    train_n = n - val_n
    ds_train, ds_val = torch.utils.data.random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(42))

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    history = []

    print(f"Using device: {device.type}; samples: {n}; train/val={train_n}/{val_n}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for img_t, heat_t, _ in loader_train:
            img_t = img_t.to(device)
            heat_t = heat_t.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(img_t)
            loss = criterion(pred, heat_t)
            loss.backward()
            optimizer.step()
            running += loss.item() * img_t.size(0)
            count += img_t.size(0)
        train_loss = running / max(1, count)

        # validation: BCE + simple localization error (pixel L2)
        model.eval()
        val_loss = 0.0
        pix_err = 0.0
        vcount = 0
        with torch.no_grad():
            for img_t, heat_t, meta in loader_val:
                img_t = img_t.to(device)
                heat_t = heat_t.to(device)
                pred = model(img_t)
                loss = criterion(pred, heat_t)
                val_loss += loss.item() * img_t.size(0)
                # localization error
                pred_xy = predict_coords(pred.cpu())
                true_xy = torch.stack([torch.tensor(m["coord"][0]) for m in meta], dim=0)
                true_xx = torch.stack([torch.tensor(m["coord"][1]) for m in meta], dim=0)
                true_xy = torch.stack([true_xy, true_xx], dim=1)
                pix_err += torch.norm(pred_xy - true_xy, dim=1).sum().item()
                vcount += img_t.size(0)
        val_loss /= max(1, vcount)
        avg_pix_err = pix_err / max(1, vcount)

        # save best
        ckpt_path = out_dir / "checkpoint.pt"
        if best_val < 0 or val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "avg_pix_err": avg_pix_err,
            }, ckpt_path)

        history.append({
            "epoch": epoch,
            "train_bce": train_loss,
            "val_bce": val_loss,
            "val_avg_pixel_error": avg_pix_err,
        })
        print(f"Epoch {epoch}/{args.epochs} train_bce={train_loss:.4f} val_bce={val_loss:.4f} val_pix_err={avg_pix_err:.2f}")

    # write metrics
    (out_dir / "metrics.json").write_text(json.dumps({
        "best_val_bce": best_val,
        "history": history,
        "components": components,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pin heatmap U-Net on HCD components")
    p.add_argument("--root", type=str, default="data/hcd/Component Port Location Data")
    p.add_argument("--components", type=str, default="Resistor,Capacitor,Diode")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--sigma", type=float, default=1.8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="runs/pin_heatmap")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
