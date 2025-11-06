#!/usr/bin/env python3
"""
Tile the unified processed dataset (images + masks) into fixed-size patches
suitable for U-Net training.

Inputs:
- data/processed/annotations/manifest.json (produced by unify_annotations.py)

Outputs (default under data/processed/tiles/):
- images/<dataset>/<stem>_x{xs}_y{ys}.png
- masks/<dataset>/<stem>_x{xs}_y{ys}.png (single-channel uint16)
- annotations/tiles_manifest.json with entries per tile:
  {
    "id": "<stem>_x{xs}_y{ys}",
    "dataset": "cghd"|"hcd",
    "parent_id": "<original filename>",
    "image_path": "images/<dataset>/<tile_name>.png",
    "mask_path": "masks/<dataset>/<tile_name>.png",
    "x": xs, "y": ys, "w": tile_size, "h": tile_size,
    "split": "train"|"val"
  }

Notes:
- Windows are generated with stride; right/bottom edges are included by anchoring
  the last window at image boundary to ensure full coverage.
- Optionally skip tiles with low foreground content using --min-foreground-pct.
- Split is assigned per-parent image using a stable hash to keep all tiles from the
  same parent in the same split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"


def load_manifest(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("images", [])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(1, length - tile + 1), stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def tile_image_mask(img: Image.Image, mask: Image.Image, tile: int, stride: int) -> Iterable[Tuple[int, int, Image.Image, Image.Image]]:
    W, H = img.size
    # Ensure sizes match; resize mask if needed (nearest to preserve ids)
    if mask.size != img.size:
        mask = mask.resize(img.size, resample=Image.NEAREST)
    xs = compute_starts(W, tile, stride)
    ys = compute_starts(H, tile, stride)
    for y in ys:
        for x in xs:
            crop_box = (x, y, x + tile, y + tile)
            # If image smaller than tile, pad
            if W < tile or H < tile:
                # create padded canvases
                img_pad = Image.new("RGB", (tile, tile), (255, 255, 255))
                mask_pad = Image.new("I;16", (tile, tile), 0)
                img_pad.paste(img, (0, 0))
                mask_pad.paste(mask, (0, 0))
                yield x, y, img_pad, mask_pad
            else:
                yield x, y, img.crop(crop_box), mask.crop(crop_box)


def stable_split(parent_id: str, val_ratio: float) -> str:
    # Hash parent_id to float in [0,1) for stable split
    h = hashlib.md5(parent_id.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / float(0xFFFFFFFF)
    return "val" if v < val_ratio else "train"


def main():
    ap = argparse.ArgumentParser(description="Tile processed dataset for U-Net training")
    ap.add_argument("--manifest", default=str(PROC_DIR / "annotations/manifest.json"))
    ap.add_argument("--out-dir", default=str(PROC_DIR / "tiles"))
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--min-foreground-pct", type=float, default=0.0, help="Skip tiles with less than this fraction of non-background pixels (0-1). 0.0 keeps all.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of parent images to process (for quick tests)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    img_out_root = out_dir / "images"
    mask_out_root = out_dir / "masks"
    ann_out_dir = out_dir / "annotations"
    ensure_dir(img_out_root)
    ensure_dir(mask_out_root)
    ensure_dir(ann_out_dir)

    entries = load_manifest(manifest_path)

    tiles: List[Dict] = []
    processed = 0

    for e in entries:
        parent_id = e.get("id")
        dataset = e.get("dataset", "unknown")
        img_path = PROC_DIR / e["image_path"]
        mask_path = PROC_DIR / e["mask_path"]

        # Respect limit by parent images
        if args.limit and processed >= args.limit:
            break

        # Open image/mask
        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
        except Exception as ex:
            print(f"[WARN] Skipping {parent_id}: {ex}")
            continue

        split = stable_split(f"{dataset}:{parent_id}", args.val_ratio)

        # Prepare per-dataset dirs
        img_out_dir = img_out_root / dataset
        mask_out_dir = mask_out_root / dataset
        ensure_dir(img_out_dir)
        ensure_dir(mask_out_dir)

        for x, y, ti, tm in tile_image_mask(img, mask, args.tile_size, args.stride):
            # Evaluate foreground ratio (nonzero mask)
            tm_np = np.array(tm)
            if tm_np.ndim == 3:
                tm_np = tm_np[..., 0]
            total = tm_np.size
            fg = int((tm_np != 0).sum())
            fg_pct = fg / float(total) if total else 0.0
            if args.min_foreground_pct > 0.0 and fg_pct < args.min_foreground_pct:
                continue

            stem = Path(parent_id).stem
            tile_name = f"{stem}_x{x}_y{y}"
            img_tile_path = img_out_dir / f"{tile_name}.png"
            mask_tile_path = mask_out_dir / f"{tile_name}.png"

            # Save tiles
            ti.save(img_tile_path)
            # Ensure mask saved as single-channel, preserving class ids
            tm = tm.convert("I;16") if tm.mode != "I;16" else tm
            tm.save(mask_tile_path)

            tiles.append({
                "id": f"{tile_name}.png",
                "dataset": dataset,
                "parent_id": parent_id,
                "image_path": str(img_tile_path.relative_to(out_dir)).replace('\\\\', '/'),
                "mask_path": str(mask_tile_path.relative_to(out_dir)).replace('\\\\', '/'),
                "x": x, "y": y, "w": args.tile_size, "h": args.tile_size,
                "split": split
            })

        processed += 1

    # Write tiles manifest
    tiles_manifest = {"tiles": tiles, "meta": {
        "tile_size": args.tile_size,
        "stride": args.stride,
        "val_ratio": args.val_ratio,
        "min_foreground_pct": args.min_foreground_pct,
        "source_manifest": str(manifest_path.relative_to(ROOT)).replace('\\\\', '/')
    }}
    out_manifest = ann_out_dir / "tiles_manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(tiles_manifest, f, indent=2)

    print(f"Tiled {processed} parent images -> {len(tiles)} tiles")
    print(f"Tiles written under: {out_dir}")
    print(f"Tiles manifest: {out_manifest}")


if __name__ == "__main__":
    main()
