#!/usr/bin/env python3
"""
Generate visualizations for processed masks:
- overlay: colored mask blended over the original image
- color: standalone colorized mask (no photo), for maximum visibility

Inputs:
- data/processed/annotations/manifest.json (lists image/mask/ann paths and dataset)

Outputs:
- data/processed/preview/<dataset>/<stem>_overlay.png  (overlay mode)
- data/processed/preview/<dataset>/<stem>_color.png    (colorized mask mode)
- data/processed/preview/legend.png                    (optional legend)

Notes:
- Uses a saturated, high-contrast color map for known class ids.
- Background (0) is left unpainted for overlay and white for colorized mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def color_map() -> Dict[int, np.ndarray]:
    # Bold, saturated colors for visibility on white backgrounds.
    # 0 is background (not used for overlay; white for colorized mask)
    cmap = {
        0: np.array([0, 0, 0], dtype=np.uint8),            # background placeholder
        1: np.array([255, 212, 0], dtype=np.uint8),        # Inductor -> yellow
        2: np.array([255, 59, 48], dtype=np.uint8),        # Diode -> red
        3: np.array([255, 0, 255], dtype=np.uint8),        # Zener -> magenta
        4: np.array([255, 149, 0], dtype=np.uint8),        # Resistor -> orange
        5: np.array([0, 122, 255], dtype=np.uint8),        # Capacitor -> blue
        6: np.array([0, 199, 190], dtype=np.uint8),        # Wire Crossover -> cyan
        7: np.array([52, 199, 89], dtype=np.uint8),        # V-DC -> green
        8: np.array([175, 82, 222], dtype=np.uint8),       # V-AC -> purple
        9: np.array([47, 79, 79], dtype=np.uint8),         # GND -> dark slate gray
        10: np.array([160, 82, 45], dtype=np.uint8),       # I-DC -> brown
        11: np.array([0, 0, 128], dtype=np.uint8),         # I-AC -> navy
    }
    return cmap


def load_label_names() -> Dict[int, str]:
    # Try to load human-readable names from label_map.json; fall back to defaults
    default_names = {
        1: "Inductor",
        2: "Diode",
        3: "Zener",
        4: "Resistor",
        5: "Capacitor",
        6: "Wire Crossover",
        7: "V-DC",
        8: "V-AC",
        9: "GND",
        10: "I-DC",
        11: "I-AC",
    }
    lm_path = ROOT / "backend/app/core/label_map.json"
    try:
        with open(lm_path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        # Support either array-form {"classes": [...]} or dict keyed by ids
        by_id: Dict[int, str] = {}
        classes = lm.get("classes") or lm.get("labels")
        if isinstance(classes, list):
            for c in classes:
                cid = c.get("id")
                # Prefer display, then canonical, then name
                name = c.get("display") or c.get("canonical") or c.get("name") or c.get("label")
                if isinstance(cid, int) and isinstance(name, str):
                    by_id[cid] = name
        elif isinstance(lm, dict):
            # dict style: {"0": {...}, "1": {...}}
            for k, v in lm.items():
                try:
                    cid = int(v.get("id", k))
                except Exception:
                    continue
                name = v.get("display") or v.get("canonical") or v.get("name") or v.get("label")
                if isinstance(name, str):
                    by_id[cid] = name
        return {**default_names, **by_id}
    except Exception:
        return default_names


def blend_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend colored regions for each class id onto the image.
    img: HxWx3 uint8, mask: HxW int (class ids)
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    H, W = mask.shape
    assert img.shape[0] == H and img.shape[1] == W

    out = img.copy().astype(np.float32)
    cmap = color_map()
    ids = np.unique(mask)
    for cid in ids:
        if cid == 0:
            continue
        color = cmap.get(int(cid), np.array([255, 255, 255], dtype=np.uint8))
        m = (mask == cid)
        if not m.any():
            continue
    # broadcast color and blend via np.where to avoid boolean indexing shape issues
        color_b = color.reshape(1, 1, 3).astype(np.float32)
        m3 = m[:, :, None]
        blended = (1.0 - alpha) * out + alpha * color_b
        out = np.where(m3, blended, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray, bg_color=(255, 255, 255)) -> np.ndarray:
    """Create a standalone colorized mask RGB image.
    Background (0) is painted with bg_color (default white).
    """
    cmap = color_map()
    H, W = mask.shape
    out = np.full((H, W, 3), np.array(bg_color, dtype=np.uint8), dtype=np.uint8)
    ids = np.unique(mask)
    for cid in ids:
        if cid == 0:
            continue
        color = cmap.get(int(cid), np.array([0, 0, 0], dtype=np.uint8))
        m = (mask == cid)
        if not m.any():
            continue
        out[m] = color
    return out


def process_one(entry: Dict, alpha: float, mode: str) -> List[Path]:
    img_path = PROC_DIR / entry["image_path"]
    mask_path = PROC_DIR / entry["mask_path"]
    dataset = entry.get("dataset", "unknown")
    stem = Path(entry["id"]).stem

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)
    # If dimensions differ, resize mask to image size using nearest-neighbor to preserve class ids
    if mask.size != img.size:
        mask = mask.resize(img.size, resample=Image.NEAREST)
    img_np = np.array(img, dtype=np.uint8)
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    if not np.issubdtype(mask_np.dtype, np.integer):
        mask_np = mask_np.astype(np.int32)

    out_dir = PROC_DIR / "preview" / dataset
    ensure_dir(out_dir)
    made: List[Path] = []

    if mode in ("overlay", "both"):
        over = blend_overlay(img_np, mask_np, alpha=alpha)
        out_path = out_dir / f"{stem}_overlay.png"
        Image.fromarray(over).save(out_path)
        made.append(out_path)

    if mode in ("color", "both"):
        col = colorize_mask(mask_np)
        out_path = out_dir / f"{stem}_color.png"
        Image.fromarray(col).save(out_path)
        made.append(out_path)

    return made


def draw_legend(out_path: Path) -> None:
    names = load_label_names()
    cmap = color_map()
    # Build a simple vertical legend
    from PIL import ImageDraw, ImageFont

    items = [cid for cid in sorted(cmap.keys()) if cid != 0]
    sw, sh = 36, 24  # swatch width, row height
    pad = 10
    width = 500
    height = pad * 2 + sh * len(items)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    y = pad
    for cid in items:
        color = tuple(int(x) for x in cmap[cid])
        draw.rectangle([pad, y, pad + sw, y + sh - 4], fill=color, outline=(0, 0, 0))
        label = f"{cid}: {names.get(cid, 'class')}"
        draw.text((pad + sw + 10, y), label, fill=(0, 0, 0), font=font)
        y += sh

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description="Visualize processed masks as overlays and/or colorized images")
    ap.add_argument("--manifest", default=str(PROC_DIR / "annotations/manifest.json"))
    ap.add_argument("--ids", nargs="*", help="Optional list of image ids (filenames) to render")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--mode", choices=["overlay", "color", "both"], default="both")
    ap.add_argument("--legend", action="store_true", help="Also write a legend image to preview directory")
    args = ap.parse_args()

    entries = load_manifest(Path(args.manifest))
    targets = set(args.ids) if args.ids else None

    made: List[Path] = []
    for e in entries:
        if targets and e.get("id") not in targets:
            continue
        paths = process_one(e, args.alpha, args.mode)
        made.extend(paths)

    if args.legend:
        legend_path = PROC_DIR / "preview/legend.png"
        draw_legend(legend_path)
        made.append(legend_path)

    if not made:
        print("No previews generated (no matching ids?).")
    else:
        print("Generated previews:")
        for p in made:
            print(str(p))


if __name__ == "__main__":
    main()
