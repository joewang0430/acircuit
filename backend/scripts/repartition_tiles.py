"""
Repartition an existing tiles manifest into train/val/test using a stable
per-parent hash so all tiles from the same parent image land in the same split.

Inputs
- tiles_manifest.json produced by tile_dataset.py with structure:
  {
    "tiles": [
      {"id": ..., "dataset": ..., "parent_id": ..., "image_path": ..., "mask_path": ..., "x": ..., "y": ..., "w": ..., "h": ..., "split": "train"|"val"},
      ...
    ],
    "meta": {"tile_size": ..., "stride": ..., "val_ratio": ..., "min_foreground_pct": ..., "source_manifest": ...}
  }

Outputs
- Updates the same manifest in place (optionally writes a backup first) and adds
  meta.test_ratio, updates meta.val_ratio to the provided value.

Usage example
  python backend/scripts/repartition_tiles.py \
    --manifest data/processed/tiles/annotations/tiles_manifest.json \
    --val-ratio 0.10 --test-ratio 0.05 --backup
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple


def _hash01(key: str) -> float:
    """Stable hash of a string into [0, 1)."""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    v = int(h, 16) / float(2 ** 128)
    return v


def stable_split(parent_key: str, val_ratio: float, test_ratio: float) -> str:
    """Assign split based on stable hash and requested ratios.

    - test: v < test_ratio
    - val:  test_ratio <= v < test_ratio + val_ratio
    - train: rest
    """
    v = _hash01(parent_key)
    if v < test_ratio:
        return "test"
    elif v < test_ratio + val_ratio:
        return "val"
    else:
        return "train"


def repartition_manifest(
    manifest_path: Path,
    val_ratio: float,
    test_ratio: float,
    make_backup: bool = True,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """Repartition tiles in-place and return summary counts.

    Returns
    - totals: dict with keys train/val/test/total
    - per_ds: dict of dataset -> {train, val, test, total}
    """
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())
    tiles = data.get("tiles", [])

    if make_backup:
        backup = manifest_path.with_name(manifest_path.stem + ".backup.json")
        backup.write_text(json.dumps(data, indent=2))

    totals = {"train": 0, "val": 0, "test": 0, "total": 0}
    per_ds: Dict[str, Dict[str, int]] = {}

    for t in tiles:
        ds = t.get("dataset", "")
        parent_id = t.get("parent_id", "")
        key = f"{ds}:{parent_id}"
        sp = stable_split(key, val_ratio=val_ratio, test_ratio=test_ratio)
        t["split"] = sp

        # counts
        totals[sp] += 1
        totals["total"] += 1
        if ds not in per_ds:
            per_ds[ds] = {"train": 0, "val": 0, "test": 0, "total": 0}
        per_ds[ds][sp] += 1
        per_ds[ds]["total"] += 1

    # update meta
    meta: Dict[str, Any] = data.setdefault("meta", {})
    meta["val_ratio"] = float(val_ratio)
    meta["test_ratio"] = float(test_ratio)
    meta["split_strategy"] = "per-parent-md5"

    # write back
    manifest_path.write_text(json.dumps(data, indent=2))

    return totals, per_ds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repartition tiles manifest into train/val/test")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/tiles/annotations/tiles_manifest.json"),
        help="Path to tiles_manifest.json",
    )
    p.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio (default 0.10)")
    p.add_argument("--test-ratio", type=float, default=0.05, help="Test ratio (default 0.05)")
    p.add_argument("--no-backup", action="store_true", help="Do not write a backup copy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    totals, per_ds = repartition_manifest(
        manifest_path=args.manifest,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        make_backup=not args.no_backup,
    )
    print("TOTALS:", totals)
    print("PER_DATASET:")
    for ds, c in sorted(per_ds.items()):
        print(f"  {ds}: {c}")


if __name__ == "__main__":
    main()
