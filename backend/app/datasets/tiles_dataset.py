import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import albumentations as A  # type: ignore
except Exception:  # pragma: no cover - optional
    A = None  # type: ignore


class TilesDataset:
    """
    Dataset reading tiles based on tiles_manifest.json.

    Returns (image, mask, meta) per __getitem__:
    - image: np.ndarray HxWx3, float32 in [0,1]
    - mask: np.ndarray HxW, int64 with values 0..11
    - meta: dict with id, dataset, parent_id, split, image_path, mask_path
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        augment: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        include_datasets: Optional[List[str]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        data = json.loads(self.manifest_path.read_text())
        tiles: List[Dict[str, Any]] = data.get("tiles", [])
        self.root = self.manifest_path.parent.parent  # .../tiles
        split = split.lower()
        include = set([s.lower() for s in include_datasets]) if include_datasets else None
        self.entries: List[Dict[str, Any]] = []
        for t in tiles:
            if t.get("split", "train") != split:
                continue
            if include and t.get("dataset", "").lower() not in include:
                continue
            self.entries.append(t)

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.augment = augment and (A is not None)
        if self.augment:
            # light, geometry-only safe augmentations for line drawings
            self.au = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),
                ],
            )
        else:
            self.au = None

    def __len__(self) -> int:
        return len(self.entries)

    def _open_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr

    def _open_mask(self, path: Path) -> np.ndarray:
        # mask stored as PNG with small integer ids 0..11
        m = Image.open(path)
        arr = np.asarray(m)
        # Ensure int64 for loss functions later
        return arr.astype(np.int64)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        t = self.entries[idx]
        img_path = self.root / t["image_path"]
        msk_path = self.root / t["mask_path"]
        image = self._open_image(img_path)
        mask = self._open_mask(msk_path)

        if self.au is not None:
            aug = self.au(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        # Normalize
        image = (image - self.mean) / self.std

        meta = {
            "id": t.get("id"),
            "dataset": t.get("dataset"),
            "parent_id": t.get("parent_id"),
            "split": t.get("split"),
            "image_path": str(img_path),
            "mask_path": str(msk_path),
        }

        return image, mask, meta


def collate_fn(batch):
    """Collate that converts numpy to torch tensors if torch is available."""
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        return batch

    images = []
    masks = []
    metas = []
    for img, msk, meta in batch:
        # HWC -> CHW
        img_t = torch.from_numpy(img.transpose(2, 0, 1))  # float32
        msk_t = torch.from_numpy(msk)  # int64
        images.append(img_t)
        masks.append(msk_t)
        metas.append(meta)

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks, metas
