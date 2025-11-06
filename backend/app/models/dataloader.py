"""
Data loader template for UNet training (online tile generation).

Features (draft):
- Read `data/processed/annotations/manifest.json` which lists processed images, masks, ann paths.
- For each image, compute sliding-window tile coordinates (tile_size=512, stride=256 by default).
- Include an object in a tile if IoU(bbox, tile) >= min_iou (default 0.5).
- Remap bboxes to tile coordinate system and provide per-tile metadata.
- Basic augmentations (random flip/rotate, color jitter) applied identically to image and mask.
- Returns (image_tensor, mask_tensor, meta_dict) where mask_tensor is single-channel class-index ints.

Usage (high level):
  dataset = ProcessedTileDataset(manifest_path, tile_size=512, stride=256)
  loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

Manifest format (example):
{
  "images": [
    {"id": "C-1_D1_P3.jpeg", "image_path":"data/processed/images/cghd/C-1_D1_P3.jpeg", "mask_path":"data/processed/masks/cghd/C-1_D1_P3.png", "ann_path":"data/processed/annotations/cghd/C-1_D1_P3.json", "split":"train"},
    ...
  ]
}

Note: this is a template and intentionally minimal. Replace/extend augmentations with albumentations or torchvision as desired.
"""

from typing import List, Tuple, Dict, Optional
import json
import math
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def bbox_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """IoU for boxes in (xmin,ymin,xmax,ymax)."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_tile_coords(H: int, W: int, tile_size: int, stride: int) -> List[Tuple[int, int, int, int]]:
    coords = []
    y = 0
    while y < H:
        x = 0
        y2 = min(y + tile_size, H)
        h = y2 - y
        while x < W:
            x2 = min(x + tile_size, W)
            w = x2 - x
            coords.append((x, y, x2, y2))
            if x2 == W:
                break
            x += stride
        if y2 == H:
            break
        y += stride
    return coords


class ProcessedTileDataset(Dataset):
    """PyTorch Dataset that yields tiles from processed images and masks.

    Each item is a tuple: (image_tensor, mask_tensor, meta)
      - image_tensor: FloatTensor (C x H x W), values in [0,1]
      - mask_tensor: LongTensor (H x W) with class indices (0..K-1)
      - meta: dict with keys: image_id, dataset, orig_size, tile_coord (x1,y1,x2,y2), objects (remapped bboxes)
    """

    def __init__(self,
                 manifest_path: str,
                 tile_size: int = 512,
                 stride: int = 256,
                 min_iou: float = 0.5,
                 min_area: int = 4,
                 include_empty: bool = True,
                 transforms: Optional[object] = None,
                 precompute_tiles: bool = True):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        self.items = manifest.get('images', [])
        self.tile_size = tile_size
        self.stride = stride
        self.min_iou = min_iou
        self.min_area = min_area
        self.include_empty = include_empty
        self.transforms = transforms
        self.index_map = []  # list of (item_idx, tile_coord)

        if precompute_tiles:
            for idx, it in enumerate(self.items):
                img_path = it['image_path']
                if not os.path.exists(img_path):
                    continue
                with Image.open(img_path) as im:
                    W, H = im.size
                coords = compute_tile_coords(H, W, tile_size, stride)
                for coord in coords:
                    self.index_map.append((idx, coord))
        else:
            # lazy: only store items; __getitem__ will compute tiles on demand
            for idx, it in enumerate(self.items):
                self.index_map.append((idx, None))

    def __len__(self):
        return len(self.index_map)

    def _load_image_mask_ann(self, item: Dict):
        img = Image.open(item['image_path']).convert('RGB')
        mask = Image.open(item['mask_path']).convert('L')
        with open(item['ann_path'], 'r', encoding='utf-8') as f:
            ann = json.load(f)
        return img, mask, ann

    def __getitem__(self, index: int):
        item_idx, coord = self.index_map[index]
        item = self.items[item_idx]
        img, mask, ann = self._load_image_mask_ann(item)
        W, H = img.size
        if coord is None:
            # compute coords lazily and pick first tile (not typical)
            coords = compute_tile_coords(H, W, self.tile_size, self.stride)
            coord = coords[0]
        x1, y1, x2, y2 = coord

        # crop
        img_patch = img.crop((x1, y1, x2, y2))
        mask_patch = mask.crop((x1, y1, x2, y2))

        # objects inclusion / remap
        objects = ann.get('objects', [])
        remapped_objs = []
        for obj in objects:
            # expect bbox [xmin,ymin,xmax,ymax]
            bbox = obj.get('bbox')
            if bbox is None:
                continue
            bx1, by1, bx2, by2 = bbox
            iou = bbox_iou((bx1, by1, bx2, by2), (x1, y1, x2, y2))
            if iou + 1e-8 >= self.min_iou:
                # remap to tile coords
                nx1 = max(0, bx1 - x1)
                ny1 = max(0, by1 - y1)
                nx2 = min(x2 - x1, bx2 - x1)
                ny2 = min(y2 - y1, by2 - y1)
                area = max(0, nx2 - nx1) * max(0, ny2 - ny1)
                if area >= self.min_area:
                    new_obj = obj.copy()
                    new_obj['bbox'] = [int(nx1), int(ny1), int(nx2), int(ny2)]
                    remapped_objs.append(new_obj)

        # skip tile if no objects and include_empty is False
        if (not self.include_empty) and len(remapped_objs) == 0:
            # find next valid tile (simple linear probe)
            next_idx = index + 1
            if next_idx < len(self):
                return self.__getitem__(next_idx)
            # fall back to returning empty

        # To tensor
        img_t = TF.to_tensor(img_patch)  # float [0,1], CxHxW
        mask_np = np.array(mask_patch, dtype=np.int64)
        mask_t = torch.from_numpy(mask_np).long()

        # simple random augmentations (basic): flip, rotate 90 deg multiple
        if self.transforms is None:
            # apply simple random flip
            if torch.rand(1).item() > 0.5:
                img_t = TF.hflip(img_t)
                mask_t = TF.hflip(mask_t.unsqueeze(0)).squeeze(0)
            if torch.rand(1).item() > 0.5:
                img_t = TF.vflip(img_t)
                mask_t = TF.vflip(mask_t.unsqueeze(0)).squeeze(0)
        else:
            # user-provided transforms should accept (image, mask) and return same
            img_t, mask_t = self.transforms(img_t, mask_t)

        meta = {
            'image_id': item.get('id'),
            'dataset': item.get('dataset', None),
            'orig_size': (H, W),
            'tile_coord': (x1, y1, x2, y2),
            'objects': remapped_objs,
        }

        return img_t, mask_t, meta


if __name__ == '__main__':
    # quick local smoke test (won't run here without data)
    print('This is a dataloader template. Import ProcessedTileDataset in your training script.')
