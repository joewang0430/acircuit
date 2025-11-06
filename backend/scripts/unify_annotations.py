#!/usr/bin/env python3
"""
Unify CGHD and HCD annotations into a single UNet-ready dataset under data/processed/.

Outputs per image (keeps original resolution and filenames):
- images/<dataset>/<original_filename>
- masks/<dataset>/<original_basename>.png  (single-channel class-index mask: 0=background)
- annotations/<dataset>/<original_basename>.json (per-image metadata: objects with bbox, class_id, rotation, text, source_label, dataset)
Also writes a dataset-level manifest: annotations/manifest.json

Notes:
- Only classes defined in backend/app/core/label_map.json are written into the mask; others (incl. text) are treated as background (0) but preserved in metadata.
- Overlap policy: larger area objects overwrite smaller ones in the mask.
- Rotation and text are optional metadata; preserved if available.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]  # repo root: .../acircuit
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_dirs():
    for sub in [
        PROCESSED_DIR / "images" / "cghd",
        PROCESSED_DIR / "images" / "hcd",
        PROCESSED_DIR / "masks" / "cghd",
        PROCESSED_DIR / "masks" / "hcd",
        PROCESSED_DIR / "annotations" / "cghd",
        PROCESSED_DIR / "annotations" / "hcd",
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def load_label_map(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    # Build dataset-specific name -> class_id maps
    cghd_map: Dict[str, int] = {}
    hcd_map: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    for k, v in lm.items():
        cid = int(v["id"]) if isinstance(v["id"], int) else int(k)
        id_to_name[cid] = v.get("canonical", str(cid))
        for name in v.get("cghd", []) or []:
            cghd_map[name.strip().lower()] = cid
        for name in v.get("hcd", []) or []:
            hcd_map[name.strip()] = cid  # HCD names are case-significant in categories
    return {"cghd": cghd_map, "hcd": hcd_map, "id_to_name": id_to_name}


def parse_cghd_xml(xml_path: Path) -> Dict[str, Any]:
    """Parse a single CGHD VOC-like XML file into a common structure."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    filename = root.findtext("filename")
    size_el = root.find("size")
    W = int(size_el.findtext("width"))
    H = int(size_el.findtext("height"))
    # try to build image path relative to xml
    img_path = root.findtext("path")
    if img_path and img_path.startswith("./"):
        # In CGHD this is relative to the CGHD root (data/cghd)
        cghd_root = (xml_path.parents[2]).resolve()  # .../data/cghd
        img_path = (cghd_root / img_path[2:]).resolve()
    else:
        # typical layout: replace annotations -> images
        img_dir = (xml_path.parent.parent / "images").resolve()
        img_path = (img_dir / filename).resolve()

    objects = []
    oid = 1
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb_el = obj.find("bndbox")
        if bb_el is None:
            continue
        xmin = int(float(bb_el.findtext("xmin")))
        ymin = int(float(bb_el.findtext("ymin")))
        xmax = int(float(bb_el.findtext("xmax")))
        ymax = int(float(bb_el.findtext("ymax")))
        # rotation may appear under <bndbox><rotation> or directly under <object>
        rot_text = obj.findtext("rotation")
        if rot_text is None:
            rot_text = bb_el.findtext("rotation")
        try:
            rotation = float(rot_text) if rot_text is not None and rot_text != "" else None
        except Exception:
            rotation = None
        text_el = obj.find("text")
        text_val = text_el.text if text_el is not None else None
        objects.append({
            "id": oid,
            "source_label": name,
            "bbox": [xmin, ymin, xmax, ymax],
            "rotation": rotation,
            "text": text_val,
        })
        oid += 1

    # Use actual image size from file (XML size has inconsistencies in CGHD)
    try:
        from PIL import Image
        with Image.open(img_path) as _im:
            W_img, H_img = _im.size
        H, W = int(H_img), int(W_img)
    except Exception:
        # fallback to XML size
        pass

    return {
        "dataset": "cghd",
        "filename": filename,
        "image_path": str(img_path),
        "image_size": [H, W],
        "objects": objects,
    }


def parse_hcd_component_json(hcd_json_path: Path, image_file_name: str) -> Dict[str, Any]:
    """Parse HCD COCO-like json and extract one image record by file_name."""
    with open(hcd_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # build image id map
    img_map = {img["id"] if isinstance(img, dict) else img.get("id"): img for img in data.get("images", [])}
    file_to_img = {img["file_name"]: img for img in data.get("images", [])}
    if image_file_name not in file_to_img:
        raise FileNotFoundError(f"HCD image '{image_file_name}' not listed in annotations")
    img_rec = file_to_img[image_file_name]
    image_id = img_rec["id"]
    H = int(img_rec["height"])
    W = int(img_rec["width"])
    # categories
    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    # annotations: COCO-like top-level key is typically 'annotations'
    anns = data.get("annotations", [])
    if not anns:
        # try alternative: some dumps might store under 'component_annotations'
        anns = data.get("component_annotations", [])
    objects = []
    oid = 1
    for a in anns:
        if int(a.get("image_id", -1)) != int(image_id):
            continue
        cat_id = int(a.get("category_id"))
        cat_name = cat_id_to_name.get(cat_id, "")
        bbox = a.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        xmin = int(round(x))
        ymin = int(round(y))
        xmax = int(round(x + w))
        ymax = int(round(y + h))
        objects.append({
            "id": oid,
            "source_label": cat_name,
            "bbox": [xmin, ymin, xmax, ymax],
            "rotation": None,
            "text": None,
        })
        oid += 1

    # compute image path
    img_path = (hcd_json_path.parent / "Circuit Diagram Images" / image_file_name).resolve()
    return {
        "dataset": "hcd",
        "filename": image_file_name,
        "image_path": str(img_path),
        "image_size": [H, W],
        "objects": objects,
    }


def build_mask_and_metadata(entry: Dict[str, Any], label_maps: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    H, W = entry["image_size"]
    dataset = entry["dataset"]
    if dataset == "cghd":
        name_to_id = label_maps["cghd"]
        normalize = lambda s: (s or "").strip().lower()
    else:
        name_to_id = label_maps["hcd"]
        normalize = lambda s: (s or "")  # HCD names match exactly

    # mask init
    mask = np.zeros((H, W), dtype=np.uint16)

    # objects sorted by area desc (larger overwrite smaller)
    def area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    objs_sorted = sorted(entry["objects"], key=lambda o: area(o["bbox"]), reverse=True)
    meta_objs = []
    for obj in objs_sorted:
        raw = obj["source_label"]
        raw_key = normalize(raw)
        class_id = name_to_id.get(raw_key, 0)
        # treat text as background in mask but preserve in metadata
        if raw_key == "text":
            class_id = 0
        x1, y1, x2, y2 = obj["bbox"]
        # clamp
        x1c = max(0, min(W, x1))
        y1c = max(0, min(H, y1))
        x2c = max(0, min(W, x2))
        y2c = max(0, min(H, y2))
        if class_id != 0 and x2c > x1c and y2c > y1c:
            mask[y1c:y2c, x1c:x2c] = class_id
        meta_obj = dict(obj)
        meta_obj["class_id"] = int(class_id)
        meta_objs.append(meta_obj)

    meta = {
        "filename": entry["filename"],
        "dataset": dataset,
        "original_path": entry["image_path"],
        "image_size": entry["image_size"],
        "objects": meta_objs,
    }
    return mask, meta


def save_sample(entry: Dict[str, Any], mask: np.ndarray, meta: Dict[str, Any]):
    dataset = entry["dataset"]
    filename = entry["filename"]
    stem = Path(filename).stem
    # copy image
    dst_img = PROCESSED_DIR / "images" / dataset / filename
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if not dst_img.exists():
        shutil.copy(entry["image_path"], dst_img)
    # save mask
    dst_mask = PROCESSED_DIR / "masks" / dataset / f"{stem}.png"
    Image.fromarray(mask.astype(np.uint16)).save(dst_mask)
    # save json
    dst_ann = PROCESSED_DIR / "annotations" / dataset / f"{stem}.json"
    with open(dst_ann, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return str(dst_img), str(dst_mask), str(dst_ann)


def update_manifest(records: List[Dict[str, Any]]):
    man_path = PROCESSED_DIR / "annotations" / "manifest.json"
    manifest = {"images": []}
    if man_path.exists():
        try:
            with open(man_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass
    manifest["images"].extend(records)
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Unify CGHD & HCD into processed dataset (sample-capable)")
    parser.add_argument("--label-map", default=str(ROOT / "backend/app/core/label_map.json"))
    parser.add_argument("--cghd-xml", help="Path to one CGHD XML to process", required=False)
    parser.add_argument("--hcd-json", default=str(DATA_DIR / "hcd/Component Symbol and Text Label Data/component_annotations.json"))
    parser.add_argument("--hcd-image", help="HCD image file_name to process (e.g., circuit_639.jpg)")
    args = parser.parse_args()

    ensure_dirs()
    lm = load_label_map(Path(args.label_map))
    records_for_manifest = []

    # CGHD sample
    if args.cghd_xml:
        cghd_entry = parse_cghd_xml(Path(args.cghd_xml))
        mask, meta = build_mask_and_metadata(cghd_entry, lm)
        img_p, mask_p, ann_p = save_sample(cghd_entry, mask, meta)
        records_for_manifest.append({
            "id": cghd_entry["filename"],
            "dataset": "cghd",
            "image_path": os.path.relpath(img_p, start=PROCESSED_DIR),
            "mask_path": os.path.relpath(mask_p, start=PROCESSED_DIR),
            "ann_path": os.path.relpath(ann_p, start=PROCESSED_DIR),
            "split": "train",
        })

    # HCD sample
    if args.hcd_image:
        hcd_entry = parse_hcd_component_json(Path(args.hcd_json), args.hcd_image)
        mask, meta = build_mask_and_metadata(hcd_entry, lm)
        img_p, mask_p, ann_p = save_sample(hcd_entry, mask, meta)
        records_for_manifest.append({
            "id": hcd_entry["filename"],
            "dataset": "hcd",
            "image_path": os.path.relpath(img_p, start=PROCESSED_DIR),
            "mask_path": os.path.relpath(mask_p, start=PROCESSED_DIR),
            "ann_path": os.path.relpath(ann_p, start=PROCESSED_DIR),
            "split": "train",
        })

    if records_for_manifest:
        update_manifest(records_for_manifest)
        print("Wrote records to manifest:")
        for r in records_for_manifest:
            print(json.dumps(r, ensure_ascii=False))
    else:
        print("No samples provided. Use --cghd-xml and/or --hcd-image.")


if __name__ == "__main__":
    main()
