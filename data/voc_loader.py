"""
VOC 2007 data loading utilities.
Parses XML annotations, provides train/val/test splits,
crops positive patches and samples negative patches for HOG+SVM training.
"""
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random
from pathlib import Path


def get_image_ids(voc_root: str, split: str) -> list[str]:
    """Return list of image IDs for a given split (train/val/trainval/test)."""
    split_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def parse_annotation(voc_root: str, image_id: str) -> dict:
    """
    Parse a single VOC XML annotation.
    Returns dict with keys:
      image_path, width, height,
      objects: list of {name, bbox: [xmin,ymin,xmax,ymax], difficult}
    """
    ann_path = os.path.join(voc_root, "Annotations", f"{image_id}.xml")
    tree = ET.parse(ann_path)
    root = tree.getroot()

    size   = root.find("size")
    width  = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name      = obj.find("name").text.lower().strip()
        difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
        bndbox    = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        objects.append({
            "name":      name,
            "bbox":      [xmin, ymin, xmax, ymax],
            "difficult": difficult,
        })

    return {
        "image_path": os.path.join(voc_root, "JPEGImages", f"{image_id}.jpg"),
        "image_id":   image_id,
        "width":      width,
        "height":     height,
        "objects":    objects,
    }


def load_all_annotations(voc_root: str, split: str, classes: list[str]) -> list[dict]:
    """Load all annotations for a split, filtering to known classes."""
    ids    = get_image_ids(voc_root, split)
    annots = []
    for img_id in ids:
        ann = parse_annotation(voc_root, img_id)
        ann["objects"] = [o for o in ann["objects"] if o["name"] in classes]
        annots.append(ann)
    return annots


# -- Patch extraction for HOG+SVM --

def iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two [xmin, ymin, xmax, ymax] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_positive_patches(annotations: list[dict], class_name: str,
                             win_size: tuple, max_patches: int,
                             seed: int = 42) -> list[np.ndarray]:
    """Crop and resize ground-truth bounding boxes for a class."""
    rng     = random.Random(seed)
    patches = []
    for ann in annotations:
        objs = [o for o in ann["objects"] if o["name"] == class_name and not o["difficult"]]
        if not objs:
            continue
        img = cv2.imread(ann["image_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        for obj in objs:
            xmin, ymin, xmax, ymax = obj["bbox"]
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            patch = img[ymin:ymax, xmin:xmax]
            patch = cv2.resize(patch, win_size)
            patches.append(patch)
            if len(patches) >= max_patches:
                return patches
    rng.shuffle(patches)
    return patches[:max_patches]


def extract_negative_patches(annotations: list[dict], class_name: str,
                             win_size: tuple, max_patches: int,
                             iou_thresh: float = 0.3,
                             seed: int = 42) -> list[np.ndarray]:
    """Sample random background crops that do not overlap ground truth."""
    rng     = random.Random(seed)
    patches = []
    rng.shuffle(annotations)
    for ann in annotations:
        gt_boxes = [o["bbox"] for o in ann["objects"] if o["name"] == class_name]
        img = cv2.imread(ann["image_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        attempts = 0
        while attempts < 30 and len(patches) < max_patches:
            attempts += 1
            pw = rng.randint(win_size[0], min(w, win_size[0] * 3))
            ph = rng.randint(win_size[1], min(h, win_size[1] * 3))
            x  = rng.randint(0, w - pw)
            y  = rng.randint(0, h - ph)
            cand = [x, y, x + pw, y + ph]
            if any(iou(cand, gt) > iou_thresh for gt in gt_boxes):
                continue
            patch = img[y:y+ph, x:x+pw]
            patch = cv2.resize(patch, win_size)
            patches.append(patch)
        if len(patches) >= max_patches:
            break
    return patches[:max_patches]


def build_detection_ground_truth(annotations: list[dict], classes: list[str]) -> dict:
    """
    Build ground-truth dict keyed by class, then image_id.
    Format: gt[class_name][image_id] = list of {bbox, difficult}
    Used for Pascal VOC mAP computation.
    """
    gt = {c: {} for c in classes}
    for ann in annotations:
        img_id = ann["image_id"]
        for obj in ann["objects"]:
            cls = obj["name"]
            if cls not in gt:
                continue
            gt[cls].setdefault(img_id, []).append({
                "bbox":      obj["bbox"],
                "difficult": obj["difficult"],
                "used":      False,
            })
    return gt
