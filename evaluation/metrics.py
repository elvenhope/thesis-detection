"""
Pascal VOC 2007 mAP evaluation.

Follows the official protocol:
  - IoU threshold 0.5
  - Difficult objects ignored
  - 11-point interpolated AP
"""
import numpy as np
from collections import defaultdict


def compute_iou(box_a: list, box_b: list) -> float:
    xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """11-point interpolated AP (VOC 2007 protocol)."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 11.0
    return ap


def compute_class_ap(detections: list[dict], ground_truth: dict,
                     iou_threshold: float = 0.5) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AP for a single class.

    Parameters
    ----------
    detections    : list of {image_id, bbox, score}
    ground_truth  : dict[image_id] -> list of {bbox, difficult, used}
    iou_threshold : float

    Returns
    -------
    ap, recall_array, precision_array
    """
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)

    gt_by_image = defaultdict(list)
    for img_id, boxes in ground_truth.items():
        for b in boxes:
            gt_by_image[img_id].append({**b, "used": False})

    n_pos = sum(
        1 for boxes in ground_truth.values()
        for b in boxes if not b["difficult"]
    )

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))

    for i, det in enumerate(detections):
        img_id   = det["image_id"]
        gt_boxes = gt_by_image.get(img_id, [])

        best_iou = -np.inf
        best_j   = -1
        for j, gt in enumerate(gt_boxes):
            ov = compute_iou(det["bbox"], gt["bbox"])
            if ov > best_iou:
                best_iou = ov
                best_j   = j

        if best_iou >= iou_threshold:
            gt_match = gt_boxes[best_j]
            if gt_match["difficult"]:
                pass
            elif not gt_match["used"]:
                tp[i] = 1
                gt_match["used"] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    rec  = cum_tp / (n_pos + 1e-10)
    prec = cum_tp / (cum_tp + cum_fp + 1e-10)

    ap = voc_ap(rec, prec)
    return ap, rec, prec


def compute_map(all_detections: dict, all_ground_truth: dict,
                classes: list[str], iou_threshold: float = 0.5) -> dict:
    """
    Compute mAP and per-class AP.

    Parameters
    ----------
    all_detections   : dict[class_name] -> list of {image_id, bbox, score}
    all_ground_truth : dict[class_name][image_id] -> list of {bbox, difficult}

    Returns
    -------
    {"mAP": float, "per_class": {class_name: {"ap", "recall", "precision"}}}
    """
    results = {}
    aps = []

    for cls in classes:
        dets = all_detections.get(cls, [])
        gt   = all_ground_truth.get(cls, {})
        ap, rec, prec = compute_class_ap(dets, gt, iou_threshold)
        results[cls] = {"ap": ap, "recall": rec, "precision": prec}
        aps.append(ap)
        print(f"  {cls:16s}  AP = {ap:.4f}")

    mAP = float(np.mean(aps))
    print(f"\n  {'mAP':16s}       = {mAP:.4f}")
    return {"mAP": mAP, "per_class": results}
