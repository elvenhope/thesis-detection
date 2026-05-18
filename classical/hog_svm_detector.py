"""
Classical HOG + LinearSVM object detector for VOC 2007.

Pipeline:
  1. Extract HOG features from positive (GT crops) and negative (background) patches
  2. Train one LinearSVC per class
  3. Hard-negative mining: re-run detection on training images, collect false positives
  4. Retrain SVMs with the augmented negative set
  5. Save all classifiers to disk
"""
import os
import time
import pickle
import numpy as np
import cv2
from skimage.feature import hog as skimage_hog
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from config import (
    VOC_ROOT, MODELS_DIR, CLASSES,
    HOG_WIN_SIZE, HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM,
    SVM_C, SVM_MAX_ITER,
    MAX_POSITIVES, MAX_NEGATIVES,
    SW_STEP_SIZE, SW_SCALES,
    NMS_THRESHOLD, DETECTION_THRESHOLD,
    SEED,
)
from data.voc_loader import (
    load_all_annotations,
    extract_positive_patches,
    extract_negative_patches,
    iou,
)


def extract_hog(img_bgr: np.ndarray) -> np.ndarray:
    """Extract HOG feature vector from a BGR image patch."""
    img = cv2.resize(img_bgr, HOG_WIN_SIZE)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = skimage_hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    )
    return feats.astype(np.float32)


def patches_to_features(patches: list[np.ndarray]) -> np.ndarray:
    """Convert list of image patches to HOG feature matrix (N x D)."""
    return np.vstack([extract_hog(p) for p in patches])


def train_class_svm(pos_feats: np.ndarray, neg_feats: np.ndarray) -> Pipeline:
    """Train a calibrated LinearSVM pipeline for one class."""
    X = np.vstack([pos_feats, neg_feats])
    y = np.array([1] * len(pos_feats) + [0] * len(neg_feats), dtype=np.int32)

    svc = LinearSVC(C=SVM_C, max_iter=SVM_MAX_ITER, random_state=SEED)
    cal = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
    scaler = StandardScaler()

    pipe = Pipeline([("scaler", scaler), ("clf", cal)])
    pipe.fit(X, y)
    return pipe


def sliding_window_detect(img_bgr: np.ndarray, pipeline: Pipeline,
                          threshold: float = DETECTION_THRESHOLD) -> list[dict]:
    """
    Run multi-scale sliding-window detection on one image.
    Returns list of {bbox: [xmin, ymin, xmax, ymax], score: float}.
    """
    h, w = img_bgr.shape[:2]
    detections = []
    win_w, win_h = HOG_WIN_SIZE

    for scale in SW_SCALES:
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < win_w or new_h < win_h:
            continue
        scaled = cv2.resize(img_bgr, (new_w, new_h))

        for y in range(0, new_h - win_h, SW_STEP_SIZE):
            for x in range(0, new_w - win_w, SW_STEP_SIZE):
                patch = scaled[y:y + win_h, x:x + win_w]
                feat  = extract_hog(patch).reshape(1, -1)
                score = pipeline.predict_proba(feat)[0][1]
                if score > threshold:
                    xmin = int(x / scale)
                    ymin = int(y / scale)
                    xmax = int((x + win_w) / scale)
                    ymax = int((y + win_h) / scale)
                    detections.append({
                        "bbox":  [xmin, ymin, xmax, ymax],
                        "score": float(score),
                    })
    return detections


def nms(detections: list[dict], iou_thresh: float = NMS_THRESHOLD) -> list[dict]:
    """Non-maximum suppression on a list of detections."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if iou(best["bbox"], d["bbox"]) < iou_thresh
        ]
    return kept


def collect_hard_negatives(annotations: list[dict], class_name: str,
                           pipeline: Pipeline, max_hn: int = 2000) -> np.ndarray:
    """
    Run the current detector on training images and collect false-positive
    patches as hard negatives for retraining.
    """
    hard_feats = []
    for ann in tqdm(annotations, desc=f"  HNM {class_name}", leave=False):
        gt_boxes = [o["bbox"] for o in ann["objects"] if o["name"] == class_name]
        img = cv2.imread(ann["image_path"])
        if img is None:
            continue
        dets = sliding_window_detect(img, pipeline, threshold=0.3)
        for det in dets:
            if all(iou(det["bbox"], gt) < 0.3 for gt in gt_boxes):
                h_img = img[det["bbox"][1]:det["bbox"][3], det["bbox"][0]:det["bbox"][2]]
                if h_img.size == 0:
                    continue
                hard_feats.append(extract_hog(h_img))
                if len(hard_feats) >= max_hn:
                    return np.vstack(hard_feats)
    return np.vstack(hard_feats) if hard_feats else np.empty((0, 0))


def train(voc_root: str = VOC_ROOT, split: str = "trainval"):
    print(f"\n{'='*60}")
    print(f"  HOG + SVM Detector: Training on VOC 2007 '{split}'")
    print(f"{'='*60}\n")

    annotations = load_all_annotations(voc_root, split, CLASSES)
    print(f"Loaded {len(annotations)} annotations\n")

    classifiers = {}

    for cls in CLASSES:
        print(f"[{CLASSES.index(cls)+1:02d}/{len(CLASSES)}]  {cls}")

        pos_patches = extract_positive_patches(
            annotations, cls, HOG_WIN_SIZE, MAX_POSITIVES, seed=SEED)
        print(f"    positives: {len(pos_patches)}", end="")

        neg_patches = extract_negative_patches(
            annotations, cls, HOG_WIN_SIZE, MAX_NEGATIVES, seed=SEED)
        print(f"  |  negatives: {len(neg_patches)}")

        if len(pos_patches) < 10:
            print(f"    WARNING: too few positives for {cls}, skipping")
            continue

        print("    extracting HOG features ...", end=" ", flush=True)
        pos_feats = patches_to_features(pos_patches)
        neg_feats = patches_to_features(neg_patches)
        print(f"done  ({pos_feats.shape[1]}D features)")

        print("    training SVM ...", end=" ", flush=True)
        t0   = time.time()
        pipe = train_class_svm(pos_feats, neg_feats)
        print(f"done  ({time.time()-t0:.1f}s)")

        print("    hard-negative mining ...", end=" ", flush=True)
        hn_feats = collect_hard_negatives(annotations, cls, pipe, max_hn=1000)
        if len(hn_feats) > 10:
            neg_feats_aug = np.vstack([neg_feats, hn_feats])
            pipe = train_class_svm(pos_feats, neg_feats_aug)
            print(f"+{len(hn_feats)} hard negs added, retrained")
        else:
            print("no hard negs found")

        classifiers[cls] = pipe
        print()

    model_path = os.path.join(MODELS_DIR, "hog_svm_classifiers.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(classifiers, f)
    print(f"\nSaved {len(classifiers)} classifiers -> {model_path}")
    return classifiers


def load_classifiers(model_path: str | None = None) -> dict:
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "hog_svm_classifiers.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def detect_image(img_bgr: np.ndarray, classifiers: dict,
                 threshold: float = DETECTION_THRESHOLD) -> list[dict]:
    """Run all per-class SVMs on a single image."""
    all_dets = []
    for cls, pipe in classifiers.items():
        dets = sliding_window_detect(img_bgr, pipe, threshold)
        dets = nms(dets)
        for d in dets:
            d["class_name"] = cls
        all_dets.extend(dets)
    return all_dets


if __name__ == "__main__":
    train()
