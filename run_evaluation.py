"""
Evaluation runner: benchmarks all five detectors on VOC 2007 test.

Outputs individual JSON results per model plus a combined summary.
COCO-pretrained models (Faster R-CNN, SSD300, YOLOv5n) are evaluated
in a zero-shot transfer setting without fine-tuning on VOC.
"""
import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import (
    VOC_ROOT, VOC_ROOT_TEST, OUTPUT_DIR, MODELS_DIR, CLASSES,
    IOU_THRESHOLD, BENCHMARK_N_IMAGES, DETECTION_THRESHOLD,
)
from data.voc_loader import (
    load_all_annotations,
    build_detection_ground_truth,
    get_image_ids,
)
from evaluation.metrics import compute_map
from evaluation.benchmark import benchmark_detector, get_model_size_mb


# -- HOG+SVM --

def evaluate_hog_svm(voc_root_test: str = VOC_ROOT_TEST) -> dict:
    from classical.hog_svm_detector import load_classifiers, nms, sliding_window_detect

    print("\n" + "="*60)
    print("  Evaluating HOG + SVM on VOC 2007 test")
    print("="*60)

    model_path  = os.path.join(MODELS_DIR, "hog_svm_classifiers.pkl")
    classifiers = load_classifiers(model_path)
    print(f"Loaded {len(classifiers)} SVM classifiers\n")

    test_anns = load_all_annotations(voc_root_test, "test", CLASSES)
    gt        = build_detection_ground_truth(test_anns, CLASSES)

    all_dets = {cls: [] for cls in CLASSES}
    print("Running sliding-window detection on test set ...")
    for ann in tqdm(test_anns):
        img = cv2.imread(ann["image_path"])
        if img is None:
            continue
        for cls, pipe in classifiers.items():
            dets = sliding_window_detect(img, pipe, threshold=DETECTION_THRESHOLD)
            dets = nms(dets)
            for d in dets:
                all_dets[cls].append({
                    "image_id": ann["image_id"],
                    "bbox":     d["bbox"],
                    "score":    d["score"],
                })

    print("\nComputing mAP ...")
    map_results = compute_map(all_dets, gt, CLASSES, IOU_THRESHOLD)

    img_paths = [ann["image_path"] for ann in test_anns]

    def hog_detect(img):
        dets = []
        for cls, pipe in classifiers.items():
            d = sliding_window_detect(img, pipe, threshold=DETECTION_THRESHOLD)
            dets.extend(nms(d))
        return dets

    bench = benchmark_detector(hog_detect, img_paths, BENCHMARK_N_IMAGES, label="HOG+SVM")

    results = {
        "model":           "HOG + LinearSVM",
        "mAP":             map_results["mAP"],
        "per_class_ap":    {c: map_results["per_class"][c]["ap"] for c in CLASSES if c in map_results["per_class"]},
        "mean_ms":         bench["mean_ms"],
        "fps":             bench["fps"],
        "ram_delta_mb":    bench["ram_delta_mb"],
        "model_size_mb":   get_model_size_mb(model_path),
        "all_detections":  {c: all_dets[c][:500] for c in CLASSES},
    }

    out = os.path.join(OUTPUT_DIR, "hog_svm_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nHOG+SVM results saved -> {out}")
    return results


# -- YOLOv8n --

def evaluate_yolo(voc_root_test: str = VOC_ROOT_TEST) -> dict:
    from ultralytics import YOLO
    from deep_learning.yolo_detector import run_inference_on_test

    print("\n" + "="*60)
    print("  Evaluating YOLOv8n on VOC 2007 test")
    print("="*60)

    weights_path = os.path.join(MODELS_DIR, "yolov8n_voc2007", "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"No trained weights at {weights_path}\n"
            "Run: python -m deep_learning.yolo_detector first"
        )

    test_anns = load_all_annotations(voc_root_test, "test", CLASSES)
    gt        = build_detection_ground_truth(test_anns, CLASSES)

    print("Running YOLOv8n inference on test set ...")
    all_dets = run_inference_on_test(weights_path, voc_root_test)

    print("\nComputing mAP ...")
    map_results = compute_map(all_dets, gt, CLASSES, IOU_THRESHOLD)

    model     = YOLO(weights_path)
    img_paths = [ann["image_path"] for ann in test_anns]

    def yolo_detect(img):
        return model(img, verbose=False, device="cpu")

    bench = benchmark_detector(yolo_detect, img_paths, BENCHMARK_N_IMAGES, label="YOLOv8n")

    results = {
        "model":           "YOLOv8n (fine-tuned on VOC 2007)",
        "mAP":             map_results["mAP"],
        "per_class_ap":    {c: map_results["per_class"][c]["ap"] for c in CLASSES if c in map_results["per_class"]},
        "mean_ms":         bench["mean_ms"],
        "fps":             bench["fps"],
        "ram_delta_mb":    bench["ram_delta_mb"],
        "model_size_mb":   get_model_size_mb(weights_path),
    }

    out = os.path.join(OUTPUT_DIR, "yolo_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nYOLO results saved -> {out}")
    return results


# -- Faster R-CNN --

def evaluate_faster_rcnn(voc_root_test: str = VOC_ROOT_TEST) -> dict:
    from deep_learning.faster_rcnn_detector import load_model, run_inference_on_test
    from torchvision import transforms as T
    from PIL import Image as PILImage

    print("\n" + "="*60)
    print("  Evaluating Faster R-CNN (COCO pretrained) on VOC 2007 test")
    print("="*60)

    test_anns = load_all_annotations(voc_root_test, "test", CLASSES)
    gt        = build_detection_ground_truth(test_anns, CLASSES)

    print("Running Faster R-CNN inference on test set ...")
    inf_result = run_inference_on_test(voc_root_test=voc_root_test)
    all_dets   = inf_result["detections"]

    print("\nComputing mAP ...")
    map_results = compute_map(all_dets, gt, CLASSES, IOU_THRESHOLD)

    frcnn_model = load_model()
    frcnn_model.eval()
    to_tensor   = T.ToTensor()
    img_paths   = [ann["image_path"] for ann in test_anns]

    def frcnn_detect(img):
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil    = PILImage.fromarray(rgb)
        tensor = to_tensor(pil).unsqueeze(0)
        with torch.no_grad():
            return frcnn_model(tensor)

    bench = benchmark_detector(frcnn_detect, img_paths, BENCHMARK_N_IMAGES,
                               label="Faster R-CNN")

    results = {
        "model":           "Faster R-CNN ResNet-50 FPN (COCO pretrained)",
        "mAP":             map_results["mAP"],
        "per_class_ap":    {c: map_results["per_class"][c]["ap"] for c in CLASSES if c in map_results["per_class"]},
        "mean_ms":         bench["mean_ms"],
        "fps":             bench["fps"],
        "ram_delta_mb":    bench["ram_delta_mb"],
        "model_size_mb":   _model_size_from_params(frcnn_model),
    }

    out = os.path.join(OUTPUT_DIR, "faster_rcnn_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFaster R-CNN results saved -> {out}")
    return results


# -- SSD300 --

def evaluate_ssd(voc_root_test: str = VOC_ROOT_TEST) -> dict:
    from deep_learning.ssd_detector import load_model, run_inference_on_test
    from torchvision import transforms as T
    from PIL import Image as PILImage

    print("\n" + "="*60)
    print("  Evaluating SSD300 (COCO pretrained) on VOC 2007 test")
    print("="*60)

    test_anns = load_all_annotations(voc_root_test, "test", CLASSES)
    gt        = build_detection_ground_truth(test_anns, CLASSES)

    print("Running SSD300 inference on test set ...")
    inf_result = run_inference_on_test(voc_root_test=voc_root_test)
    all_dets   = inf_result["detections"]

    print("\nComputing mAP ...")
    map_results = compute_map(all_dets, gt, CLASSES, IOU_THRESHOLD)

    ssd_model = load_model()
    ssd_model.eval()
    to_tensor = T.ToTensor()
    img_paths = [ann["image_path"] for ann in test_anns]

    def ssd_detect(img):
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil    = PILImage.fromarray(rgb)
        tensor = to_tensor(pil).unsqueeze(0)
        with torch.no_grad():
            return ssd_model(tensor)

    bench = benchmark_detector(ssd_detect, img_paths, BENCHMARK_N_IMAGES,
                               label="SSD300")

    results = {
        "model":           "SSD300 VGG-16 (COCO pretrained)",
        "mAP":             map_results["mAP"],
        "per_class_ap":    {c: map_results["per_class"][c]["ap"] for c in CLASSES if c in map_results["per_class"]},
        "mean_ms":         bench["mean_ms"],
        "fps":             bench["fps"],
        "ram_delta_mb":    bench["ram_delta_mb"],
        "model_size_mb":   _model_size_from_params(ssd_model),
    }

    out = os.path.join(OUTPUT_DIR, "ssd_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSSD300 results saved -> {out}")
    return results


# -- YOLOv5n --

def evaluate_yolov5(voc_root_test: str = VOC_ROOT_TEST) -> dict:
    from deep_learning.yolov5_detector import load_model, run_inference_on_test

    print("\n" + "="*60)
    print("  Evaluating YOLOv5n (COCO pretrained) on VOC 2007 test")
    print("="*60)

    test_anns = load_all_annotations(voc_root_test, "test", CLASSES)
    gt        = build_detection_ground_truth(test_anns, CLASSES)

    print("Running YOLOv5n inference on test set ...")
    inf_result = run_inference_on_test(voc_root_test=voc_root_test)
    all_dets   = inf_result["detections"]

    print("\nComputing mAP ...")
    map_results = compute_map(all_dets, gt, CLASSES, IOU_THRESHOLD)

    yv5_model = load_model()
    img_paths = [ann["image_path"] for ann in test_anns]

    def yv5_detect(img):
        return yv5_model(img)

    bench = benchmark_detector(yv5_detect, img_paths, BENCHMARK_N_IMAGES,
                               label="YOLOv5n")

    results = {
        "model":           "YOLOv5n (COCO pretrained)",
        "mAP":             map_results["mAP"],
        "per_class_ap":    {c: map_results["per_class"][c]["ap"] for c in CLASSES if c in map_results["per_class"]},
        "mean_ms":         bench["mean_ms"],
        "fps":             bench["fps"],
        "ram_delta_mb":    bench["ram_delta_mb"],
        "model_size_mb":   _model_size_from_params(yv5_model.model),
    }

    out = os.path.join(OUTPUT_DIR, "yolov5_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nYOLOv5n results saved -> {out}")
    return results


# -- Combine --

def combine_results(hog_res, yolo_res, frcnn_res=None, ssd_res=None, yv5_res=None) -> dict:
    """Merge all model results into combined_results.json."""
    combined = {"HOG_SVM": hog_res, "YOLOv8n": yolo_res}
    if frcnn_res is not None:
        combined["Faster_RCNN"] = frcnn_res
    if ssd_res is not None:
        combined["SSD300"] = ssd_res
    if yv5_res is not None:
        combined["YOLOv5n"] = yv5_res

    out = os.path.join(OUTPUT_DIR, "combined_results.json")
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved -> {out}")

    print(f"\n{'Model':<22}  {'mAP@0.5':>8}  {'FPS':>8}")
    print("-" * 44)
    for key, res in combined.items():
        if res is None:
            continue
        print(f"{res.get('model', key):<22.22}  "
              f"{res.get('mAP', 0):>8.4f}  "
              f"{res.get('fps', 0):>8.2f}")
    print()
    return combined


def _model_size_from_params(model: torch.nn.Module) -> float:
    """Approximate model size in MB from parameter count."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total_bytes / (1024 ** 2), 2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hog-only",   action="store_true")
    parser.add_argument("--yolo-only",  action="store_true")
    parser.add_argument("--frcnn-only", action="store_true")
    parser.add_argument("--ssd-only",   action="store_true")
    parser.add_argument("--yv5-only",   action="store_true")
    args = parser.parse_args()

    run_all = not any([
        args.hog_only, args.yolo_only,
        args.frcnn_only, args.ssd_only, args.yv5_only,
    ])

    hog_res = yolo_res = frcnn_res = ssd_res = yv5_res = None

    if run_all or args.hog_only:
        hog_res   = evaluate_hog_svm()
    if run_all or args.yolo_only:
        yolo_res  = evaluate_yolo()
    if run_all or args.frcnn_only:
        frcnn_res = evaluate_faster_rcnn()
    if run_all or args.ssd_only:
        ssd_res   = evaluate_ssd()
    if run_all or args.yv5_only:
        yv5_res   = evaluate_yolov5()

    if run_all:
        combine_results(hog_res, yolo_res, frcnn_res, ssd_res, yv5_res)
