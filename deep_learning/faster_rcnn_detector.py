"""
Faster R-CNN (ResNet-50 + FPN) detector, evaluated on VOC 2007 test.

Uses COCO-pretrained weights from torchvision without fine-tuning.
The 20 VOC classes are mapped from their COCO indices at inference time.
"""
import os
import time

import torch
from torchvision import transforms as T
from PIL import Image

from config import VOC_ROOT_TEST, CLASSES
from deep_learning.coco_voc_mapping import COCO_IDX_TO_VOC

CONF_THRESHOLD = 0.3
MODEL_NAME = "faster_rcnn"


def load_model() -> torch.nn.Module:
    """Load Faster R-CNN ResNet-50-FPN with COCO weights."""
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    print("    Loading Faster R-CNN weights (COCO) ...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model


def run_inference_on_test(
    voc_root_test:  str   = VOC_ROOT_TEST,
    conf_threshold: float = CONF_THRESHOLD,
) -> dict:
    """
    Run Faster R-CNN on every VOC 2007 test image.

    Returns dict with keys:
        model_name, detections, inference_times
    """
    from data.voc_loader import get_image_ids

    device = torch.device("cpu")
    model  = load_model().to(device)
    to_tensor = T.ToTensor()

    test_ids        = get_image_ids(voc_root_test, "test")
    all_dets        = {cls: [] for cls in CLASSES}
    inference_times = []

    print(f"  Running Faster R-CNN on {len(test_ids)} test images ...")

    for i, img_id in enumerate(test_ids):
        img_path = os.path.join(voc_root_test, "JPEGImages", f"{img_id}.jpg")
        img      = Image.open(img_path).convert("RGB")
        tensor   = to_tensor(img).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(tensor)[0]
        inference_times.append(time.perf_counter() - t0)

        boxes  = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if float(score) < conf_threshold:
                continue
            voc_cls = COCO_IDX_TO_VOC.get(int(label))
            if voc_cls is None:
                continue
            all_dets[voc_cls].append({
                "image_id": img_id,
                "bbox":     [int(box[0]), int(box[1]),
                             int(box[2]), int(box[3])],
                "score":    float(score),
            })

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(test_ids)} done ...")

    total_dets = sum(len(v) for v in all_dets.values())
    avg_ms     = (sum(inference_times) / len(inference_times)) * 1000
    print(f"  Faster R-CNN done: {total_dets} detections, avg {avg_ms:.1f} ms/image")

    return {
        "model_name":      MODEL_NAME,
        "detections":      all_dets,
        "inference_times": inference_times,
    }


if __name__ == "__main__":
    result = run_inference_on_test()
    fps = 1.0 / (sum(result["inference_times"]) / len(result["inference_times"]))
    print(f"Approximate FPS: {fps:.2f}")
