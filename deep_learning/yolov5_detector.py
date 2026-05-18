"""
YOLOv5n detector, evaluated on VOC 2007 test.

Uses COCO-pretrained weights loaded via torch.hub (ultralytics/yolov5).
No fine-tuning on VOC: COCO class names are mapped to VOC at inference time.
"""
import os
import time

from config import VOC_ROOT_TEST, CLASSES
from deep_learning.coco_voc_mapping import COCO_NAME_TO_VOC

CONF_THRESHOLD = 0.3
IOU_THRESHOLD  = 0.45
MODEL_NAME     = "yolov5n"


def load_model():
    """Load YOLOv5n via torch.hub (downloads weights on first run)."""
    import torch

    print("    Loading YOLOv5n via torch.hub ...")
    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5n",
        pretrained=True,
        verbose=False,
    )
    model.eval()
    model.conf = CONF_THRESHOLD
    model.iou  = IOU_THRESHOLD
    return model


def run_inference_on_test(
    voc_root_test:  str   = VOC_ROOT_TEST,
    conf_threshold: float = CONF_THRESHOLD,
) -> dict:
    """
    Run YOLOv5n on every VOC 2007 test image.

    Returns dict with keys:
        model_name, detections, inference_times
    """
    from data.voc_loader import get_image_ids

    model = load_model()
    model.conf = conf_threshold

    test_ids        = get_image_ids(voc_root_test, "test")
    all_dets        = {cls: [] for cls in CLASSES}
    inference_times = []

    print(f"  Running YOLOv5n on {len(test_ids)} test images ...")

    for i, img_id in enumerate(test_ids):
        img_path = os.path.join(voc_root_test, "JPEGImages", f"{img_id}.jpg")

        t0      = time.perf_counter()
        results = model(img_path)
        inference_times.append(time.perf_counter() - t0)

        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            coco_name = row["name"]
            voc_cls   = COCO_NAME_TO_VOC.get(coco_name)
            if voc_cls is None:
                continue
            score = float(row["confidence"])
            if score < conf_threshold:
                continue
            all_dets[voc_cls].append({
                "image_id": img_id,
                "bbox":     [int(row["xmin"]), int(row["ymin"]),
                             int(row["xmax"]), int(row["ymax"])],
                "score":    score,
            })

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(test_ids)} done ...")

    total_dets = sum(len(v) for v in all_dets.values())
    avg_ms     = (sum(inference_times) / len(inference_times)) * 1000
    print(f"  YOLOv5n done: {total_dets} detections, avg {avg_ms:.1f} ms/image")

    return {
        "model_name":      MODEL_NAME,
        "detections":      all_dets,
        "inference_times": inference_times,
    }


if __name__ == "__main__":
    result = run_inference_on_test()
    fps = 1.0 / (sum(result["inference_times"]) / len(result["inference_times"]))
    print(f"Approximate FPS: {fps:.2f}")
