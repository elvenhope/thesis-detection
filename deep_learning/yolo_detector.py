"""
YOLOv8n detector pipeline for VOC 2007.

Steps:
  1. Convert VOC annotations to YOLO txt format
  2. Write data YAML for Ultralytics
  3. Train YOLOv8n on VOC 2007 trainval
  4. Run inference on VOC 2007 test
"""
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from config import (
    VOC_ROOT, VOC_ROOT_TEST, OUTPUT_DIR, MODELS_DIR, CLASSES, CLASS_TO_IDX,
    YOLO_MODEL, YOLO_EPOCHS, YOLO_BATCH, YOLO_IMG_SIZE,
    YOLO_LR, YOLO_DATA_YAML, SEED,
)


def convert_voc_to_yolo(voc_root: str, output_dir: str, split: str):
    """
    Convert VOC XML annotations to YOLO txt format.
    Each line: class_id cx cy w h (normalised 0-1).
    """
    images_out = os.path.join(output_dir, "images", split)
    labels_out = os.path.join(output_dir, "labels", split)
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    split_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")
    with open(split_file) as f:
        image_ids = [line.strip() for line in f if line.strip()]

    skipped = 0
    for img_id in image_ids:
        ann_path = os.path.join(voc_root, "Annotations", f"{img_id}.xml")
        img_path = os.path.join(voc_root, "JPEGImages", f"{img_id}.jpg")

        tree = ET.parse(ann_path)
        root = tree.getroot()
        size = root.find("size")
        W = int(size.find("width").text)
        H = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name not in CLASS_TO_IDX:
                skipped += 1
                continue
            cls_id = CLASS_TO_IDX[name]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            cx = ((xmin + xmax) / 2) / W
            cy = ((ymin + ymax) / 2) / H
            bw = (xmax - xmin) / W
            bh = (ymax - ymin) / H
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_file = os.path.join(labels_out, f"{img_id}.txt")
        with open(label_file, "w") as f:
            f.write("\n".join(lines))

        dst_img = os.path.join(images_out, f"{img_id}.jpg")
        if not os.path.exists(dst_img):
            os.symlink(os.path.abspath(img_path), dst_img)

    print(f"  Converted '{split}': {len(image_ids)} images  ({skipped} unknown-class objects skipped)")


def build_yolo_dataset(voc_root: str, voc_root_test: str, dataset_dir: str):
    """Convert both splits and write the YAML config."""
    print("\nConverting VOC 2007 -> YOLO format ...")
    convert_voc_to_yolo(voc_root, dataset_dir, "trainval")
    convert_voc_to_yolo(voc_root_test, dataset_dir, "test")

    yaml_content = f"""# VOC 2007 dataset for YOLOv8 training
path: {os.path.abspath(dataset_dir)}
train: images/trainval
val:   images/test
test:  images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "voc2007.yaml"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  YAML written -> {yaml_path}")
    return yaml_path


def train(voc_root: str = VOC_ROOT, voc_root_test: str = VOC_ROOT_TEST):
    from ultralytics import YOLO

    dataset_dir = os.path.join(OUTPUT_DIR, "yolo_dataset")
    yaml_path   = build_yolo_dataset(voc_root, voc_root_test, dataset_dir)

    print(f"\nTraining YOLOv8n on VOC 2007 trainval ...")
    print(f"  epochs={YOLO_EPOCHS}  batch={YOLO_BATCH}  imgsz={YOLO_IMG_SIZE}  device=cpu")

    model = YOLO(YOLO_MODEL)
    results = model.train(
        data      = yaml_path,
        epochs    = YOLO_EPOCHS,
        batch     = YOLO_BATCH,
        imgsz     = YOLO_IMG_SIZE,
        lr0       = YOLO_LR,
        device    = "cpu",
        seed      = SEED,
        project   = MODELS_DIR,
        name      = "yolov8n_voc2007",
        verbose   = True,
        workers   = 2,
    )

    best_weights = os.path.join(
        MODELS_DIR, "yolov8n_voc2007", "weights", "best.pt"
    )
    print(f"\nBest weights saved -> {best_weights}")
    return best_weights


def run_inference_on_test(weights_path: str, voc_root_test: str = VOC_ROOT_TEST) -> dict:
    """Run YOLOv8 on every VOC 2007 test image and collect detections."""
    from ultralytics import YOLO
    from data.voc_loader import get_image_ids
    import cv2

    model    = YOLO(weights_path)
    test_ids = get_image_ids(voc_root_test, "test")
    all_dets = {cls: [] for cls in CLASSES}

    for img_id in test_ids:
        img_path = os.path.join(voc_root_test, "JPEGImages", f"{img_id}.jpg")
        results  = model(img_path, verbose=False, device="cpu")
        res      = results[0]

        for box in res.boxes:
            cls_id   = int(box.cls.item())
            cls_name = CLASSES[cls_id]
            score    = float(box.conf.item())
            xyxy     = box.xyxy[0].tolist()
            bbox     = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
            all_dets[cls_name].append({
                "image_id": img_id,
                "bbox":     bbox,
                "score":    score,
            })

    return all_dets


if __name__ == "__main__":
    weights = train()
    print("\nTo evaluate, run: python run_evaluation.py")
