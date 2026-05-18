# Object Detection: Classical vs. Deep Learning

Companion code for the bachelor thesis *"Design, Implementation, and Comparative Evaluation of Classical and Deep Learning Object Detection"* (Turiba University, 2026).

Five object detection models are benchmarked on the **Pascal VOC 2007** test set:

| Model | Type | mAP@0.5 | FPS (CPU) |
|---|---|---|---|
| Faster R-CNN (ResNet-50 + FPN) | Two-stage, COCO pretrained | 0.7848 | ~0.90 |
| YOLOv8n | Single-stage, fine-tuned on VOC | 0.7225 | ~20.40 |
| SSD300 (VGG-16) | Single-stage, COCO pretrained | 0.6956 | ~5.28 |
| YOLOv5n | Single-stage, COCO pretrained | 0.5827 | ~24.50 |
| HOG + LinearSVM | Classical sliding-window | 0.0018 | ~0.10 |

All experiments run on an Apple M1 (CPU-only), Python 3.14, PyTorch 2.11.

---

## Repository structure

```
.
├── config.py                   # Central configuration (paths, hyperparameters)
├── run_all.py                  # Master pipeline runner
├── run_evaluation.py           # Evaluate all five detectors
├── classical/
│   └── hog_svm_detector.py     # HOG feature extraction, SVM training, sliding-window detection
├── data/
│   └── voc_loader.py           # VOC 2007 XML parsing, patch extraction, ground-truth builder
├── deep_learning/
│   ├── coco_voc_mapping.py     # COCO to VOC class-index mapping
│   ├── faster_rcnn_detector.py # Faster R-CNN inference (torchvision)
│   ├── ssd_detector.py         # SSD300 inference (torchvision)
│   ├── yolo_detector.py        # YOLOv8n training and inference (Ultralytics)
│   └── yolov5_detector.py      # YOLOv5n inference (torch.hub)
├── evaluation/
│   ├── benchmark.py            # Inference timing and RAM measurement
│   └── metrics.py              # Pascal VOC mAP (11-point interpolation)
└── visualization/
    ├── thesis_figures.py        # Main comparison figures
    ├── fig_voc_class_distribution.py
    ├── fig_voc_sample_grid.py
    ├── fig_per_class_ap_table.py
    ├── fig_pr_curve_ap.py
    ├── fig_image_pyramid.py
    └── fig_nms_diagram.py
```

## Setup

1. **Clone and install dependencies:**

```bash
git clone https://github.com/elvenhope/thesis-detection.git
cd thesis-detection
pip install -r requirements.txt
```

2. **Download Pascal VOC 2007** (trainval + test) and extract to a local directory.

3. **Create a `.env` file** from the example and set the paths:

```bash
cp .env.example .env
# Edit .env with your local VOC paths
```

## Usage

Run the full pipeline (train HOG+SVM and YOLOv8n, evaluate all five models, generate figures):

```bash
python run_all.py
```

Or evaluate individual models:

```bash
python run_evaluation.py --frcnn-only
python run_evaluation.py --ssd-only
```

Generate thesis figures from existing results:

```bash
python -m visualization.thesis_figures
python -m visualization.fig_voc_class_distribution
```