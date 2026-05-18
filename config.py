"""
Central configuration for all detection experiments.
Paths are loaded from a .env file; copy .env.example and set your local paths.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Paths (set via .env file)
VOC_ROOT      = os.environ.get("VOC_ROOT", "/path/to/VOCdevkit/VOC2007")
VOC_ROOT_TEST = os.environ.get("VOC_ROOT_TEST", "/path/to/VOCdevkit/VOC2007_test")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "results")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "models")
FIGURES_DIR   = os.path.join(os.path.dirname(__file__), "figures")

# VOC classes
CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES  = len(CLASSES)

# HOG + SVM
HOG_WIN_SIZE        = (64, 128)
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM      = "L2-Hys"

SVM_C               = 0.01
SVM_MAX_ITER        = 5000

# Sliding window
SW_STEP_SIZE        = 32
SW_SCALES           = [0.75, 1.0, 1.25]
NMS_THRESHOLD       = 0.5
DETECTION_THRESHOLD = 0.5

# Hard-negative mining
HNM_ROUNDS          = 1
MAX_NEGATIVES       = 4000
MAX_POSITIVES       = 2000

# YOLO
YOLO_MODEL          = "yolov8n.pt"
YOLO_EPOCHS         = 80
YOLO_BATCH          = 8
YOLO_IMG_SIZE       = 640
YOLO_LR             = 0.01
YOLO_DATA_YAML      = os.path.join(os.path.dirname(__file__), "deep_learning", "voc2007.yaml")

# Evaluation
IOU_THRESHOLD       = 0.5
BENCHMARK_N_IMAGES  = 100

# Reproducibility
SEED = 42

# Create output directories
for d in [OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
