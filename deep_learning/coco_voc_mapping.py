"""
COCO -> VOC class mapping utilities.

The pretrained detectors (Faster R-CNN, SSD300, YOLOv5n) output COCO class
indices or names. Since the 20 Pascal VOC classes are a subset of COCO,
predictions are mapped at inference time and non-VOC classes are discarded.

Two dicts are provided:
  COCO_IDX_TO_VOC   : torchvision models (1-based integer label -> VOC name)
  COCO_NAME_TO_VOC  : YOLOv5 hub model   (COCO string name -> VOC name)
"""

# torchvision uses 1-based COCO category IDs (0 = __background__)
COCO_IDX_TO_VOC: dict[int, str] = {
    1:  "person",
    2:  "bicycle",
    3:  "car",
    4:  "motorbike",       # COCO: "motorcycle"
    5:  "aeroplane",       # COCO: "airplane"
    6:  "bus",
    7:  "train",
    9:  "boat",            # index 8 = "truck", not in VOC
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    44: "bottle",
    62: "chair",
    63: "sofa",            # COCO: "couch"
    64: "pottedplant",     # COCO: "potted plant"
    67: "diningtable",     # COCO: "dining table"
    72: "tvmonitor",       # COCO: "tv"
}

# torch.hub YOLOv5 exposes COCO class names as strings
COCO_NAME_TO_VOC: dict[str, str] = {
    "person":        "person",
    "bicycle":       "bicycle",
    "car":           "car",
    "motorcycle":    "motorbike",
    "airplane":      "aeroplane",
    "bus":           "bus",
    "train":         "train",
    "boat":          "boat",
    "bird":          "bird",
    "cat":           "cat",
    "dog":           "dog",
    "horse":         "horse",
    "sheep":         "sheep",
    "cow":           "cow",
    "bottle":        "bottle",
    "chair":         "chair",
    "couch":         "sofa",
    "potted plant":  "pottedplant",
    "dining table":  "diningtable",
    "tv":            "tvmonitor",
}
