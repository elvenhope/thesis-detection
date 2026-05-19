"""
migrate_json_to_db.py
Reads the existing combined_results.json produced by the experiment
pipeline and inserts every record into the SQLite database created
by results_db.py.

Run once after results_db.py is in place:
    python migrate_json_to_db.py

The script is idempotent: running it twice will not duplicate data
because model names have a UNIQUE constraint.
"""

import json
import os
from results_db import ResultsDatabase

# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "results",
                            "combined_results.json")
DB_PATH      = os.path.join(os.path.dirname(__file__), "results",
                            "benchmark.db")

# ------------------------------------------------------------------ #
#  Model metadata (matches Chapter 2 of the thesis)                    #
# ------------------------------------------------------------------ #
MODEL_META = {
    "HOG_SVM": {
        "type": "classical",
        "architecture": "sliding_window",
        "backbone": None,
        "pretrained_on": None,
        "fine_tuned_on": "VOC2007",
        "conf_threshold": 0.5,
        "nms_iou": 0.5,
    },
    "Faster_RCNN": {
        "type": "deep_learning",
        "architecture": "two_stage",
        "backbone": "ResNet-50 FPN",
        "pretrained_on": "COCO",
        "fine_tuned_on": None,
        "conf_threshold": 0.3,
        "nms_iou": 0.5,
    },
    "SSD300": {
        "type": "deep_learning",
        "architecture": "one_stage",
        "backbone": "VGG-16",
        "pretrained_on": "COCO",
        "fine_tuned_on": None,
        "conf_threshold": 0.3,
        "nms_iou": 0.5,
    },
    "YOLOv5n": {
        "type": "deep_learning",
        "architecture": "one_stage",
        "backbone": "CSPDarknet (C3)",
        "pretrained_on": "COCO",
        "fine_tuned_on": None,
        "conf_threshold": 0.25,
        "nms_iou": 0.45,
    },
    "YOLOv8n": {
        "type": "deep_learning",
        "architecture": "one_stage",
        "backbone": "CSPDarknet (C2f)",
        "pretrained_on": "COCO",
        "fine_tuned_on": "VOC2007",
        "conf_threshold": 0.25,
        "nms_iou": 0.7,
    },
}

# VOC 2007 test set class instance counts (non-difficult)
CLASS_INSTANCES = {
    "aeroplane": 285, "bicycle": 337, "bird": 459, "boat": 263,
    "bottle": 469, "bus": 213, "car": 1201, "cat": 358,
    "chair": 756, "cow": 244, "diningtable": 206, "dog": 489,
    "horse": 348, "motorbike": 325, "person": 4528,
    "pottedplant": 480, "sheep": 242, "sofa": 239, "train": 282,
    "tvmonitor": 308,
}


def main():
    # Load the JSON results
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    # Open database
    db = ResultsDatabase(DB_PATH)
    db.init_schema()

    # Insert dataset (one record: VOC 2007 test split)
    dataset_id = db.insert_dataset(
        name="Pascal VOC",
        version="2007",
        split="test",
        num_images=4952,
        num_classes=20,
    )

    # Insert the 20 object classes
    class_ids = {}
    for cls_name, count in CLASS_INSTANCES.items():
        class_ids[cls_name] = db.insert_object_class(cls_name, count)

    # Insert each model, experiment, and results
    for key, data in results.items():
        meta = MODEL_META[key]

        # Model
        model_id = db.insert_model(
            name=data["model"],
            type_=meta["type"],
            architecture=meta["architecture"],
            backbone=meta["backbone"],
            model_size_mb=data.get("model_size_mb"),
            pretrained_on=meta["pretrained_on"],
            fine_tuned_on=meta["fine_tuned_on"],
        )

        # Experiment
        experiment_id = db.insert_experiment(
            model_id=model_id,
            dataset_id=dataset_id,
            hardware="Apple M1 CPU (8 GB RAM)",
            confidence_threshold=meta["conf_threshold"],
            nms_iou_threshold=meta["nms_iou"],
        )

        # Overall result
        db.insert_overall_result(
            experiment_id=experiment_id,
            map_50=data["mAP"],
            fps=data.get("fps"),
            mean_inference_ms=data.get("mean_ms"),
            ram_delta_mb=data.get("ram_delta_mb"),
        )

        # Per-class results
        for cls_name, ap in data.get("per_class_ap", {}).items():
            if cls_name in class_ids:
                db.insert_per_class_result(
                    experiment_id=experiment_id,
                    class_id=class_ids[cls_name],
                    average_precision=ap,
                )

    db.close()
    print(f"Migration complete. Database saved to: {DB_PATH}")
    print(f"  Models inserted:      {len(results)}")
    print(f"  Classes inserted:     {len(class_ids)}")
    print(f"  Per-class AP records: "
          f"{sum(len(d.get('per_class_ap', {})) for d in results.values())}")


if __name__ == "__main__":
    main()
