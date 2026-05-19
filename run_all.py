#!/usr/bin/env python3
"""
Master runner: executes the complete thesis pipeline.

Steps:
    1. Train HOG+SVM on VOC 2007 trainval
    2. Train YOLOv8n on VOC 2007 trainval
    3. Evaluate all five detectors on VOC 2007 test
    4. Generate thesis figures

Faster R-CNN, SSD300, and YOLOv5n use COCO pretrained weights
and require no training step.
"""
import argparse
import os

import config


def main():
    parser = argparse.ArgumentParser(description="Thesis detector pipeline")

    parser.add_argument("--skip-hog-train",  action="store_true")
    parser.add_argument("--skip-yolo-train", action="store_true")
    parser.add_argument("--eval-only",       action="store_true")

    parser.add_argument("--skip-hog-eval",   action="store_true")
    parser.add_argument("--skip-yolo-eval",  action="store_true")
    parser.add_argument("--skip-frcnn-eval", action="store_true")
    parser.add_argument("--skip-ssd-eval",   action="store_true")
    parser.add_argument("--skip-yv5-eval",   action="store_true")

    parser.add_argument("--samples", nargs="*", default=[],
                        help="Image paths for detection example figure")
    args = parser.parse_args()

    print(f"\nVOC2007 root:      {config.VOC_ROOT}")
    print(f"VOC2007_test root: {config.VOC_ROOT_TEST}")
    print(f"Output dir:        {config.OUTPUT_DIR}")
    print(f"Models dir:        {config.MODELS_DIR}")
    print(f"Figures dir:       {config.FIGURES_DIR}\n")

    # 1. Train HOG+SVM
    if not args.eval_only and not args.skip_hog_train:
        print("STEP 1/4  Training HOG + SVM ...")
        from classical.hog_svm_detector import train as train_hog
        train_hog(voc_root=config.VOC_ROOT)
    else:
        print("STEP 1/4  HOG training skipped")

    # 2. Train YOLOv8n
    if not args.eval_only and not args.skip_yolo_train:
        print("\nSTEP 2/4  Training YOLOv8n ...")
        from deep_learning.yolo_detector import train as train_yolo
        train_yolo(voc_root=config.VOC_ROOT, voc_root_test=config.VOC_ROOT_TEST)
    else:
        print("STEP 2/4  YOLOv8n training skipped")

    # 3. Evaluate all detectors
    print("\nSTEP 3/4  Evaluating all detectors ...")
    from run_evaluation import (
        evaluate_hog_svm, evaluate_yolo,
        evaluate_faster_rcnn, evaluate_ssd, evaluate_yolov5,
        combine_results, persist_to_database,
    )

    voc_test = config.VOC_ROOT_TEST

    hog_res   = evaluate_hog_svm(voc_root_test=voc_test)     if not args.skip_hog_eval   else None
    yolo_res  = evaluate_yolo(voc_root_test=voc_test)         if not args.skip_yolo_eval  else None
    frcnn_res = evaluate_faster_rcnn(voc_root_test=voc_test)  if not args.skip_frcnn_eval else None
    ssd_res   = evaluate_ssd(voc_root_test=voc_test)          if not args.skip_ssd_eval   else None
    yv5_res   = evaluate_yolov5(voc_root_test=voc_test)       if not args.skip_yv5_eval   else None

    combined = combine_results(hog_res, yolo_res, frcnn_res, ssd_res, yv5_res)
    persist_to_database(combined)

    # 4. Generate figures
    print("\nSTEP 4/4  Generating thesis figures ...")
    from visualization.thesis_figures import (
        load_results, fig_map_comparison, fig_per_class_ap,
        fig_speed_accuracy, fig_comparison_table,
        fig_pr_curves, fig_timing_distribution, fig_detection_examples,
    )

    results = load_results()
    fig_map_comparison(results)
    fig_per_class_ap(results)
    fig_speed_accuracy(results)
    fig_comparison_table(results)
    fig_pr_curves(results)
    fig_timing_distribution(results)

    if args.samples:
        hog_path  = os.path.join(config.MODELS_DIR, "hog_svm_classifiers.pkl")
        yolo_path = os.path.join(config.MODELS_DIR, "yolov8n_voc2007", "weights", "best.pt")
        fig_detection_examples(args.samples, hog_path, yolo_path)

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print(f"  Figures: {config.FIGURES_DIR}/")
    print(f"  Results: {config.OUTPUT_DIR}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()