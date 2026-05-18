"""
Inference timing and peak RAM measurement.
Works with any detector that accepts a numpy BGR image.
"""
import time
import os
import gc
import numpy as np
import cv2
import psutil


def _current_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def benchmark_detector(
    detect_fn,
    image_paths: list[str],
    n_images: int = 100,
    warmup: int = 3,
    label: str = "detector",
) -> dict:
    """
    Time a detector over n_images images.

    detect_fn : callable(img_bgr: np.ndarray) -> list[dict]
    Returns dict with timing and memory stats.
    """
    paths = image_paths[:n_images + warmup]
    if len(paths) < n_images:
        print(f"WARNING: only {len(paths)} images available")

    for p in paths[:warmup]:
        img = cv2.imread(p)
        if img is not None:
            detect_fn(img)
    gc.collect()

    times_ms = []
    ram_before = _current_ram_mb()

    for p in paths[warmup: warmup + n_images]:
        img = cv2.imread(p)
        if img is None:
            continue
        t0 = time.perf_counter()
        detect_fn(img)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    ram_after = _current_ram_mb()
    times_ms  = np.array(times_ms)

    stats = {
        "label":        label,
        "n_images":     len(times_ms),
        "mean_ms":      float(np.mean(times_ms)),
        "median_ms":    float(np.median(times_ms)),
        "std_ms":       float(np.std(times_ms)),
        "min_ms":       float(np.min(times_ms)),
        "max_ms":       float(np.max(times_ms)),
        "fps":          float(1000.0 / np.mean(times_ms)),
        "ram_delta_mb": float(ram_after - ram_before),
        "times_ms":     times_ms.tolist(),
    }

    print(f"\n[Benchmark] {label}")
    print(f"  Mean latency : {stats['mean_ms']:.1f} ms/img")
    print(f"  Median       : {stats['median_ms']:.1f} ms/img")
    print(f"  Std          : {stats['std_ms']:.1f} ms")
    print(f"  FPS          : {stats['fps']:.2f}")
    print(f"  RAM delta    : {stats['ram_delta_mb']:.1f} MB")

    return stats


def get_model_size_mb(path: str) -> float:
    """Return file size in MB."""
    return os.path.getsize(path) / 1024**2 if os.path.exists(path) else 0.0
