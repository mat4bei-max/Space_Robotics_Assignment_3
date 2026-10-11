#!/usr/bin/env python3
"""Create depth labels from RGB labels.

Reads YOLO labels from labels/, evaluates the depth at the centre of each
bounding box in raw_depth_images/ (pixel values in metres) and writes only the
boxes with depth <= MAX_DEPTH_M to depth_labels/.
"""
from pathlib import Path

import cv2
import numpy as np

MAX_DEPTH_M = 12  # boxes whose centre depth exceeds this are removed
PATCH_RADIUS = 2   # median over (2r+1)^2 pixels around the centre; 0 = single pixel
DROP_INVALID = True  # depth 0 / NaN (no measurement) -> remove box

DATA_DIR = Path(__file__).resolve().parent
LABEL_DIR = DATA_DIR / "labels"
DEPTH_DIR = DATA_DIR / "raw_depth_images"
OUT_DIR = DATA_DIR / "depth_labels"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".exr")


def find_depth_image(stem):
    for ext in IMG_EXTS:
        p = DEPTH_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def center_depth(depth, xc, yc):
    h, w = depth.shape[:2]
    cx = min(max(int(round(xc * w - 0.5)), 0), w - 1)
    cy = min(max(int(round(yc * h - 0.5)), 0), h - 1)
    r = PATCH_RADIUS
    patch = depth[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1].astype(np.float64)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    kept_total = removed_total = 0
    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        depth_path = find_depth_image(label_path.stem)
        if depth_path is None:
            print(f"[skip] no depth image for {label_path.name}")
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"[skip] cannot read {depth_path.name}")
            continue
        if depth.ndim == 3:
            depth = depth[..., 0]

        kept = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            d = center_depth(depth, float(parts[1]), float(parts[2]))
            if d is None:
                remove = DROP_INVALID
            else:
                remove = d > MAX_DEPTH_M
            if remove:
                removed_total += 1
            else:
                kept.append(line.strip())
                kept_total += 1
        (OUT_DIR / label_path.name).write_text("\n".join(kept) + ("\n" if kept else ""))

    print(f"kept {kept_total}, removed {removed_total} boxes (max depth {MAX_DEPTH_M} m)")


if __name__ == "__main__":
    main()
