#!/usr/bin/env python3
"""Rebuild the benchmark crops from the public Roboflow export (no other files needed).

1. Download the source images (CC BY 4.0):  Roboflow project "dental-implant-nyw2h/-dental-implant", version 1, format "yolov8"
       pip install roboflow
       python -c "from roboflow import Roboflow; Roboflow(api_key='YOUR_KEY').workspace('dental-implant-nyw2h').project('-dental-implant').version(1).download('yolov8', location='roboflow_v1')"
2. python build_crops.py roboflow_v1 crops.npy  [--png out_dir]

Each crop = the box in crops.csv (cx, cy, w, h, normalised) enlarged by 15 % per side, cut from the grey-scale original image, padded to a square
with the median border value and resized to 224 x 224 (INTER_AREA when shrinking, INTER_CUBIC when enlarging). Row i of crops.npy = idx i of crops.csv.
"""
import argparse, glob, hashlib
from pathlib import Path
import cv2, numpy as np, pandas as pd

MARGIN, SIZE = 0.15, 224


def crop_box(img, box, margin=MARGIN):
    h, w = img.shape[:2]; cx, cy, bw, bh = box
    x1 = max(0, int(round((cx - bw / 2) * w - margin * bw * w))); y1 = max(0, int(round((cy - bh / 2) * h - margin * bh * h)))
    x2 = min(w - 1, int(round((cx + bw / 2) * w + margin * bw * w))); y2 = min(h - 1, int(round((cy + bh / 2) * h + margin * bh * h)))
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    return img[y1:y2 + 1, x1:x2 + 1]


def letterbox(img, size=SIZE):
    h, w = img.shape[:2]; s = max(h, w)
    pad = int(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))
    canvas = np.full((s, s), pad, np.uint8); canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("roboflow_dir"); ap.add_argument("out"); ap.add_argument("--png")
    ap.add_argument("--meta", default=str(Path(__file__).with_name("crops.csv"))); a = ap.parse_args()
    meta = pd.read_csv(a.meta); files = {Path(p).name: p for p in glob.glob(f"{a.roboflow_dir}/*/images/*")}
    missing = sorted(set(meta.original_file) - set(files)); assert not missing, f"{len(missing)} source images not found, e.g. {missing[:3]}"
    X = np.zeros((len(meta), SIZE, SIZE), np.uint8); cache = {}
    for r in meta.itertuples():
        if r.original_file not in cache: cache = {r.original_file: cv2.imread(files[r.original_file], cv2.IMREAD_GRAYSCALE)}
        X[r.idx] = letterbox(crop_box(cache[r.original_file], (r.cx, r.cy, r.w, r.h)))
        if a.png: Path(a.png).mkdir(parents=True, exist_ok=True); cv2.imwrite(f"{a.png}/{r.idx:04d}.png", X[r.idx])
    np.save(a.out, X); print(f"{len(X)} crops -> {a.out}; md5 of the array bytes: {hashlib.md5(X.tobytes()).hexdigest()}")
