#!/usr/bin/env python3
"""Build leakage-free crop tables for three input versions.

Uses the *original* Roboflow images/labels only (no synthetic class-balancing copies).
  hat      : 2560px HAT-enhanced image, crop taken directly
  raw      : HAT image area-downsampled to 640px (proxy for the original Roboflow 640px image)
  pipeline : raw + the authors' sharpen/CLAHE(ROI)/unsharp/percentile pipeline (dental_xray_pipeline_v4)
Also stores a dHash per source image for near-duplicate detection.
"""
import argparse, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASS_NAMES, crop_box, letterbox, parse_boxes, source_id

RES = 640


def feather(mask, sigma=2.0):
    m = (mask > 0).astype(np.float32)
    if m.max() == 0:
        return m
    a = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(a / max(a.max(), 1e-6), 0, 1)


def box_mask(boxes, W=RES, H=RES):
    mask = np.zeros((H, W), np.uint8)
    for _, cx, cy, w, h in boxes:  # boxes in place of original polygons (as the classifier crops are boxes anyway)
        x1, y1 = max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H))
        x2, y2 = min(W - 1, int((cx + w / 2) * W)), min(H - 1, int((cy + h / 2) * H))
        mask[y1:y2 + 1, x1:x2 + 1] = 255
    return mask


def pipeline(img, mask):
    """Same operations/parameters as dental_xray_pipeline_v4.run_pipeline."""
    blur = cv2.GaussianBlur(img, (0, 0), 1.1)
    sharp = cv2.addWeighted(img, 1.9, blur, -0.9, 0)
    blur3 = cv2.GaussianBlur(sharp, (3, 3), 0)
    alpha = feather(mask)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(blur3)
    blend = (alpha * cl + (1 - alpha) * blur3).astype(np.uint8)
    g = cv2.GaussianBlur(blend, (0, 0), 1.0)
    un = cv2.addWeighted(blend, 2.2, g, -1.2, 0).astype(np.float32)
    mb = mask > 0
    if mb.any():
        lo, hi = np.percentile(un[mb], 2.0), np.percentile(un[mb], 98.0)
        if hi > lo:
            sc = np.clip((un - lo) / (hi - lo), 0, 1)
            fin = mb * sc + (~mb) * (un / 255.0)
        else:
            fin = un / 255.0
    else:
        fin = un / 255.0
    return (fin * 255).astype(np.uint8)


def dhash(img, n=16):
    s = cv2.resize(img, (n + 1, n), interpolation=cv2.INTER_AREA)
    return np.packbits((s[:, 1:] > s[:, :-1]).flatten())


def work(job):
    split, img_path, lbl_path = job
    boxes = parse_boxes(Path(lbl_path))
    if not boxes:
        return []
    hat = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    raw = cv2.resize(hat, (RES, RES), interpolation=cv2.INTER_AREA)
    pipe = pipeline(raw, box_mask(boxes))
    name = Path(img_path).name
    rows = []
    for i, b in enumerate(boxes):
        rows.append(dict(split=split, image=name, group=source_id(name), box_idx=i, brand=b[0],
                         hat=letterbox(crop_box(hat, b[1:])), raw=letterbox(crop_box(raw, b[1:])),
                         pipeline=letterbox(crop_box(pipe, b[1:])), dh=dhash(raw)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/data/toothfairy")
    ap.add_argument("--out", default="/tmp/data/crops")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(sp, str(p), str(src / sp / "labels" / (p.stem + ".txt")))
            for sp in ("train", "valid", "test") for p in sorted((src / sp / "images").glob("*.png"))]
    rows = []
    with Pool(4) as pool:
        for k, r in enumerate(pool.imap(work, jobs, chunksize=4), 1):
            rows += r
            if k % 100 == 0:
                print(f"{k}/{len(jobs)} images, {len(rows)} crops", flush=True)
    for v in ("raw", "pipeline", "hat"):
        np.save(out / f"{v}.npy", np.stack([r[v] for r in rows]))
    meta = pd.DataFrame([{k: r[k] for k in ("split", "image", "group", "box_idx", "brand")} for r in rows])
    meta.to_csv(out / "meta.csv", index_label="idx")
    np.save(out / "dhash.npy", np.stack([r["dh"] for r in rows]))
    print(meta.groupby(["split", "brand"]).size().unstack(fill_value=0))
    print("groups per brand:", meta.groupby("brand").group.nunique().to_dict(), "DONE")
