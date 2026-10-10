#!/usr/bin/env python3
"""Second-round input variants (same crop rows/order as prep_crops.py -> meta.csv).

Ablations of the original ROI pipeline (one step removed at a time, 640px, 224 crops):
  ab_no_sharpen, ab_no_blur, ab_no_clahe, ab_no_unsharp, ab_no_rescale
Revised preprocessing (no ground-truth ROI, no hard box edge):
  fixed : non-local-means denoise -> global CLAHE(clip 2, 8x8) -> mild unsharp (sigma 1, amount 0.5)
  ch3   : 3-channel input  R = raw, G = global CLAHE, B = denoised + unsharp(amount 1.0)
Resolution:
  raw384, hat384 : 384px letterboxed crops from the 640px proxy / the 2560px HAT image
"""
import argparse, sys
from multiprocessing import Pool
from pathlib import Path

import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import crop_box, letterbox, parse_boxes, source_id
from prep_crops import RES, box_mask, feather

ABL = ["sharpen", "blur", "clahe", "unsharp", "rescale"]


def roi_pipeline(img, mask, skip=None):
    """dental_xray_pipeline_v4 with one step optionally removed."""
    x = img
    if skip != "sharpen":
        x = cv2.addWeighted(x, 1.9, cv2.GaussianBlur(x, (0, 0), 1.1), -0.9, 0)
    if skip != "blur":
        x = cv2.GaussianBlur(x, (3, 3), 0)
    if skip != "clahe":
        a = feather(mask)
        x = (a * cv2.createCLAHE(2.0, (4, 4)).apply(x) + (1 - a) * x).astype(np.uint8)
    if skip != "unsharp":
        x = cv2.addWeighted(x, 2.2, cv2.GaussianBlur(x, (0, 0), 1.0), -1.2, 0)
    f = x.astype(np.float32); mb = mask > 0
    if skip != "rescale" and mb.any():
        lo, hi = np.percentile(f[mb], 2), np.percentile(f[mb], 98)
        if hi > lo:
            f = np.where(mb, np.clip((f - lo) / (hi - lo), 0, 1) * 255, f)
    return np.clip(f, 0, 255).astype(np.uint8)


def unsharp(x, amount, sigma=1.0):
    return cv2.addWeighted(x, 1 + amount, cv2.GaussianBlur(x, (0, 0), sigma), -amount, 0)


def work(job):
    split, img_path, lbl_path = job
    boxes = parse_boxes(Path(lbl_path))
    if not boxes:
        return []
    hat = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    raw = cv2.resize(hat, (RES, RES), interpolation=cv2.INTER_AREA)
    mask = box_mask(boxes)
    abl = {f"ab_no_{s}": roi_pipeline(raw, mask, s) for s in ABL}
    den = cv2.fastNlMeansDenoising(raw, None, h=5, templateWindowSize=7, searchWindowSize=21)
    clahe_g = cv2.createCLAHE(2.0, (8, 8)).apply(raw)
    fixed = unsharp(cv2.createCLAHE(2.0, (8, 8)).apply(den), 0.5)
    detail = unsharp(den, 1.0)
    name = Path(img_path).name
    rows = []
    for i, b in enumerate(boxes):
        bb = b[1:]
        r = dict(image=name, box_idx=i, fixed=letterbox(crop_box(fixed, bb)),
                 ch3=np.stack([letterbox(crop_box(c, bb)) for c in (raw, clahe_g, detail)]),
                 raw384=letterbox(crop_box(raw, bb), 384), hat384=letterbox(crop_box(hat, bb), 384))
        r.update({k: letterbox(crop_box(v, bb)) for k, v in abl.items()})
        rows.append(r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/data/toothfairy")
    ap.add_argument("--out", default="/tmp/data/crops")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    jobs = [(sp, str(p), str(src / sp / "labels" / (p.stem + ".txt")))
            for sp in ("train", "valid", "test") for p in sorted((src / sp / "images").glob("*.png"))]
    rows = []
    with Pool(4) as pool:
        for k, r in enumerate(pool.imap(work, jobs, chunksize=4), 1):
            rows += r
            if k % 200 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    meta = pd.read_csv(out / "meta.csv", index_col="idx")
    assert len(meta) == len(rows) and all(m == (r["image"], r["box_idx"]) for m, r in
                                          zip(zip(meta.image, meta.box_idx), rows)), "row order mismatch"
    for v in [f"ab_no_{s}" for s in ABL] + ["fixed", "ch3", "raw384", "hat384"]:
        np.save(out / f"{v}.npy", np.stack([r[v] for r in rows]))
    print("DONE", flush=True)
