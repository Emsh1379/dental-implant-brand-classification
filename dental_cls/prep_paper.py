#!/usr/bin/env python3
"""Inputs for the Tiryaki et al. (J Prosthet Dent 2023) pipeline on our dataset.

usage: ROBOFLOW_API_KEY=... python prep_paper.py --crops <dir with meta.csv> --match <raw_orig_match.csv> --out <dir>
* paper.npy: every crop of meta.csv (same row order) cut from the ORIGINAL Roboflow image (project dental-implant-nyw2h/-dental-implant v1)
  exactly at the box (no margin) and stretched to 224 x 224, as in the paper ("cropped ... resized to 224x224").
* angles.csv: implant tilt from the vertical image axis, in degrees (-90, 90], positive = top of the implant leans to the right.
  The paper measured it by hand with a protractor; here it is automatic: the metal inside the box (threshold halfway between Otsu and the
  99.5th percentile) -> largest component -> principal axis (PCA of its pixel coordinates). elong = ratio of the axis lengths (low = unreliable).
* angle_check.jpg: 40 crops with the estimated axis drawn, for a visual check.
"""
import argparse, os, sys, zipfile
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import crop_box, parse_boxes

ap = argparse.ArgumentParser()
for k in ("crops", "match", "out"): ap.add_argument("--" + k)
ap.add_argument("--rf", default="/tmp/data/rf/orig_v1_yolo"); ap.add_argument("--labels", default="/tmp/data/hat_labels")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if not os.path.exists(a.rf):
    from roboflow import Roboflow
    Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace("dental-implant-nyw2h").project("-dental-implant").version(1).download("yolov8", location=a.rf)
if not os.path.exists(a.labels):
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open("datasets/Mesh2001/Dental_implant/Teeth-Segmentation-ToothFairy-1-enahnced.zip", "rb", block_size=2 ** 22) as f:
        z = zipfile.ZipFile(f)
        for n in z.namelist():
            if n.endswith(".txt") and "/labels/" in n:
                d = Path(a.labels) / n.split("/")[-3]; d.mkdir(parents=True, exist_ok=True); (d / Path(n).name).write_bytes(z.read(n))
meta = pd.read_csv(f"{a.crops}/meta.csv").iloc[:3276]
orig = pd.read_csv(a.match).set_index("idx").original
paths = {Path(p).name: p for p in Path(a.rf).glob("*/images/*")}


def tilt(g, b):
    H, W = g.shape; cx, cy, bw, bh = b
    x1, y1, x2, y2 = int(max(0, (cx - bw / 2) * W)), int(max(0, (cy - bh / 2) * H)), int(min(W, (cx + bw / 2) * W)), int(min(H, (cy + bh / 2) * H))
    roi = cv2.GaussianBlur(g[y1:y2, x1:x2], (5, 5), 0)
    t, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx((roi > 0.5 * (t + np.percentile(roi, 99.5))).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return np.nan, np.nan, None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); ys, xs = np.nonzero(lab == i)
    pts = np.stack([xs - xs.mean(), ys - ys.mean()]); ev, evec = np.linalg.eigh(np.cov(pts))
    vx, vy = evec[:, 1]
    if vy < 0: vx, vy = -vx, -vy          # axis pointing down the image
    ang = float(np.degrees(np.arctan2(-vx, vy)))  # 0 = vertical; positive = top leans right
    if ang <= -90: ang += 180
    if ang > 90: ang -= 180
    return ang, float(np.sqrt(ev[1] / max(ev[0], 1e-6))), (x1 + xs.mean(), y1 + ys.mean(), vx, vy)


X = np.zeros((len(meta), 224, 224), np.uint8); rows = []; last = None
for r in meta.itertuples():
    if r.image != last:
        g = cv2.imread(paths[orig[r.idx]], cv2.IMREAD_GRAYSCALE)
        boxes = parse_boxes(Path(a.labels) / r.split / (Path(r.image).stem + ".txt")); last = r.image
    b = boxes[r.box_idx]; assert b[0] == r.brand
    X[r.idx] = cv2.resize(crop_box(g, b[1:], margin=0.0), (224, 224), interpolation=cv2.INTER_AREA)
    ang, el, ax = tilt(g, b[1:]); rows.append(dict(idx=r.idx, image=r.image, brand=r.brand, angle=ang, elong=el))
    if r.idx < 400 and r.idx % 10 == 0 and ax is not None:  # sample for the visual check
        H, W = g.shape; v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); cx, cy, vx, vy = ax; L = b[4] * H * 0.6
        cv2.line(v, (int(cx - vx * L), int(cy - vy * L)), (int(cx + vx * L), int(cy + vy * L)), (0, 0, 255), 2)
        c = cv2.resize(crop_box(v, b[1:], margin=0.3), (160, 160)); cv2.putText(c, f"{ang:+.0f}", (3, 15), 0, 0.5, (0, 255, 0), 1); rows[-1]["_tile"] = c
np.save(out / "paper.npy", X)
A = pd.DataFrame([{k: v for k, v in r.items() if k != "_tile"} for r in rows]); A.to_csv(out / "angles.csv", index=False)
tiles = [r["_tile"] for r in rows if "_tile" in r][:40]
cv2.imwrite(str(out / "angle_check.jpg"), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 40, 8)]))
print("crops", X.shape, "| angle: valid", int(A.angle.notna().sum()), "median |angle| %.1f, sd %.1f, elong median %.1f, elong<2: %d" % (
    A.angle.abs().median(), A.angle.std(), A.elong.median(), int((A.elong < 2).sum())))
print(np.histogram(A.angle.dropna(), bins=[-90, -45, -20, -10, -5, 5, 10, 20, 45, 90])[0])
