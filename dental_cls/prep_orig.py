#!/usr/bin/env python3
"""Crops from the *original* Roboflow images (no HAT super-resolution), aligned row by row with crops/meta.csv.

Source: universe.roboflow.com/dental-implant-nyw2h/-dental-implant, version 1 (2,064 images, 640x640, no augmentation, CC BY 4.0).
That export has the same images, splits and boxes as the HAT set, but different file hashes, so each of our images is matched to
its original inside the same (source image, split) group by crop correlation with the proxy raw crops (one-to-one assignment).
Boxes are taken from the HAT set's label files, so only the pixels differ from raw.npy.

Writes raw_orig.npy, raw_orig384.npy, pipeline_orig.npy and raw_orig_match.csv to --crops.
Needs: ROBOFLOW_API_KEY in the environment, raw.npy + meta.csv in --crops, `pip install roboflow`.
"""
import argparse, collections, glob, os, sys, zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment as lsa

sys.path.insert(0, str(Path(__file__).parent))
from common import crop_box, letterbox, parse_boxes, source_id
from prep_crops import box_mask, pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--crops", default="/tmp/data/crops")
ap.add_argument("--rf", default="/tmp/data/rf/orig_v1_yolo")
ap.add_argument("--labels", default="/tmp/data/hat_labels")
a = ap.parse_args()

if not os.path.exists(a.rf):
    from roboflow import Roboflow
    Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace("dental-implant-nyw2h").project("-dental-implant").version(1).download("yolov8", location=a.rf)

if not os.path.exists(a.labels):  # label files of the HAT set, read from the 11.8 GB zip by range requests (images are not downloaded)
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open("datasets/Mesh2001/Dental_implant/Teeth-Segmentation-ToothFairy-1-enahnced.zip", "rb", block_size=2 ** 22) as f:
        z = zipfile.ZipFile(f)
        for n in z.namelist():
            if n.endswith(".txt") and "/labels/" in n:
                d = Path(a.labels) / n.split("/")[-3]; d.mkdir(parents=True, exist_ok=True)
                (d / Path(n).name).write_bytes(z.read(n))

m = pd.read_csv(f"{a.crops}/meta.csv")
raw = np.load(f"{a.crops}/raw.npy")
MAP = {0: 1, 1: 0, 2: 2}  # class id in the original export -> our brand id (Bego 0, Bicon 1, ITI 2)


def grey(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE)


def z(x):
    x = x.astype(np.float32).ravel(); x -= x.mean()
    return x / (np.linalg.norm(x) + 1e-6)


# 1. match every image of ours to one original image
cand = collections.defaultdict(list)
for p in sorted(glob.glob(f"{a.rf}/*/images/*")):
    boxes = parse_boxes(Path(p.replace("/images/", "/labels/")).with_suffix(".txt"))
    cand[(source_id(Path(p).name), Path(p).parts[-3])].append((p, boxes, [z(letterbox(crop_box(grey(p), b[1:]))) for b in boxes]))
match, score = {}, {}
for (grp, sp), gg in m.groupby(["group", "split"], sort=False):
    ours = [(im, g) for im, g in gg.groupby("image", sort=False)]; cs = cand[(grp, sp)]
    assert len(ours) == len(cs), (grp, sp, len(ours), len(cs))
    M = np.full((len(ours), len(cs)), -9.0)
    for i, (im, g) in enumerate(ours):
        A = [z(raw[k]) for k in g.idx]
        for j, (p, boxes, crops) in enumerate(cs):
            if len(crops) == len(A):
                S = np.array([[x @ c for c in crops] for x in A]); ri, ci = lsa(-S); M[i, j] = S[ri, ci].mean()
    for i, j in zip(*lsa(-M)):
        assert M[i, j] > -9, (grp, sp)
        match[ours[i][0]] = cs[j][0]; score[ours[i][0]] = M[i, j]

# 2. crops from the original pixels with our boxes
raw_o, raw_o384, pipe_o = np.zeros_like(raw), np.zeros((len(m), 384, 384), np.uint8), np.zeros_like(raw)
cors, last = [], None
for i, r in enumerate(m.itertuples()):
    if r.image != last:
        boxes = parse_boxes(Path(a.labels) / r.split / (Path(r.image).stem + ".txt"))
        theirs = parse_boxes(Path(match[r.image].replace("/images/", "/labels/")).with_suffix(".txt"))
        assert len(boxes) == len(theirs) and sorted(b[0] for b in boxes) == sorted(MAP[t[0]] for t in theirs), r.image
        img = grey(match[r.image]); pipe = pipeline(img, box_mask(boxes)); last = r.image
    b = boxes[r.box_idx]; assert b[0] == r.brand
    raw_o[i], raw_o384[i], pipe_o[i] = letterbox(crop_box(img, b[1:])), letterbox(crop_box(img, b[1:]), 384), letterbox(crop_box(pipe, b[1:]))
    cors.append(float(z(raw_o[i]) @ z(raw[i])))
cors = np.array(cors)
print(f"{len(m)} crops from {len(match)} original images; correlation with proxy raw: median {np.median(cors):.4f}, min {cors.min():.4f}, "
      f"below 0.9: {(cors < .9).sum()}")
assert (cors > 0.9).all(), "a crop does not match its proxy - check the matching"
np.save(f"{a.crops}/raw_orig.npy", raw_o); np.save(f"{a.crops}/raw_orig384.npy", raw_o384); np.save(f"{a.crops}/pipeline_orig.npy", pipe_o)
pd.DataFrame(dict(idx=m.idx, image=m.image, original=[Path(match[x]).name for x in m.image], image_match_score=[score[x] for x in m.image],
                  crop_corr=cors)).to_csv(f"{a.crops}/raw_orig_match.csv", index=False)
print("DONE")
