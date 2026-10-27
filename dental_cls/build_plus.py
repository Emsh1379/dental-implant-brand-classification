#!/usr/bin/env python3
"""crops_plus: the original-image crops + the 3 auto-annotated Bego radiographs (18 images) from lamiaa/implant-recognition v3.

usage: python -I build_plus.py <code dir> <crops dir (meta.csv, meta_groups.csv, folds.json, raw_orig*.npy, pipeline_orig.npy)> <bego_extra dir> <out dir> [<lamiaa_v3 dir>]
Rows 0..3275 and every existing fold assignment are unchanged; the new crops are appended (idx 3276+).
Each new source image becomes its own group. Per rep it is the test group of the fold with the fewest Bego test groups
(primary image only, as for every other group); in the other 4 folds all its images are training data. Validation sets are unchanged.
Only the original-image inputs exist for the new images (raw_orig, raw_orig384, pipeline_orig); there is no HAT version of them.
"""
import json, re, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

CODE, CROPS, EXTRA, OUT = map(Path, sys.argv[1:5])
sys.path.insert(0, str(CODE))
from common import crop_box, letterbox, parse_boxes
from prep_crops import box_mask, pipeline

OUT.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(CROPS / "meta_groups.csv")
F = json.load(open(CROPS / "folds.json"))
coco = json.load(open(EXTRA / "_annotations.coco.json"))
split_of = {}  # Roboflow split of each new image, kept in the split column like for the other images
lam = Path(sys.argv[5]) if len(sys.argv) > 5 else None
if lam:
    for sp in ("train", "valid", "test"):
        for i in json.load(open(lam / sp / "_annotations.coco.json"))["images"]: split_of[i["file_name"]] = sp
names = sorted(i["file_name"] for i in coco["images"])
rows, X, X384, P = [], [], [], []
for n in names:
    g = cv2.imread(str(EXTRA / "images" / n), cv2.IMREAD_GRAYSCALE)
    boxes = parse_boxes(EXTRA / "labels" / (Path(n).stem + ".txt")); assert len(boxes) == 1 and boxes[0][0] == 0, n
    pipe = pipeline(g, box_mask(boxes))
    for k, b in enumerate(boxes):
        grp = re.split(r"_(?:jpg|jpeg|png)\.rf\.", n)[0]
        rows.append(dict(split=split_of.get(n, "train"), image=n, group=grp, box_idx=k, brand=0, cgroup=grp))
        X.append(letterbox(crop_box(g, b[1:]))); X384.append(letterbox(crop_box(g, b[1:]), 384)); P.append(letterbox(crop_box(pipe, b[1:])))
new = pd.DataFrame(rows)
new["primary"] = new.image == new.groupby("cgroup").image.transform("min")
new.insert(0, "idx", np.arange(len(meta), len(meta) + len(new)))
plus = pd.concat([meta, new], ignore_index=True)
assert (plus.idx.values == np.arange(len(plus))).all()
plus.to_csv(OUT / "meta_groups.csv", index=False)
plus[["idx", "split", "image", "group", "box_idx", "brand"]].to_csv(OUT / "meta.csv", index=False)
for v, arr in (("raw_orig", X), ("raw_orig384", X384), ("pipeline_orig", P)):
    old = np.load(CROPS / f"{v}.npy"); np.save(OUT / f"{v}.npy", np.concatenate([old, np.stack(arr)]))
groups = sorted(new.cgroup.unique()); assign = {}
for rep in sorted({f["rep"] for f in F["folds"]}):
    fs = [f for f in F["folds"] if f["rep"] == rep]
    nb = {f["fold"]: meta.loc[f["test"]].query("brand == 0").cgroup.nunique() for f in fs}
    for grp in groups:
        k = min(nb, key=lambda x: (nb[x], x)); nb[k] += 1; assign[(rep, grp)] = k
    for f in fs:
        for grp in groups:
            g = new[new.cgroup == grp]
            if assign[(rep, grp)] == f["fold"]: f["test"] += g[g.primary].idx.tolist()
            else: f["train"] += g.idx.tolist()
F["added"] = dict(source="lamiaa/implant-recognition v3 (CC BY 4.0), 18 unannotated images of 3 Bego radiographs; boxes auto-annotated "
                         "(YOLO11n 5-fold consensus, snapped to the implant metal, checked visually)",
                  idx=new.idx.tolist(), test_fold={f"rep{r}:{g}": k for (r, g), k in assign.items()})
json.dump(F, open(OUT / "folds.json", "w"))
print(new.groupby("cgroup").agg(images=("image", "nunique"), primary=("primary", "sum")).to_string())
print("test fold per rep:", {f"rep{r}:{g}": k for (r, g), k in assign.items()})
for f in [f for f in F["folds"] if f["rep"] == 0]:
    t = plus.loc[f["test"]]; print("rep0 fold", f["fold"], "test crops", len(t), "Bego", int((t.brand == 0).sum()), "| train", len(f["train"]), "val", len(f["val"]))
print("crops", len(plus), "Bego", int((plus.brand == 0).sum()), "unique Bego implants (primary)", int(((plus.brand == 0) & plus.primary).sum()))
