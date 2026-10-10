#!/usr/bin/env python3
"""Detection datasets on the SAME group-stratified folds as the classifiers (folds.json, rep 0).

Images: 640px raw proxy (HAT area-downsampled), boxes from the original Roboflow labels.
Per fold k writes /tmp/data/det/fold{k}/
  yolo/{train,valid,test}/{images,labels}  + data.yaml     (Ultralytics)
  coco/{train,valid,test}/ images + _annotations.coco.json (RF-DETR, also used as GT for evaluation)
train = every image (incl. Roboflow variants) of training groups; valid/test = primary image of each group.
"""
import json, os, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASS_NAMES, parse_boxes
from prep_crops import box_mask, pipeline

SRC, CROPS, RES = Path("/tmp/data/toothfairy"), Path("/tmp/data/crops"), 640
# DET_ROI=1: images processed with the original pipeline, whose CLAHE/rescale ROI is the ground-truth box
# (reproduces the previous runs' setting, to measure how much that leaks the box location)
ROI = os.environ.get("DET_ROI") == "1"
OUT = Path("/tmp/data/det_roi" if ROI else "/tmp/data/det")
meta = pd.read_csv(CROPS / "meta.csv", index_col="idx")
folds = [f for f in json.load(open(CROPS / "folds.json"))["folds"] if f["rep"] == 0]
img_dir = OUT / "images"; img_dir.mkdir(parents=True, exist_ok=True)
split_of = meta.drop_duplicates("image").set_index("image").split.to_dict()
boxes = {}
for name, sp in split_of.items():
    dst = img_dir / name
    if not dst.exists():
        im = cv2.resize(cv2.imread(str(SRC / sp / "images" / name), cv2.IMREAD_GRAYSCALE), (RES, RES), interpolation=cv2.INTER_AREA)
        if ROI:
            im = pipeline(im, box_mask(parse_boxes(SRC / sp / "labels" / (Path(name).stem + ".txt"))))
        cv2.imwrite(str(dst), im)
    boxes[name] = parse_boxes(SRC / sp / "labels" / (Path(name).stem + ".txt"))
print("images", len(boxes), flush=True)


def link(src, dst):
    if not dst.exists():
        os.symlink(src, dst)


for fd in folds:
    root = OUT / f"fold{fd['fold']}"
    for part, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        names = sorted(meta.loc[fd[key], "image"].unique())
        yi, yl, cd = root / "yolo" / part / "images", root / "yolo" / part / "labels", root / "coco" / part
        for d in (yi, yl, cd): d.mkdir(parents=True, exist_ok=True)
        coco = dict(images=[], annotations=[], categories=[{"id": i, "name": n, "supercategory": "implant"} for i, n in enumerate(CLASS_NAMES)])
        aid = 0
        for iid, n in enumerate(names):
            link(img_dir / n, yi / n); link(img_dir / n, cd / n)
            (yl / (Path(n).stem + ".txt")).write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes[n]))
            coco["images"].append(dict(id=iid, file_name=n, width=RES, height=RES))
            for c, x, y, w, h in boxes[n]:
                coco["annotations"].append(dict(id=aid, image_id=iid, category_id=c, iscrowd=0,
                                                bbox=[(x - w / 2) * RES, (y - h / 2) * RES, w * RES, h * RES], area=w * h * RES * RES))
                aid += 1
        json.dump(coco, open(cd / "_annotations.coco.json", "w"))
        print(f"fold{fd['fold']} {part}: {len(names)} images, boxes per class {np.bincount([a['category_id'] for a in coco['annotations']], minlength=3).tolist()}", flush=True)
    (root / "yolo" / "data.yaml").write_text(f"path: {root / 'yolo'}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n" +
                                             "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
print("DONE")
