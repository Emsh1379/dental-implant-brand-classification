#!/usr/bin/env python3
"""crops_trainplus: original-image crops + extra Bego images used as TRAINING data only, in every fold.

usage: python -I build_trainplus.py <code dir> <crops dir> <out dir> <labelled dir> [<labelled dir> ...]
Each labelled dir has images/ and labels/ (YOLO, class 0 = Bego). Rows 0..3275, validation and test sets are unchanged,
so results are directly comparable with the raw_orig runs on the same 1,590 test implants. Writes raw_orig.npy, meta.csv, folds.json.
"""
import json, re, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

CODE, CROPS, OUT = map(Path, sys.argv[1:4]); EXTRA = [Path(p) for p in sys.argv[4:]]
sys.path.insert(0, str(CODE))
from common import crop_box, letterbox, parse_boxes

OUT.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(CROPS / "meta.csv")
rows, X = [], []
for d in EXTRA:
    for p in sorted((d / "images").glob("*")):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        for k, b in enumerate(parse_boxes(d / "labels" / (p.stem + ".txt"))):
            assert b[0] == 0
            rows.append(dict(split="extra", image=p.name, group=re.split(r"_(?:jpg|jpeg|png)\.rf\.", p.name)[0], box_idx=k, brand=0, source=d.name))
            X.append(letterbox(crop_box(g, b[1:])))
new = pd.DataFrame(rows); new.insert(0, "idx", np.arange(len(meta), len(meta) + len(new)))
pd.concat([meta, new.drop(columns="source")], ignore_index=True).to_csv(OUT / "meta.csv", index=False)
new.to_csv(OUT / "extra_rows.csv", index=False)
np.save(OUT / "raw_orig.npy", np.concatenate([np.load(CROPS / "raw_orig.npy"), np.stack(X)]))
F = json.load(open(CROPS / "folds.json"))
for f in F["folds"]: f["train"] += new.idx.tolist()
F["added_train_only"] = dict(idx=new.idx.tolist(), sources=sorted(new.source.unique().tolist()))
json.dump(F, open(OUT / "folds.json", "w"))
print(new.groupby("source").agg(images=("image", "nunique"), crops=("idx", "size")).to_string())
print("total crops", len(meta) + len(new), "| Bego crops", int((meta.brand == 0).sum()), "->", int((meta.brand == 0).sum()) + len(new))
