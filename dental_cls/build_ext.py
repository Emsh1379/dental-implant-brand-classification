#!/usr/bin/env python3
"""crops_ext: original-image crops (rows 0..3275) + external pseudo-labelled candidates as TRAINING data only, in every fold.

usage: python build_ext.py <code dir> <base crops dir (raw_orig.npy, meta.csv, folds.json; extra rows >= 3276 are dropped)> <out dir>
                           <candidates.csv>:<crop dir>:<brand> [...]
Candidates with a non-empty visual_reject column are skipped. Validation and test sets are unchanged.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

CODE, BASE, OUT = map(Path, sys.argv[1:4]); SPECS = [s.split(":") for s in sys.argv[4:]]
sys.path.insert(0, str(CODE))
from common import letterbox
NAMES = {"Bego": 0, "Bicon": 1, "ITI": 2}
OUT.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(BASE / "meta.csv").iloc[:3276]
X0 = np.load(BASE / "raw_orig.npy", mmap_mode="r")[:3276]
rows, X = [], []
for csvp, cdir, brand in SPECS:
    d = pd.read_csv(csvp)
    if "visual_reject" in d: d = d[d.visual_reject.fillna("") == ""]
    for f in d.file:
        X.append(letterbox(cv2.imread(f"{cdir}/{f}", cv2.IMREAD_GRAYSCALE)))
        rows.append(dict(split="external", image=f, group=Path(csvp).parent.name + ":" + f.rsplit("_", 1)[0], box_idx=0, brand=NAMES[brand], source=Path(csvp).parent.name))
new = pd.DataFrame(rows); new.insert(0, "idx", np.arange(3276, 3276 + len(new)))
pd.concat([meta, new.drop(columns="source")], ignore_index=True).to_csv(OUT / "meta.csv", index=False)
new.to_csv(OUT / "extra_rows.csv", index=False)
np.save(OUT / "raw_orig.npy", np.concatenate([np.asarray(X0), np.stack(X)]))
F = json.load(open(BASE / "folds.json"))
for f in F["folds"]:
    f["train"] = [i for i in f["train"] if i < 3276] + new.idx.tolist()
F.pop("added_train_only", None); F["added_external_train_only"] = dict(idx=new.idx.tolist())
json.dump(F, open(OUT / "folds.json", "w"))
print(new.groupby(["source", "brand"]).size().to_string())
print("total", 3276 + len(new), "| train sizes rep0", [len(f["train"]) for f in F["folds"] if f["rep"] == 0])
