#!/usr/bin/env python3
"""Crops that most (model, version) configurations get wrong – candidates for label review. CPU only."""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", default=["/kaggle/working/results", "/kaggle/working/results2"])
ap.add_argument("--meta", default="/tmp/data/crops/meta.csv")
ap.add_argument("--out", default="/kaggle/working/report3")
ap.add_argument("--min_frac", type=float, default=0.5)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
CL = ["Bego", "Bicon", "ITI"]
R = [r for r in (json.load(open(f)) for d in a.results for f in sorted(glob.glob(f"{d}/*.json"))) if r["rep"] == 0]
P = defaultdict(list)
for r in R:
    for i, y, p in zip(r["test_idx"], r["y"], r["probs"]): P[(i, y)].append(p)
meta = pd.read_csv(a.meta).set_index("idx")
rows = []
for (i, y), p in P.items():
    p = np.array(p); pred = p.argmax(1); wrong = pred != y
    if wrong.mean() < a.min_frac: continue
    top = np.bincount(pred[wrong], minlength=3).argmax()
    rows.append(dict(idx=i, image=meta.image[i], group=meta.group[i], box_idx=meta.box_idx[i], label=CL[y], n_configs=len(p), n_wrong=int(wrong.sum()),
                     usual_prediction=CL[top], mean_p_label=p[:, y].mean(), mean_p_prediction=p[:, top].mean()))
T = pd.DataFrame(rows).sort_values(["n_wrong", "mean_p_label"], ascending=[False, True]); T.to_csv(f"{a.out}/consistent_errors.csv", index=False)
print(T.drop(columns="image").round(3).to_string(index=False))
print(len(T), "of", len(P), "crops wrong in >=", a.min_frac, "of configurations")
