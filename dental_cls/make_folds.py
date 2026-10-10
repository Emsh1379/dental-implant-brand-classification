#!/usr/bin/env python3
"""Group-aware stratified folds. Groups = Roboflow source image, merged when dHash near-duplicates
(same radiograph re-exported/cropped under another name) so no group straddles train/test."""
import argparse, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--crops", default="/tmp/data/crops")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--dist", type=int, default=12, help="max Hamming distance (of 256 bits) to call two images near-duplicates")
a = ap.parse_args()
meta = pd.read_csv(f"{a.crops}/meta.csv", index_col="idx")
dh = np.unpackbits(np.load(f"{a.crops}/dhash.npy"), axis=1)

# one hash per image, then union-find over near-duplicate images
imgs = meta.drop_duplicates("image")
pos = imgs.index.values
H = dh[pos].astype(np.int8)
D = (H[:, None, :] != H[None, :, :]).sum(-1)
np.fill_diagonal(D, 999)
parent = {g: g for g in meta.group.unique()}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
pairs = []
g = imgs.group.values; sp = imgs.split.values
for i, j in zip(*np.where(np.triu(D <= a.dist))):
    if g[i] != g[j]:
        pairs.append((imgs.image.values[i], imgs.image.values[j], int(D[i, j]), sp[i], sp[j]))
        parent[find(g[i])] = find(g[j])
meta["cgroup"] = meta.group.map(find)
cross = [p for p in pairs if p[3] != p[4]]
print(f"{len(pairs)} near-duplicate image pairs (dist<={a.dist}); {len(cross)} cross the ORIGINAL train/valid/test split")
for p in sorted(cross, key=lambda p: p[2])[:15]: print("  ", p)
print("source groups:", meta.group.nunique(), "-> merged groups:", meta.cgroup.nunique())

# primary variant: the first image (sorted by name) of each group is used for evaluation
first = meta.groupby("cgroup").image.transform(lambda s: sorted(s.unique())[0])
meta["primary"] = meta.image == first
out = {"pairs": pairs, "folds": []}
y = meta.brand.values
for rep in range(a.reps):
    sgkf = StratifiedGroupKFold(a.k, shuffle=True, random_state=rep)
    for f, (tr, te) in enumerate(sgkf.split(meta, y, meta.cgroup)):
        trm = meta.iloc[tr]
        # validation = ~1/6 of the training groups (primary variant only), stratified
        vsp = StratifiedGroupKFold(6, shuffle=True, random_state=100 + rep)
        tr2, va = next(vsp.split(trm, trm.brand, trm.cgroup))
        va_idx = trm.index[va][trm.primary.values[va]]
        tr_idx = trm.index[tr2]
        te_idx = meta.index[te][meta.primary.values[te]]
        out["folds"].append(dict(rep=rep, fold=f, train=tr_idx.tolist(), val=va_idx.tolist(), test=te_idx.tolist()))
        c = lambda ix: np.bincount(meta.loc[ix, "brand"], minlength=3).tolist()
        print(f"rep{rep} fold{f}: train {c(tr_idx)} val {c(va_idx)} test {c(te_idx)}")
    assert not (set(meta.cgroup[out['folds'][-1]['train']]) & set(meta.cgroup[out['folds'][-1]['test']]))
json.dump(out, open(f"{a.crops}/folds.json", "w"))
meta.to_csv(f"{a.crops}/meta_groups.csv")
