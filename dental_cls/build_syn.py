#!/usr/bin/env python3
"""crops_syn: original crops + synthetic Bego crops; fold k's synthetic crops (made by the generator trained on fold k's training crops) are added
to fold k's TRAINING set only (in rep 0). Validation and test sets are unchanged.
usage: python build_syn.py <crops dir> <syn dir> <out dir> [brands to add, default 0 (Bego)]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

C, S, O = map(Path, sys.argv[1:4]); BR = [int(b) for b in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]
O.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(C / "meta.csv").iloc[:3276]; X = [np.load(C / "raw_orig.npy")[:3276]]
F = json.load(open(C / "folds.json")); rows = []; n = 3276
for f in [f for f in F["folds"] if f["rep"] == 0]:
    k = f["fold"]; A = np.load(S / f"syn_f{k}.npy"); L = np.load(S / f"syn_f{k}_labels.npy"); keep = np.isin(L, BR)
    A, L = A[keep], L[keep]; idx = list(range(n, n + len(A))); n += len(A); X.append(A)
    rows += [dict(idx=i, split="synthetic", image=f"syn_f{k}_{j}", group=f"syn_f{k}_{j}", box_idx=0, brand=int(l)) for j, (i, l) in enumerate(zip(idx, L))]
    f["train"] = f["train"] + idx
    print(f"fold {k}: +{len(idx)} synthetic training crops {np.bincount(L, minlength=3).tolist()} -> train {len(f['train'])}")
F["added_synthetic_train_only"] = dict(rows=len(rows), brands=BR)
pd.concat([meta, pd.DataFrame(rows)], ignore_index=True).to_csv(O / "meta.csv", index=False)
np.save(O / "raw_orig.npy", np.concatenate(X)); json.dump(F, open(O / "folds.json", "w")); print("total crops", n)
