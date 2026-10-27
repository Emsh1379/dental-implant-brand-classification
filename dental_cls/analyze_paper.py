#!/usr/bin/env python3
"""Results of the Tiryaki et al. pipeline on our dataset (train_paper.py): per-model metrics, majority-vote fusion, per-brand sensitivity /
specificity / F1 as in the paper, grouped vs random split, and angle RMSE (with a trivial baseline).
usage: python analyze_paper.py <results dir> <out dir> [<data dir with angles.csv>]"""
import glob, json, sys, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

R, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); NAMES = ["Bego", "Bicon", "ITI"]
J = [json.load(open(f)) for f in glob.glob(f"{R}/*.json")]
C = [r for r in J if r["task"] == "cls"]; AN = [r for r in J if r["task"] == "angle"]
MODELS = ["vgg16", "vgg19", "resnet50", "resnet101", "googlenet"]


def per_class(y, p):
    cm = confusion_matrix(y, p, labels=[0, 1, 2]); out = {}
    for c in range(3):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        out[NAMES[c]] = dict(sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), f1=2 * tp / max(2 * tp + fp + fn, 1))
    return out


rows, fus = [], []
for split in ("group", "random"):
    for k in range(5):
        rs = {r["model"]: r for r in C if r["split"] == split and r["fold"] == k}
        for m, r in rs.items():
            y = np.array(r["y"]); p = np.array(r["probs"]).argmax(1); pc = per_class(y, p)
            rows.append(dict(split=split, fold=k, model=m, n=len(y), acc=(p == y).mean(), macro_f1=f1_score(y, p, average="macro"),
                             **{f"{b}_{s}": pc[b][s] for b in NAMES for s in ("sens", "spec", "f1")}))
        if len(rs) == 5:  # majority vote (>= 3 of 5); if no class gets 3 votes, the class with the larger summed probability
            y = np.array(rs["vgg16"]["y"]); idx = rs["vgg16"]["test_idx"]
            assert all(r["test_idx"] == idx for r in rs.values())
            P = np.stack([np.array(rs[m]["probs"]) for m in MODELS]); votes = np.stack([(P.argmax(2) == c).sum(0) for c in range(3)], 1)
            p = np.where(votes.max(1) >= 3, votes.argmax(1), P.sum(0).argmax(1)); pc = per_class(y, p)
            rows.append(dict(split=split, fold=k, model="fusion", n=len(y), acc=(p == y).mean(), macro_f1=f1_score(y, p, average="macro"),
                             **{f"{b}_{s}": pc[b][s] for b in NAMES for s in ("sens", "spec", "f1")}))
            fus.append(dict(split=split, fold=k, idx=idx, y=y, p=p))
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/per_fold.csv", index=False)
pd.set_option("display.width", 250)
S = D.groupby(["split", "model"])[["acc", "macro_f1", "Bego_sens", "Bego_spec", "Bego_f1", "Bicon_f1", "ITI_f1"]].agg(["mean", "std"]).round(4)
print(S.to_string()); S.to_csv(f"{OUT}/summary.csv")
print("\nfolds per split/model:", D.groupby(["split", "model"]).size().to_dict())
for split in ("group", "random"):
    f = [x for x in fus if x["split"] == split]
    if len(f) == 5:
        y = np.concatenate([x["y"] for x in f]); p = np.concatenate([x["p"] for x in f])
        print(f"\nfusion, {split} split, pooled over folds: acc {(p == y).mean():.4f}, macro-F1 {f1_score(y, p, average='macro'):.4f}")
        print(pd.DataFrame(confusion_matrix(y, p, labels=[0, 1, 2]), index=[f"true {n}" for n in NAMES], columns=[f"pred {n}" for n in NAMES]).to_string())
if AN:
    A = pd.DataFrame([dict(split=r["split"], fold=r["fold"], n=r["metrics"]["n"], rmse=r["metrics"]["rmse"], mae=r["metrics"]["mae"]) for r in AN])
    base = []
    if len(sys.argv) > 3:  # trivial baseline: always predict the circular (axial) mean angle of all labelled crops
        ang = pd.read_csv(f"{sys.argv[3]}/angles.csv").set_index("idx").angle
        for r in AN:
            t = np.array(r["true"]); allv = np.radians(ang.dropna().values * 2); mu = np.degrees(np.arctan2(np.sin(allv).mean(), np.cos(allv).mean())) / 2
            e = (mu - t + 90) % 180 - 90; base.append(np.sqrt(np.mean(e ** 2)))
        A["baseline_rmse"] = base
    A.to_csv(f"{OUT}/angle.csv", index=False); print("\nangle regression (ResNet-101), degrees, mean over folds:")
    print(A.groupby("split")[[c for c in A if c in ("n", "rmse", "mae", "baseline_rmse")]].mean().round(2).to_string())
    for split in ("group", "random"):  # pooled error distribution
        e = np.concatenate([(np.array(r["pred"]) - np.array(r["true"]) + 90) % 180 - 90 for r in AN if r["split"] == split])
        if len(e): print(f"{split}: n {len(e)}, median |err| {np.median(np.abs(e)):.2f}, within 5 deg {np.mean(np.abs(e) <= 5):.3f}, "
                         f"within 10 deg {np.mean(np.abs(e) <= 10):.3f}, > 20 deg {np.mean(np.abs(e) > 20):.3f}")
