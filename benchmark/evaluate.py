#!/usr/bin/env python3
"""Score predictions on the leakage-free implant-brand benchmark.

usage: python evaluate.py predictions.csv [--compare baseline.csv] [--folds folds.json] [--meta crops.csv]

predictions.csv: one row per test crop with columns  idx, p_Bego, p_Bicon, p_ITI   (probabilities; or a column "pred" with 0/1/2 or a brand name)
  * idx = crop index of crops.csv; the test crops of fold k are folds.json -> folds[rep 0, fold k] -> "test" (1,590 crops in total, each exactly once).
  * Train each fold's model only on that fold's "train" (and "val") indices. Never use other crops of a test crop's source radiograph.
Reports per fold and pooled: accuracy, balanced accuracy, macro-F1, per-brand recall / F1, macro ROC-AUC, ECE; mean +- SD over folds.
With --compare: exact McNemar test on the pooled predictions and the per-fold macro-F1 differences.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

NAMES = ["Bego", "Bicon", "ITI"]; HERE = Path(__file__).parent


def load(path, test):
    d = pd.read_csv(path)
    if "idx" not in d: raise SystemExit(f"{path}: needs an 'idx' column")
    d = d.drop_duplicates("idx").set_index("idx")
    miss = sorted(set(test) - set(d.index))
    if miss: raise SystemExit(f"{path}: {len(miss)} test crops have no prediction, e.g. {miss[:5]}")
    d = d.loc[test]
    if all(c in d for c in ("p_Bego", "p_Bicon", "p_ITI")):
        P = d[["p_Bego", "p_Bicon", "p_ITI"]].to_numpy(float); P = P / P.sum(1, keepdims=True); return P.argmax(1), P
    pred = d["pred"].map(lambda v: NAMES.index(v) if isinstance(v, str) else int(v)).to_numpy(); return pred, None


def ece(y, P, bins=15):
    c, pr = P.max(1), P.argmax(1); e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (c > lo) & (c <= hi)
        if m.any(): e += m.mean() * abs((pr[m] == y[m]).mean() - c[m].mean())
    return e


def metrics(y, pred, P):
    out = dict(n=len(y), accuracy=(pred == y).mean(), balanced_accuracy=balanced_accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"))
    f1 = f1_score(y, pred, average=None, labels=[0, 1, 2])
    for c, n in enumerate(NAMES): out[f"recall_{n}"] = (pred[y == c] == c).mean(); out[f"f1_{n}"] = f1[c]
    if P is not None: out["macro_roc_auc"] = roc_auc_score(y, P, multi_class="ovr"); out["ece"] = ece(y, P)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("predictions"); ap.add_argument("--compare")
    ap.add_argument("--folds", default=str(HERE / "folds.json")); ap.add_argument("--meta", default=str(HERE / "crops.csv")); a = ap.parse_args()
    meta = pd.read_csv(a.meta).set_index("idx"); folds = sorted([f for f in json.load(open(a.folds))["folds"] if f["rep"] == 0], key=lambda f: f["fold"])
    test = [i for f in folds for i in f["test"]]; y_all = meta.brand.loc[test].to_numpy(); fold_of = np.concatenate([[f["fold"]] * len(f["test"]) for f in folds])
    pred, P = load(a.predictions, test)
    rows = [dict(fold=k, **metrics(y_all[fold_of == k], pred[fold_of == k], None if P is None else P[fold_of == k])) for k in range(5)]
    T = pd.DataFrame(rows); pd.set_option("display.width", 200)
    print("per fold:"); print(T.round(4).to_string(index=False))
    print("\nmean +- SD over the 5 folds:")
    for c in T.columns[2:]: print(f"  {c:18s} {T[c].mean():.4f} +- {T[c].std(ddof=1):.4f}")
    print("\npooled over all 1,590 test crops:"); [print(f"  {k:18s} {v:.4f}" if isinstance(v, float) else f"  {k:18s} {v}") for k, v in metrics(y_all, pred, P).items()]
    if a.compare:
        pb, _ = load(a.compare, test); b, c = int(((pred == y_all) & (pb != y_all)).sum()), int(((pred != y_all) & (pb == y_all)).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        fb = [f1_score(y_all[fold_of == k], pb[fold_of == k], average="macro") for k in range(5)]
        print(f"\nvs {a.compare}: only this model right {b}, only baseline right {c}, exact McNemar p = {p:.4g}; "
              f"macro-F1 difference per fold {np.round(T.macro_f1.to_numpy() - fb, 4).tolist()}")
