#!/usr/bin/env python3
"""Aggregate CV results: mean±SD per fold, 95% CIs, pooled out-of-fold metrics, paired significance tests."""
import argparse, glob, json, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="/kaggle/working/results")
ap.add_argument("--crops", default="/tmp/data/crops")
ap.add_argument("--out", default="/kaggle/working/report")
a = ap.parse_args()
import os; os.makedirs(a.out, exist_ok=True)
CL = ["Bego", "Bicon", "ITI"]
R = [json.load(open(f)) for f in sorted(glob.glob(f"{a.results}/*.json"))]
rows = [dict(rep=r["rep"], fold=r["fold"], version=r["version"], model=r["model"], acc=r["metrics"]["acc"],
             bal_acc=r["metrics"]["bal_acc"], macro_f1=r["metrics"]["macro_f1"],
             **{f"f1_{c}": r["metrics"]["f1"][i] for i, c in enumerate(CL)},
             **{f"rec_{c}": r["metrics"]["recall"][i] for i, c in enumerate(CL)}) for r in R]
df = pd.DataFrame(rows); df.to_csv(f"{a.out}/per_fold_metrics.csv", index=False)

def ci95(x):
    x = np.asarray(x); n = len(x)
    return stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan

# fold-level mean ± SD (+ 95% CI of the mean)
g = df.groupby(["model", "version"])
tab = []
for (m, v), d in g:
    row = dict(model=m, version=v, n_runs=len(d))
    for k in ("acc", "bal_acc", "macro_f1", "rec_Bego", "rec_Bicon", "rec_ITI"):
        row[k] = f"{d[k].mean():.3f} ± {d[k].std(ddof=1):.3f}"; row[k + "_ci"] = ci95(d[k])
    tab.append(row)
T = pd.DataFrame(tab); T.to_csv(f"{a.out}/summary_mean_sd.csv", index=False)
print(T[["model", "version", "n_runs", "acc", "bal_acc", "macro_f1", "rec_Bego"]].to_string(index=False))

# pooled out-of-fold predictions (rep 0 only; every primary crop is tested exactly once)
oof = {}
for r in R:
    if r["rep"] != 0: continue
    k = (r["model"], r["version"])
    d = oof.setdefault(k, {})
    for i, y, p in zip(r["test_idx"], r["y"], r["probs"]): d[i] = (y, int(np.argmax(p)))
rng = np.random.default_rng(0)
pooled = []
for (m, v), d in sorted(oof.items()):
    ids = sorted(d); y = np.array([d[i][0] for i in ids]); p = np.array([d[i][1] for i in ids])
    bs = [f1_score(y[s], p[s], average="macro") for s in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    pooled.append(dict(model=m, version=v, n=len(y), acc=(y == p).mean(), bal_acc=balanced_accuracy_score(y, p),
                       macro_f1=f1_score(y, p, average="macro"), f1_lo=np.percentile(bs, 2.5), f1_hi=np.percentile(bs, 97.5),
                       **{f"rec_{c}": ((p == i) & (y == i)).sum() / max((y == i).sum(), 1) for i, c in enumerate(CL)}))
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(3.4, 3.2)); ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i, j in itertools.product(range(3), range(3)): ax.text(j, i, cm[i, j], ha="center", va="center", color="k")
    ax.set_xticks(range(3), CL); ax.set_yticks(range(3), CL); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{m} / {v}", fontsize=9); fig.tight_layout(); fig.savefig(f"{a.out}/cm_{m}_{v}.png", dpi=200); plt.close(fig)
P = pd.DataFrame(pooled); P.to_csv(f"{a.out}/pooled_oof.csv", index=False)
print(P.round(3).to_string(index=False))

# significance
sig = []
def add(kind, m, v1, v2, p, extra=""):
    sig.append(dict(test=kind, model=m, a=v1, b=v2, p=p, note=extra))
for m in sorted(df.model.unique()):
    for v1, v2 in itertools.combinations(["raw", "pipeline", "hat"], 2):
        d1 = df[(df.model == m) & (df.version == v1)].sort_values(["rep", "fold"]).macro_f1.values
        d2 = df[(df.model == m) & (df.version == v2)].sort_values(["rep", "fold"]).macro_f1.values
        if len(d1) == len(d2) and len(d1) > 1:
            add("paired t (fold macro-F1)", m, v1, v2, stats.ttest_rel(d1, d2).pvalue, f"Δ={d2.mean() - d1.mean():+.3f}")
            if np.any(d1 != d2): add("wilcoxon (fold macro-F1)", m, v1, v2, stats.wilcoxon(d1, d2).pvalue)
        if (m, v1) in oof and (m, v2) in oof:  # McNemar on pooled OOF correctness
            ids = sorted(set(oof[(m, v1)]) & set(oof[(m, v2)]))
            c1 = np.array([oof[(m, v1)][i][0] == oof[(m, v1)][i][1] for i in ids]); c2 = np.array([oof[(m, v2)][i][0] == oof[(m, v2)][i][1] for i in ids])
            tb = [[(c1 & c2).sum(), (c1 & ~c2).sum()], [(~c1 & c2).sum(), (~c1 & ~c2).sum()]]
            add("McNemar (pooled OOF)", m, v1, v2, mcnemar(tb, exact=True).pvalue, f"b={tb[0][1]} c={tb[1][0]}")
S = pd.DataFrame(sig)
if len(S):
    for t, d in S.groupby("test"):
        S.loc[d.index, "p_holm"] = multipletests(d.p, method="holm")[1]
    S.to_csv(f"{a.out}/significance.csv", index=False); print(S.round(4).to_string(index=False))

# learning-curve style bar chart: macro-F1 per model/version
fig, ax = plt.subplots(figsize=(8, 3.6)); W = .26
ms = sorted(df.model.unique())
for k, v in enumerate(["raw", "pipeline", "hat"]):
    mu = [df[(df.model == m) & (df.version == v)].macro_f1.mean() for m in ms]
    sd = [df[(df.model == m) & (df.version == v)].macro_f1.std(ddof=1) for m in ms]
    ax.bar(np.arange(len(ms)) + (k - 1) * W, mu, W, yerr=sd, capsize=2, label=v)
ax.set_xticks(range(len(ms)), ms, rotation=20); ax.set_ylabel("Macro-F1 (mean ± SD over folds)"); ax.legend(); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(f"{a.out}/macro_f1_by_model_version.png", dpi=200)
