#!/usr/bin/env python3
"""Per-class ROC-AUC / PR-AUC (one-vs-rest) from the saved test probabilities; CPU only.
Per (model, version): mean ± SD over folds, pooled out-of-fold value with bootstrap 95% CI,
and a paired bootstrap of every version against raw on the same crops."""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", default=["/kaggle/working/results", "/kaggle/working/results2"])
ap.add_argument("--out", default="/kaggle/working/report3")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
CL = ["Bego", "Bicon", "ITI"]
R = [r for r in (json.load(open(f)) for d in a.results for f in sorted(glob.glob(f"{d}/*.json"))) if r["rep"] == 0]
ORDER = ["raw", "hat", "pipeline", "ab_no_sharpen", "ab_no_blur", "ab_no_clahe", "ab_no_unsharp", "ab_no_rescale",
         "fixed", "ch3", "raw384", "hat384"]
MODELS = ["custom_cnn", "vgg16", "resnet50", "densenet121", "efficientnet_b0", "vit_b_16"]


def roc_pr(t, s):
    """ROC-AUC (Mann-Whitney, ties = 0.5) and average precision (step-wise, ties grouped) – same values as sklearn, fast enough to bootstrap."""
    n1 = t.sum(); n0 = len(t) - n1
    roc = (stats.rankdata(s)[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    o = np.argsort(-s, kind="stable"); ss, tp = s[o], np.cumsum(t[o])
    last = np.r_[ss[1:] != ss[:-1], True]  # last element of each tied block
    tp = tp[last]; k = np.flatnonzero(last) + 1
    return roc, float((np.diff(np.r_[0, tp]) * tp / k).sum() / n1)


def aucs(y, p):
    """[roc_Bego, roc_Bicon, roc_ITI, roc_macro, pr_Bego, pr_Bicon, pr_ITI, pr_macro]; NaN when a class is absent."""
    roc, pr = [], []
    for c in range(3):
        t = (y == c).astype(int)
        r, q = roc_pr(t, p[:, c]) if 0 < t.sum() < len(t) else (np.nan, np.nan)
        roc.append(r); pr.append(q)
    return np.array(roc + [np.mean(roc)] + pr + [np.mean(pr)])


KEYS = [f"roc_{c}" for c in CL] + ["roc_macro"] + [f"pr_{c}" for c in CL] + ["pr_macro"]
fold, oof = {}, {}
for r in R:
    k = (r["model"], r["version"]); y, p = np.array(r["y"]), np.array(r["probs"])
    fold.setdefault(k, []).append(aucs(y, p))
    d = oof.setdefault(k, {})
    for i, yi, pi in zip(r["test_idx"], y, p): d[i] = (yi, pi)
P = {}
for k, d in oof.items():
    ids = sorted(d); P[k] = (np.array(ids), np.array([d[i][0] for i in ids]), np.array([d[i][1] for i in ids]))

rng = np.random.default_rng(0)
rows = []
for (m, v), (ids, y, p) in P.items():
    f = np.array(fold[(m, v)]); pooled = aucs(y, p)
    bs = np.array([aucs(y[s], p[s]) for s in (rng.integers(0, len(y), len(y)) for _ in range(a.boot))])
    lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
    row = dict(model=m, version=v, folds=len(f), n=len(y), n_Bego=int((y == 0).sum()))
    for j, key in enumerate(KEYS):
        row.update({key: f[:, j].mean(), key + "_sd": f[:, j].std(ddof=1), key + "_pooled": pooled[j], key + "_lo": lo[j], key + "_hi": hi[j]})
    rows.append(row)
T = pd.DataFrame(rows)
T["om"], T["ov"] = T.model.map({m: i for i, m in enumerate(MODELS)}), T.version.map({v: i for i, v in enumerate(ORDER)})
T = T.sort_values(["om", "ov"]).drop(columns=["om", "ov"]); T.to_csv(f"{a.out}/auc_per_class.csv", index=False)
show = ["model", "version"] + [k + "_pooled" for k in KEYS]
print(T[show].round(3).to_string(index=False))

# paired bootstrap vs raw (same crops resampled for both versions)
cmp = []
for (m, v), (ids, y, p) in P.items():
    if v == "raw" or (m, "raw") not in P: continue
    ids0, y0, p0 = P[(m, "raw")]
    assert np.array_equal(ids, ids0) and np.array_equal(y, y0)
    d0 = aucs(y, p) - aucs(y0, p0)
    bs = np.array([aucs(y[s], p[s]) - aucs(y0[s], p0[s]) for s in (rng.integers(0, len(y), len(y)) for _ in range(a.boot))])
    for j, key in enumerate(KEYS):
        if key not in ("roc_Bego", "roc_macro", "pr_Bego", "pr_macro"): continue
        b = bs[:, j][~np.isnan(bs[:, j])]
        cmp.append(dict(model=m, a="raw", b=v, metric=key, delta=d0[j], lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                        p_boot=min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))))
C = pd.DataFrame(cmp); C.to_csv(f"{a.out}/auc_vs_raw.csv", index=False)
print(C[C.metric == "pr_Bego"].round(4).to_string(index=False))

# pooled Bego curves for the best model
fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
for v, col in zip(["raw", "hat", "pipeline", "fixed", "raw384"], ["#333", "#37c", "#c63", "#3a7", "#a3c"]):
    if ("densenet121", v) not in P: continue
    _, y, p = P[("densenet121", v)]; t = (y == 0).astype(int)
    fpr, tpr, _ = roc_curve(t, p[:, 0]); pre, rec, _ = precision_recall_curve(t, p[:, 0])
    ax[0].plot(fpr, tpr, c=col, lw=1.2, label=f"{v} ({roc_auc_score(t, p[:, 0]):.3f})")
    ax[1].plot(rec, pre, c=col, lw=1.2, label=f"{v} ({average_precision_score(t, p[:, 0]):.3f})")
ax[0].set_xlim(0, 0.2); ax[0].set_ylim(0.6, 1.005); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title("Bego vs rest – ROC (zoomed)", fontsize=9)
ax[1].set_ylim(0.4, 1.005); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Bego vs rest – precision-recall", fontsize=9)
for x in ax: x.legend(fontsize=7, title="DenseNet121 (AUC)", title_fontsize=7)
fig.tight_layout(); fig.savefig(f"{a.out}/bego_curves_densenet121.png", dpi=200)
