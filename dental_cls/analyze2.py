#!/usr/bin/env python3
"""Round-2 analysis: preprocessing ablations, revised pipelines, 3-channel input, 384px resolution.
Combines round-1 results (raw/pipeline/hat) with round-2 results; every comparison is paired on identical folds."""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, balanced_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", default=["/kaggle/working/results", "/kaggle/working/results2"])
ap.add_argument("--out", default="/kaggle/working/report2")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
R = [json.load(open(f)) for d in a.results for f in sorted(glob.glob(f"{d}/*.json")) if json.load(open(f))["rep"] == 0]
MODELS = ["densenet121", "vit_b_16", "efficientnet_b0"]
R = [r for r in R if r["model"] in MODELS]
df = pd.DataFrame([dict(fold=r["fold"], version=r["version"], model=r["model"], **{k: r["metrics"][k] for k in ("acc", "bal_acc", "macro_f1")},
                        rec_Bego=r["metrics"]["recall"][0]) for r in R])
oof = {}
for r in R:
    d = oof.setdefault((r["model"], r["version"]), {})
    for i, y, p in zip(r["test_idx"], r["y"], r["probs"]): d[i] = (y, int(np.argmax(p)))

ORDER = ["raw", "hat", "pipeline", "ab_no_sharpen", "ab_no_blur", "ab_no_clahe", "ab_no_unsharp", "ab_no_rescale",
         "fixed", "ch3", "raw384", "hat384"]
rows = []
for (m, v), d in df.groupby(["model", "version"]):
    o = oof[(m, v)]; ids = sorted(o); y = np.array([o[i][0] for i in ids]); p = np.array([o[i][1] for i in ids])
    rows.append(dict(model=m, version=v, folds=len(d), macro_f1=d.macro_f1.mean(), macro_f1_sd=d.macro_f1.std(ddof=1),
                     bal_acc=d.bal_acc.mean(), acc=d.acc.mean(), rec_Bego=d.rec_Bego.mean(),
                     pooled_macro_f1=f1_score(y, p, average="macro"), pooled_bal_acc=balanced_accuracy_score(y, p)))
T = pd.DataFrame(rows); T["o"] = T.version.map({v: i for i, v in enumerate(ORDER)})
T = T.sort_values(["model", "o"]).drop(columns="o"); T.to_csv(f"{a.out}/summary.csv", index=False)
print(T.round(3).to_string(index=False))

# planned comparisons (paired, same folds/crops)
PAIRS = [("raw", "pipeline"), ("raw", "fixed"), ("raw", "ch3"), ("pipeline", "fixed")] + \
        [("pipeline", f"ab_no_{s}") for s in ("sharpen", "blur", "clahe", "unsharp", "rescale")] + \
        [("raw", "raw384"), ("hat", "hat384"), ("raw384", "hat384")]
sig = []
for m in MODELS:
    for v1, v2 in PAIRS:
        if (m, v1) not in oof or (m, v2) not in oof: continue
        d1 = df[(df.model == m) & (df.version == v1)].sort_values("fold").macro_f1.values
        d2 = df[(df.model == m) & (df.version == v2)].sort_values("fold").macro_f1.values
        ids = sorted(set(oof[(m, v1)]) & set(oof[(m, v2)]))
        c1 = np.array([oof[(m, v1)][i][0] == oof[(m, v1)][i][1] for i in ids]); c2 = np.array([oof[(m, v2)][i][0] == oof[(m, v2)][i][1] for i in ids])
        b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
        sig.append(dict(model=m, a=v1, b=v2, delta_macro_f1=d2.mean() - d1.mean(), t_p=stats.ttest_rel(d1, d2).pvalue,
                        a_only_correct=b, b_only_correct=c, mcnemar_p=mcnemar([[0, b], [c, 0]], exact=True).pvalue))
S = pd.DataFrame(sig)
S["mcnemar_p_holm"] = multipletests(S.mcnemar_p, method="holm")[1]
S.to_csv(f"{a.out}/comparisons.csv", index=False); print(S.round(4).to_string(index=False))

for m in MODELS:
    d = T[T.model == m]
    if d.empty: continue
    fig, ax = plt.subplots(figsize=(8, 3.4))
    col = ["#888" if v in ("raw", "hat", "pipeline") else "#3a7" if v in ("fixed", "ch3") else "#c63" if v.startswith("ab_") else "#37c" for v in d.version]
    ax.bar(d.version, d.macro_f1, yerr=d.macro_f1_sd, color=col, capsize=2)
    base = d[d.version == "raw"].macro_f1
    if len(base): ax.axhline(base.iloc[0], ls="--", c="k", lw=.8)
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("Macro-F1 (mean ± SD, 5 folds)"); ax.set_title(m)
    plt.xticks(rotation=35, ha="right"); fig.tight_layout(); fig.savefig(f"{a.out}/round2_{m}.png", dpi=200); plt.close(fig)
