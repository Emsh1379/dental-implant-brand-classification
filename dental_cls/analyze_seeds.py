#!/usr/bin/env python3
"""Seed repeats (train_cv.py --seed): same folds, several training seeds. Seed-averaged results and paired comparisons. CPU only.
--results lists every directory with result JSONs; a JSON without a "seed" field is seed 0."""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, balanced_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", default=["/kaggle/working/results", "/kaggle/working/results2", "/kaggle/working/results_seeds/s1", "/kaggle/working/results_seeds/s2"])
ap.add_argument("--out", default="/kaggle/working/report_seeds")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
CONFIGS = [("densenet121", v) for v in ("raw", "pipeline", "hat", "raw384")] + [("vit_b_16", v) for v in ("raw", "pipeline", "hat")]
PAIRS = [("densenet121", "raw", "pipeline"), ("densenet121", "raw", "hat"), ("densenet121", "raw", "raw384"), ("densenet121", "hat", "pipeline"),
         ("vit_b_16", "raw", "pipeline"), ("vit_b_16", "raw", "hat"), ("vit_b_16", "hat", "pipeline")]
R = {}
for d in a.results:
    for f in sorted(glob.glob(f"{d}/*.json")):
        r = json.load(open(f))
        if r["rep"] == 0 and (r["model"], r["version"]) in CONFIGS: R[(r["model"], r["version"], r.get("seed", 0), r["fold"])] = r
seeds = sorted({k[2] for k in R})
df = pd.DataFrame([dict(model=m, version=v, seed=s, fold=f, acc=r["metrics"]["acc"], bal_acc=r["metrics"]["bal_acc"], macro_f1=r["metrics"]["macro_f1"],
                        rec_Bego=r["metrics"]["recall"][0]) for (m, v, s, f), r in R.items()])
df.to_csv(f"{a.out}/per_run.csv", index=False)
n = df.groupby(["model", "version"]).size()
bad = [k for k in CONFIGS if n.get(k, 0) != 5 * len(seeds)]
if bad: print("incomplete, left out:", [(k, int(n.get(k, 0))) for k in bad])
CONFIGS = [k for k in CONFIGS if k not in bad]; PAIRS = [p for p in PAIRS if (p[0], p[1]) in CONFIGS and (p[0], p[2]) in CONFIGS]
df = df[[k in CONFIGS for k in zip(df.model, df.version)]]


def pooled(m, v, s):
    """Out-of-fold probabilities of one seed, ordered by crop idx."""
    d = {}
    for f in range(5):
        r = R[(m, v, s, f)]
        for i, y, p in zip(r["test_idx"], r["y"], r["probs"]): d[i] = (y, p)
    ids = sorted(d)
    return np.array(ids), np.array([d[i][0] for i in ids]), np.array([d[i][1] for i in ids])


rows = []
for m, v in CONFIGS:
    d = df[(df.model == m) & (df.version == v)]
    by_seed = d.groupby("seed").macro_f1.mean()
    ids, y, _ = pooled(m, v, seeds[0]); P = np.mean([pooled(m, v, s)[2] for s in seeds], 0); e = P.argmax(1)
    row = dict(model=m, version=v, seeds=len(seeds), runs=len(d), macro_f1=d.macro_f1.mean(), macro_f1_sd_runs=d.macro_f1.std(ddof=1),
               macro_f1_sd_folds=d.groupby("fold").macro_f1.mean().std(ddof=1), macro_f1_sd_seeds=by_seed.std(ddof=1) if len(seeds) > 1 else np.nan,
               macro_f1_seed_min=by_seed.min(), macro_f1_seed_max=by_seed.max(),
               within_fold_seed_sd=d.groupby("fold").macro_f1.std(ddof=1).mean() if len(seeds) > 1 else np.nan,
               acc=d.acc.mean(), bal_acc=d.bal_acc.mean(), rec_Bego=d.rec_Bego.mean(), rec_Bego_seed_min=d.groupby("seed").rec_Bego.mean().min(),
               rec_Bego_seed_max=d.groupby("seed").rec_Bego.mean().max(),
               ens_acc=(e == y).mean(), ens_macro_f1=f1_score(y, e, average="macro"), ens_bal_acc=balanced_accuracy_score(y, e), ens_rec_Bego=(e[y == 0] == 0).mean())
    row.update({f"macro_f1_seed{s}": by_seed[s] for s in seeds}); rows.append(row)
T = pd.DataFrame(rows); T.to_csv(f"{a.out}/summary.csv", index=False)
print(T[["model", "version", "runs", "macro_f1", "macro_f1_sd_folds", "macro_f1_sd_seeds", "within_fold_seed_sd"] + [f"macro_f1_seed{s}" for s in seeds] +
        ["acc", "rec_Bego", "ens_macro_f1", "ens_rec_Bego"]].round(3).to_string(index=False))

sig = []
for m, v1, v2 in PAIRS:
    d1 = df[(df.model == m) & (df.version == v1)].set_index(["seed", "fold"]).macro_f1; d2 = df[(df.model == m) & (df.version == v2)].set_index(["seed", "fold"]).macro_f1
    dl = (d2 - d1)
    f = dl.groupby("fold").mean()  # seed-averaged difference per fold: the 5 independent units
    row = dict(model=m, a=v1, b=v2, delta_macro_f1=dl.mean(), delta_per_seed=" ".join(f"{x:+.3f}" for x in dl.groupby("seed").mean()),
               b_better_seeds=int((dl.groupby("seed").mean() > 0).sum()), b_better_folds=int((f > 0).sum()), t_p_folds=stats.ttest_1samp(f, 0).pvalue,
               wilcoxon_p_folds=stats.wilcoxon(f).pvalue if np.any(f != 0) else np.nan)
    ps = []
    for s in seeds + ["ens"]:  # McNemar per seed, and on the seed-averaged probabilities
        if s == "ens":
            y = pooled(m, v1, seeds[0])[1]; p1 = np.mean([pooled(m, v1, t)[2] for t in seeds], 0).argmax(1); p2 = np.mean([pooled(m, v2, t)[2] for t in seeds], 0).argmax(1)
        else:
            _, y, q1 = pooled(m, v1, s); _, _, q2 = pooled(m, v2, s); p1, p2 = q1.argmax(1), q2.argmax(1)
        b, c = int(((p1 == y) & (p2 != y)).sum()), int(((p1 != y) & (p2 == y)).sum())
        p = mcnemar([[0, b], [c, 0]], exact=True).pvalue
        if s == "ens": row.update(ens_a_only=b, ens_b_only=c, ens_mcnemar_p=p)
        else: ps.append(p)
    row["mcnemar_p_per_seed"] = " ".join(f"{p:.3f}" for p in ps); sig.append(row)
S = pd.DataFrame(sig); S["ens_mcnemar_p_holm"] = multipletests(S.ens_mcnemar_p, method="holm")[1]
S.to_csv(f"{a.out}/comparisons.csv", index=False); print(S.round(4).to_string(index=False))
