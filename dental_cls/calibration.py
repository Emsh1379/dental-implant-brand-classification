#!/usr/bin/env python3
"""Calibration and selective prediction for the original-image classifiers (results_orig: DenseNet121 / ViT-B/16, 3 seeds x 5 folds). CPU only.

usage: python calibration.py <results_orig dir with s0,s1,s2> <out dir>
* calibration of the pooled out-of-fold probabilities: ECE (15 equal-width bins on the top-class probability), classwise ECE, Brier score, NLL;
  before and after temperature scaling. Temperature is cross-fitted: for the crops of fold k it is fitted (min NLL) on the other 4 folds' crops,
  so no crop is calibrated with its own label.
* selective prediction: crops ranked by top-class probability; accuracy, macro-F1, Bego recall and share of errors removed at 100/95/90/80 %
  coverage; AURC (area under the risk-coverage curve, lower is better). Also for the 3-seed ensemble (mean probabilities).
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
runs = {}
for f in glob.glob(f"{R}/s*/r0_f*_raw_orig_*.json"):
    r = json.load(open(f))
    runs.setdefault((r["model"], r["seed"]), {})[r["fold"]] = r


def pooled(folds):
    idx, y, p, fo = [], [], [], []
    for k in sorted(folds):
        r = folds[k]; idx += r["test_idx"]; y += r["y"]; p += r["probs"]; fo += [k] * len(r["y"])
    o = np.argsort(idx); return np.array(idx)[o], np.array(y)[o], np.clip(np.array(p)[o], 1e-7, 1), np.array(fo)[o]


def ece(y, p, bins=15):
    conf, pred = p.max(1), p.argmax(1); e = 0; rel = []
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean()); rel.append((conf[m].mean(), (pred[m] == y[m]).mean(), int(m.sum())))
    return e, rel


def cw_ece(y, p, bins=15):
    return np.mean([ece((y == c).astype(int), np.stack([1 - p[:, c], p[:, c]], 1), bins)[0] for c in range(3)])


def temp(p, t): q = np.exp(np.log(p) / t); return q / q.sum(1, keepdims=True)


def crossfit_T(y, p, fo):
    out, Ts = np.zeros_like(p), []
    for k in np.unique(fo):
        tr = fo != k
        T = minimize_scalar(lambda t: -np.log(temp(p[tr], t)[np.arange(tr.sum()), y[tr]]).mean(), bounds=(0.05, 10), method="bounded").x
        out[fo == k] = temp(p[fo == k], T); Ts.append(T)
    return out, Ts


def metrics(y, p):
    e, rel = ece(y, p)
    return dict(acc=float((p.argmax(1) == y).mean()), ece=float(e), classwise_ece=float(cw_ece(y, p)),
                brier=float(((p - np.eye(3)[y]) ** 2).sum(1).mean()), nll=float(-np.log(p[np.arange(len(y)), y]).mean()), mean_conf=float(p.max(1).mean())), rel


def selective(y, p):
    conf = p.max(1); pred = p.argmax(1); o = np.argsort(-conf); err = (pred != y)[o]
    risk = np.cumsum(err) / np.arange(1, len(y) + 1); aurc = float(risk.mean()); rows = {}
    for cov in (1.0, 0.95, 0.9, 0.8):
        n = int(round(cov * len(y))); k = o[:n]
        rows[cov] = dict(acc=float((pred[k] == y[k]).mean()), macro_f1=float(f1_score(y[k], pred[k], average="macro")),
                         bego_recall=float((pred[k][y[k] == 0] == 0).mean()) if (y[k] == 0).any() else np.nan, bego_kept=int((y[k] == 0).sum()),
                         errors_removed=float(1 - err[:n].sum() / max(err.sum(), 1)))
    return aurc, rows, risk


cal_rows, sel_rows, curves, rels = [], [], {}, {}
for model in ("densenet121", "vit_b_16"):
    sets = {f"seed {s}": pooled(runs[(model, s)]) for s in (0, 1, 2) if (model, s) in runs and len(runs[(model, s)]) == 5}
    i0, y0, _, f0 = sets["seed 0"]
    sets["3-seed ensemble"] = (i0, y0, np.mean([v[2] for v in sets.values()], 0), f0)
    for name, (idx, y, p, fo) in sets.items():
        pt, Ts = crossfit_T(y, p, fo)
        for tag, q in (("raw", p), ("temperature-scaled", pt)):
            m, rel = metrics(y, q); cal_rows.append(dict(model=model, run=name, probs=tag, T_mean=float(np.mean(Ts)) if tag != "raw" else 1.0, **m)); rels[(model, name, tag)] = rel
        aurc, rows, risk = selective(y, pt); curves[(model, name)] = risk
        for cov, r in rows.items(): sel_rows.append(dict(model=model, run=name, coverage=cov, aurc=aurc, **r))
C, S = pd.DataFrame(cal_rows), pd.DataFrame(sel_rows)
C.to_csv(f"{OUT}/calibration.csv", index=False); S.to_csv(f"{OUT}/selective.csv", index=False)
pd.set_option("display.width", 220)
print(C.round(4).to_string(index=False)); print(); print(S.round(4).to_string(index=False))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for j, model in enumerate(("densenet121", "vit_b_16")):
    for tag, st in (("raw", "o--"), ("temperature-scaled", "s-")):
        rel = [r for r in rels[(model, "3-seed ensemble", tag)] if r[2] >= 20]  # bins with >= 20 crops only
        ax[j].plot([r[0] for r in rel], [r[1] for r in rel], st, label=f"{tag} (ECE {C[(C.model == model) & (C.run == '3-seed ensemble') & (C.probs == tag)].ece.iloc[0]:.3f})")
    ax[j].plot([0, 1], [0, 1], "k:", lw=1); ax[j].set(title=f"{model}, 3-seed ensemble", xlabel="confidence", ylabel="accuracy", xlim=(0.3, 1), ylim=(0.3, 1)); ax[j].legend()
for model in ("densenet121", "vit_b_16"):
    r = curves[(model, "3-seed ensemble")]; ax[2].plot(np.arange(1, len(r) + 1) / len(r), r, label=model)
ax[2].set(title="risk-coverage (3-seed ensemble)", xlabel="coverage (share of implants the model answers)", ylabel="error rate among accepted",
          xlim=(0.7, 1), ylim=(0, 0.025)); ax[2].legend()
ax[0].set_title("densenet121, 3-seed ensemble (bins >= 20 crops)"); ax[1].set_title("vit_b_16, 3-seed ensemble (bins >= 20 crops)")
plt.tight_layout(); plt.savefig(f"{OUT}/calibration.png", dpi=130)
