#!/usr/bin/env python3
"""Aggregate two_stage.py results: one-stage vs two-stage vs classifier on ground-truth boxes, operating-point
precision/recall, and the box-jitter curves. CPU only."""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="/kaggle/working/ts_results")
ap.add_argument("--out", default="/kaggle/working/ts_report")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
CL = ["Bego", "Bicon", "ITI"]
TS = [json.load(open(f)) for f in sorted(glob.glob(f"{a.results}/two_stage_*.json"))]
JT = [json.load(open(f)) for f in sorted(glob.glob(f"{a.results}/jitter_f*.json"))]


def score(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(acc=float((y == p).mean()), macro_f1=float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
                **{f"rec_{c}": float(((p == i) & (y == i)).sum() / max((y == i).sum(), 1)) for i, c in enumerate(CL)})


# classifier on ground-truth boxes (jitter condition scale 1.0), keyed by crop idx
gt_pred = {}
for j in JT:
    for i, p in zip(j["test_idx"], j["gt_probs"]): gt_pred[i] = int(np.argmax(p))
print("GT-box crops identical to stored crops per fold:", [round(j["crops_identical_to_stored"], 4) for j in JT])

# --- A. GT-matched brand metrics
rows, sig = [], []
systems = {}
for det in sorted({t["detector"] for t in TS}):
    R = [t for t in TS if t["detector"] == det]
    imp = [r for t in R for r in t["implants"]]
    y = [r["y"] for r in imp]
    systems[det] = {"one-stage (detector class)": ([r["det"] for r in imp], "det"), "two-stage (detector box -> classifier)": ([r["two_stage"] for r in imp], "two_stage"),
                    "classifier on ground-truth box": ([gt_pred[r["idx"]] for r in imp], None)}
    for name, (p, key) in systems[det].items():
        per = [score([r["y"] for r in t["implants"]], [r[key] if key else gt_pred[r["idx"]] for r in t["implants"]]) for t in R]
        row = dict(detector=det, system=name, folds=len(R), n=len(y), missed=sum(t["missed"] for t in R) if key else 0,
                   mean_iou=np.mean([r["iou"] for r in imp if r["iou"] is not None]) if key else 1.0)
        row.update({"pooled_" + k: v for k, v in score(y, p).items()})
        row.update({k: np.mean([q[k] for q in per]) for k in per[0]}); row.update({k + "_sd": np.std([q[k] for q in per], ddof=1) for k in per[0]})
        rows.append(row)
    names = list(systems[det])
    for n1, n2 in [(names[0], names[1]), (names[1], names[2]), (names[0], names[2])]:
        c1, c2 = np.array(systems[det][n1][0]) == np.array(y), np.array(systems[det][n2][0]) == np.array(y)
        b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
        yb = np.array(y) == 0
        sig.append(dict(detector=det, a=n1, b=n2, a_only_correct=b, b_only_correct=c, mcnemar_p=mcnemar([[0, b], [c, 0]], exact=True).pvalue,
                        bego_a_only=int((c1 & ~c2 & yb).sum()), bego_b_only=int((~c1 & c2 & yb).sum())))
A = pd.DataFrame(rows); A.to_csv(f"{a.out}/gt_matched.csv", index=False)
S = pd.DataFrame(sig); S.to_csv(f"{a.out}/gt_matched_mcnemar.csv", index=False)
print(A[["detector", "system", "n", "missed", "mean_iou", "acc", "acc_sd", "macro_f1", "macro_f1_sd", "rec_Bego", "rec_Bicon", "rec_ITI", "pooled_macro_f1"]].round(3).to_string(index=False))
print(S.round(4).to_string(index=False))

# --- B. operating point (pooled counts over folds)
rows = []
for det in sorted({t["detector"] for t in TS}):
    R = [t for t in TS if t["detector"] == det]
    for ti, thr in enumerate([o["thr"] for o in R[0]["operating"]]):
        ops = [t["operating"][ti] for t in R]
        for key in ("det", "two_stage"):
            row = dict(detector=det, thr=thr, labels=key, n_dets=sum(o["n_dets"] for o in ops), unmatched_dets=sum(o["n_unmatched_dets"] for o in ops)); f1s = []
            for i, c in enumerate(CL):
                tp, fp, fn = (sum(o[key][i][k] for o in ops) for k in ("tp", "fp", "fn"))
                pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1); f1s.append(2 * pr * rc / max(pr + rc, 1e-9))
                row.update({f"prec_{c}": pr, f"rec_{c}": rc, f"f1_{c}": f1s[-1]})
            row["macro_f1"] = np.mean(f1s); rows.append(row)
B = pd.DataFrame(rows); B.to_csv(f"{a.out}/operating_point.csv", index=False)
print(B.round(3).to_string(index=False))

# --- C. box jitter
rows = []
for j in JT:
    for c in j["conditions"]:
        rows.append(dict(fold=j["fold"], kind=c["kind"], level=c["level"], rep=c["rep"], mean_iou=c["mean_iou"], acc=c["acc"], macro_f1=c["macro_f1"],
                         **{f"rec_{n}": c["recall"][i] for i, n in enumerate(CL)}))
J = pd.DataFrame(rows); J.to_csv(f"{a.out}/jitter_per_fold.csv", index=False)
F = J.groupby(["kind", "level", "fold"], as_index=False).mean(numeric_only=True)  # average the random repeats inside a fold first
G = F.groupby(["kind", "level"]).agg(mean_iou=("mean_iou", "mean"), acc=("acc", "mean"), macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                                     rec_Bego=("rec_Bego", "mean"), rec_Bego_sd=("rec_Bego", "std"), rec_Bicon=("rec_Bicon", "mean"), rec_ITI=("rec_ITI", "mean")).reset_index()
G.to_csv(f"{a.out}/jitter_summary.csv", index=False); print(G.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
LAB = {"scale": "box scale factor", "shift": "centre shift (fraction of box size)", "edges": "edge noise SD (fraction of box size)"}
for x, kind in zip(ax, ["scale", "shift", "edges"]):
    d = G[G.kind == kind]
    x.errorbar(d.level, d.macro_f1, yerr=d.macro_f1_sd, c="#333", marker="o", ms=3, lw=1.2, capsize=2, label="macro-F1")
    x.errorbar(d.level, d.rec_Bego, yerr=d.rec_Bego_sd, c="#c63", marker="s", ms=3, lw=1.2, capsize=2, label="Bego recall")
    for lv, io, m in zip(d.level, d.mean_iou, d.macro_f1): x.annotate(f"{io:.2f}", (lv, m), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=6, color="#555")
    x.set_xlabel(LAB[kind]); x.grid(alpha=.3)
ax[0].set_ylabel("mean ± SD over 5 folds"); ax[0].legend(fontsize=7, loc="lower center"); ax[1].set_title("Small numbers: mean IoU with the ground-truth box", fontsize=8)
fig.tight_layout(); fig.savefig(f"{a.out}/jitter.png", dpi=200)
