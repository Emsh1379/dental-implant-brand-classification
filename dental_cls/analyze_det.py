#!/usr/bin/env python3
"""Aggregate detection results (det_train.py): mean ± SD over folds, box-weighted brand accuracy, paired fold-level tests."""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="/kaggle/working/det_results")
ap.add_argument("--out", default="/kaggle/working/det_report")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
CL = ["Bego", "Bicon", "ITI"]
R = [json.load(open(f)) for f in sorted(glob.glob(f"{a.results}/*.json"))]
df = pd.DataFrame([dict(model=r["model"], fold=r["fold"], n_images=r["n_images"], n_boxes=r["n_boxes"], mAP50=r["mAP50"], mAP=r["mAP"],
                        mAP75=r["mAP75"], AR100=r["AR100"], **{f"AP_{c}": r[f"AP_{c}"] for c in CL}, **{f"AP50_{c}": r[f"AP50_{c}"] for c in CL},
                        brand_acc=r["det_brand_acc"], brand_macro_f1=r["det_brand_macro_f1"], missed=r["det_missed"],
                        **{f"rec_{c}": r["det_brand_recall"][i] for i, c in enumerate(CL)}) for r in R])
df.to_csv(f"{a.out}/per_fold.csv", index=False)
MET = ["mAP50", "mAP", "mAP75", "AR100"] + [f"AP_{c}" for c in CL] + [f"AP50_{c}" for c in CL] + ["brand_acc", "brand_macro_f1"] + [f"rec_{c}" for c in CL]
ORDER = ["yolo11n", "rfdetr_small", "yolo11n_gtroi"]
rows = []
for m in [m for m in ORDER if m in set(df.model)]:
    d = df[df.model == m]
    row = dict(model=m, folds=len(d), n_boxes=int(d.n_boxes.sum()), missed=int(d.missed.sum()),
               brand_acc_pooled=(d.brand_acc * d.n_boxes).sum() / d.n_boxes.sum())
    for k in MET: row[k] = d[k].mean(); row[k + "_sd"] = d[k].std(ddof=1)
    rows.append(row)
T = pd.DataFrame(rows); T.to_csv(f"{a.out}/summary.csv", index=False)
print(T[["model", "folds", "n_boxes", "missed", "brand_acc_pooled"] + MET].round(3).T.to_string())

sig = []
for m1, m2 in [("yolo11n", "rfdetr_small"), ("yolo11n", "yolo11n_gtroi")]:
    d1, d2 = df[df.model == m1].set_index("fold"), df[df.model == m2].set_index("fold")
    f = sorted(set(d1.index) & set(d2.index))
    if len(f) < 2: continue
    for k in ("mAP50", "mAP", "AP_Bego", "brand_acc", "brand_macro_f1", "rec_Bego"):
        x, y = d1.loc[f, k].values, d2.loc[f, k].values
        sig.append(dict(a=m1, b=m2, metric=k, folds=len(f), delta=(y - x).mean(), b_better_folds=int((y > x).sum()),
                        t_p=stats.ttest_rel(x, y).pvalue, wilcoxon_p=stats.wilcoxon(x, y).pvalue if np.any(x != y) else np.nan))
S = pd.DataFrame(sig); S.to_csv(f"{a.out}/comparisons.csv", index=False); print(S.round(4).to_string(index=False))
