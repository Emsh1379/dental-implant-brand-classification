#!/usr/bin/env python3
"""Summary of lashaki_protocol.py results: mean (SD) over 5 folds per task and protocol, and the paired difference to the leaky protocol.
usage: python analyze_lashaki.py <results dir> <out dir>"""
import glob, json, os, sys
import numpy as np, pandas as pd

R, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
rows = []
for f in glob.glob(f"{R}/*.json"):
    r = json.load(open(f))
    bego = r["recall"][1] if r["task"] == "bicon_bego" else r["recall"][0]  # binary recall order [Bicon, Bego]; 3class [Bego, Bicon, ITI]
    rows.append(dict(task=r["task"], protocol=r["protocol"], fold=r["fold"], n_train=r["n_train"], n_test=r["n_test"], epochs=r["epochs"],
                     acc=r["acc"], macro_f1=r["macro_f1"], auc=r["auc"], bego_recall=bego))
D = pd.DataFrame(rows).sort_values(["task", "protocol", "fold"]); D.to_csv(f"{OUT}/per_fold.csv", index=False)
order = ["leaky", "grouped", "grouped_nomask", "grouped_raw"]
S = D.groupby(["task", "protocol"]).agg(folds=("fold", "size"), n_test=("n_test", "mean"), acc=("acc", "mean"), acc_sd=("acc", "std"), macro_f1=("macro_f1", "mean"),
                                         macro_f1_sd=("macro_f1", "std"), auc=("auc", "mean"), bego_recall=("bego_recall", "mean")).reset_index()
S["protocol"] = pd.Categorical(S.protocol, order); S = S.sort_values(["task", "protocol"]); S.to_csv(f"{OUT}/summary.csv", index=False)
pd.set_option("display.width", 200); print(S.round(4).to_string(index=False))
