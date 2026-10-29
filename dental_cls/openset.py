#!/usr/bin/env python3
"""Open-set rejection: can the brand classifier say "not Bego / Bicon / ITI"?

usage: python openset.py --ckpt <dir with r0_f{k}_raw_orig_{model}.pt> --crops <dir: raw_orig.npy, meta.csv, folds.json> --code <dental_cls>
                         --ood name=<dir of .jpg crops> [name=<dir> ...] --out <dir>
In-distribution (ID): each fold's real test crops scored by that fold's model (never seen in training).
Out-of-distribution (OOD): implant crops from other datasets (brand unknown, mostly other brands), scored by all 5 fold models (mean score).
Rejection scores (higher = more likely ID): max softmax probability (MSP), energy (logsumexp of logits), kNN = cosine similarity of the
penultimate feature to the nearest training crop of the fold. Metrics: AUROC (ID vs OOD), FPR at 95 % ID retained, and the share of OOD accepted /
ID accuracy at that threshold (threshold chosen on ID only, so it is not fitted to the OOD set).
"""
import argparse, glob, json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

ap = argparse.ArgumentParser()
for k in ("ckpt", "crops", "code", "out"): ap.add_argument("--" + k)
ap.add_argument("--ood", nargs="+"); ap.add_argument("--models", nargs="+", default=["vit_b_16", "densenet121"])
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, a.code)
from models import build
from common import letterbox
dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
X = np.load(f"{a.crops}/raw_orig.npy", mmap_mode="r"); meta = pd.read_csv(f"{a.crops}/meta.csv").iloc[:3276]; y = meta.brand.values
folds = {f["fold"]: f for f in json.load(open(f"{a.crops}/folds.json"))["folds"] if f["rep"] == 0}
OOD = {}
for spec in a.ood:
    name, d = spec.split("=", 1); fs = sorted(glob.glob(f"{d}/*.jpg"))
    OOD[name] = np.stack([letterbox(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in fs]); print("OOD", name, len(fs), flush=True)


@torch.no_grad()
def run(model, head, A):
    feats, logits = [], []
    h = head.register_forward_hook(lambda m, i, o: feats.append(i[0].float()))
    for s in range(0, len(A), 128):
        x = torch.from_numpy(np.ascontiguousarray(A[s:s + 128])).to(dev).float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
        with torch.autocast("cuda", dtype=torch.float16): logits.append(model((x - MEAN) / STD).float())
    h.remove(); return F.normalize(torch.cat(feats), dim=1), torch.cat(logits)


def scores(feat, logit, ftr):
    sim = torch.cat([(feat[s:s + 2048] @ ftr.T).max(1).values for s in range(0, len(feat), 2048)])
    return dict(msp=F.softmax(logit, 1).max(1).values.cpu().numpy(), energy=torch.logsumexp(logit, 1).cpu().numpy(), knn=sim.cpu().numpy(),
                pred=logit.argmax(1).cpu().numpy())


rows, keep = [], {}
for name in a.models:
    idS, oodS = [], {o: [] for o in OOD}
    for k in range(5):
        model, head = build(name, pretrained=False); model.load_state_dict(torch.load(f"{a.ckpt}/r0_f{k}_raw_orig_{name}.pt", map_location="cpu")); model.to(dev).eval()
        tr = np.array([i for i in folds[k]["train"] if i < 3276]); te = np.array(folds[k]["test"])
        ftr, _ = run(model, head, X[tr]); f, l = run(model, head, X[te]); s = scores(f, l, ftr); s["y"] = y[te]; idS.append(s)
        for o, A in OOD.items(): f, l = run(model, head, A); oodS[o].append(scores(f, l, ftr))
        del model; torch.cuda.empty_cache(); print(name, "fold", k, "done", flush=True)
    ID = {key: np.concatenate([s[key] for s in idS]) for key in idS[0]}
    for o in OOD:
        O = {key: np.mean([s[key] for s in oodS[o]], 0) for key in ("msp", "energy", "knn")}
        O["pred"] = np.array([np.bincount(c, minlength=3).argmax() for c in np.stack([s["pred"] for s in oodS[o]], 1)])
        for sc in ("msp", "energy", "knn"):
            yid, ood = ID[sc], O[sc]
            thr = np.quantile(yid, 0.05)  # keep 95 % of ID
            acc_kept = float((ID["pred"][yid >= thr] == ID["y"][yid >= thr]).mean())
            rows.append(dict(model=name, ood_set=o, n_id=len(yid), n_ood=len(ood), score=sc, auroc=float(roc_auc_score(np.r_[np.ones(len(yid)), np.zeros(len(ood))], np.r_[yid, ood])),
                             ood_accepted_at_95_id=float((ood >= thr).mean()), id_acc_all=float((ID["pred"] == ID["y"]).mean()), id_acc_kept=acc_kept,
                             ood_pred_share=json.dumps(np.round(np.bincount(O["pred"], minlength=3) / len(O["pred"]), 3).tolist())))
            print(rows[-1], flush=True)
        keep[(name, o)] = O
    keep[(name, "ID")] = ID
R = pd.DataFrame(rows); R.to_csv(out / "openset.csv", index=False)
pd.set_option("display.width", 220); print(R.drop(columns=["n_id"]).round(4).to_string(index=False))
np.savez_compressed(out / "openset_scores.npz", **{f"{m}|{o}|{k}": v for (m, o), d in keep.items() for k, v in d.items()})
