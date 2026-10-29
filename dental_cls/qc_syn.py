#!/usr/bin/env python3
"""Quality check of the synthetic crops of each fold (cdiff_brand.py sample output).

usage: python qc_syn.py --syn <dir with syn_f*.npy> --crops <dir: raw_orig.npy, meta.csv, folds.json> --ckpt <dir with r0_f{k}_raw384_densenet121.pt> --code <dental_cls> --out <dir>
* brand fidelity: the fold-k DenseNet121 (raw 384 px, trained on fold k's training crops only, Hugging Face checkpoints_ts/) classifies the
  synthetic crops; reference = the same model on fold k's real test crops of that brand.
* memorisation: highest normalised correlation (64 x 64, with horizontal flip) of each synthetic Bego with any real TRAINING crop of the fold;
  reference = the same for the fold's real test Bego (unseen real images). Clearly higher values for synthetic images would mean copies.
* diversity: mean pairwise correlation among synthetic Bego vs among real training Bego.
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
for k in ("syn", "crops", "ckpt", "code", "out"): ap.add_argument("--" + k)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, a.code)
from models import build
dev = torch.device("cuda"); NAMES = ["Bego", "Bicon", "ITI"]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
X = np.load(f"{a.crops}/raw_orig.npy", mmap_mode="r"); meta = pd.read_csv(f"{a.crops}/meta.csv").iloc[:3276]; y = meta.brand.values
folds = {f["fold"]: f for f in json.load(open(f"{a.crops}/folds.json"))["folds"] if f["rep"] == 0}


@torch.no_grad()
def classify(model, A):
    P = []
    for s in range(0, len(A), 64):
        x = torch.from_numpy(np.stack([cv2.resize(np.asarray(i), (384, 384), interpolation=cv2.INTER_CUBIC) for i in A[s:s + 64]])).to(dev).float().div(255)
        with torch.autocast("cuda", dtype=torch.float16):
            P.append(F.softmax(model(((x.unsqueeze(1).expand(-1, 3, -1, -1)) - MEAN) / STD).float(), 1).cpu())
    return torch.cat(P).numpy()


def vec(A):
    v = np.stack([cv2.resize(np.asarray(i), (64, 64), interpolation=cv2.INTER_AREA) for i in A]).reshape(len(A), -1).astype(np.float32)
    v -= v.mean(1, keepdims=True); return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)


rows = []
for k in sorted(folds):
    p = Path(a.syn) / f"syn_f{k}.npy"
    if not p.exists(): continue
    S, L = np.load(p), np.load(Path(a.syn) / f"syn_f{k}_labels.npy")
    model, _ = build("densenet121", pretrained=False); model.load_state_dict(torch.load(f"{a.ckpt}/r0_f{k}_raw384_densenet121.pt", map_location="cpu")); model.to(dev).eval()
    tr = np.array([i for i in folds[k]["train"] if i < 3276]); te = np.array(folds[k]["test"])
    Ps, Pt = classify(model, S), classify(model, X[te])
    Vtr = vec(X[tr]); Vtr_f = vec(np.ascontiguousarray(np.asarray(X[tr])[:, :, ::-1]))
    nn_sim = lambda V: np.maximum((V @ Vtr.T).max(1), (V @ Vtr_f.T).max(1))
    for c in range(3):
        s = L == c
        if not s.any(): continue
        r = dict(fold=k, brand=NAMES[c], n_syn=int(s.sum()), syn_pred_as_brand=float((Ps[s].argmax(1) == c).mean()),
                 syn_mean_prob=float(Ps[s, c].mean()), real_test_recall=float((Pt[y[te] == c].argmax(1) == c).mean()))
        if c == 0:
            Vs = vec(S[s]); Vr = vec(X[te][y[te] == 0]); ns, nr = nn_sim(Vs), nn_sim(Vr)
            Vb = vec(X[tr][y[tr] == 0]); div = lambda V: float((V @ V.T)[np.triu_indices(len(V), 1)].mean())
            r.update(syn_nn_corr_median=float(np.median(ns)), syn_nn_corr_max=float(ns.max()), syn_over_0_95=int((ns > 0.95).sum()),
                     real_test_nn_corr_median=float(np.median(nr)), syn_pairwise_corr=div(Vs), real_train_bego_pairwise_corr=div(Vb))
            idx = np.argsort(-ns)[:6]  # closest synthetic/real pairs for a visual memorisation check
            pairs = []
            for i in idx:
                j = int(np.argmax(np.maximum(Vs[i] @ Vtr.T, Vs[i] @ Vtr_f.T)))
                pairs.append(np.hstack([cv2.resize(S[s][i], (112, 112)), cv2.resize(np.asarray(X[tr[j]]), (112, 112))]))
            cv2.imwrite(str(out / f"closest_pairs_f{k}.jpg"), np.vstack(pairs))
        rows.append(r); print(r, flush=True)
    del model; torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(out / "qc_syn.csv", index=False)
