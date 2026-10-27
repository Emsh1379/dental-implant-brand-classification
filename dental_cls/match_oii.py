#!/usr/bin/env python3
"""Find OII-DS single-implant crops that look like our brands (Bego / Bicon / ITI). No brand labels exist in OII-DS.

usage: python match_oii.py --oii <single_implants dir> --crops <dir with raw_orig.npy, meta.csv, folds.json> --ckpt <dir with s0_ckpt, s1_ckpt, s2_ckpt> --out <dir>
Models: the fold-0 DenseNet121 and ViT-B/16 checkpoints of 3 seeds (6 models, trained on raw_orig crops).
For every crop: mean softmax of the 6 models, model agreement, and k-NN (k = 10, cosine) against the fold-0 TRAINING crops in each model's feature space.
Out-of-distribution check: the nearest-neighbour similarity must reach the 5th percentile of that of our own fold-0 TEST crops (unseen by the models).
A crop is a candidate for brand c when: all 6 models predict c, mean probability >= that brand's 10th percentile of correct predictions on our test crops, >= 8 of 10 neighbours are c in both feature spaces,
and the NN similarity passes the OOD threshold in both feature spaces. The same rule is applied to our fold-0 test crops to measure its precision.
"""
import argparse, glob, json, os, re, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
for k in ("oii", "crops", "ckpt", "out", "code"): ap.add_argument("--" + k)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, a.code)
from common import letterbox
from models import build
NAMES = ["Bego", "Bicon", "ITI"]; dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

meta = pd.read_csv(f"{a.crops}/meta.csv").iloc[:3276]  # original dataset rows only (reliable brand labels)
X = np.load(f"{a.crops}/raw_orig.npy", mmap_mode="r")[:3276]
fd = next(f for f in json.load(open(f"{a.crops}/folds.json"))["folds"] if f["rep"] == 0 and f["fold"] == 0)
tr = np.array([i for i in fd["train"] if i < 3276]); te = np.array(fd["test"]); y = meta.brand.values

files = sorted(glob.glob(f"{a.oii}/*.jpg"))  # any folder of implant crops (OII-DS single_implants, or crops_from_yolo / detect_crops output)
O = np.stack([letterbox(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in files])
print("OII crops", O.shape, "| reference train", len(tr), "test", len(te), flush=True)


@torch.no_grad()
def run(model, head, A):
    feats, probs = [], []
    h = head.register_forward_hook(lambda m, i, o: feats.append(i[0].float()))
    for s in range(0, len(A), 128):
        x = torch.from_numpy(np.ascontiguousarray(A[s:s + 128])).to(dev).float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
        with torch.autocast("cuda", dtype=torch.float16):
            probs.append(F.softmax(model((x - MEAN) / STD).float(), 1))
    h.remove()
    return F.normalize(torch.cat(feats), dim=1), torch.cat(probs).cpu().numpy()


res = {}  # model tag -> dict
for s in (0, 1, 2):
    for name in ("densenet121", "vit_b_16"):
        model, head = build(name, pretrained=False)
        model.load_state_dict(torch.load(f"{a.ckpt}/s{s}_ckpt/r0_f0_raw_orig_{name}.pt", map_location="cpu")); model.to(dev).eval()
        fr, _ = run(model, head, X[tr]); ft, pt = run(model, head, X[te]); fo, po = run(model, head, O)
        def knn(q, k=10):
            S = q @ fr.T; v, i = S.topk(k, dim=1); return v[:, 0].cpu().numpy(), y[tr][i.cpu().numpy()]
        st, nt = knn(ft); so, no = knn(fo)
        res[(s, name)] = dict(pt=pt, po=po, st=st, so=so, vt=np.stack([(nt == c).sum(1) for c in range(3)], 1), vo=np.stack([(no == c).sum(1) for c in range(3)], 1))
        print(f"seed {s} {name}: test acc {(pt.argmax(1) == y[te]).mean():.3f}, NN-sim test median {np.median(st):.3f} / OII median {np.median(so):.3f}", flush=True)


PT = np.mean([r["pt"] for r in res.values()], 0)  # per-brand probability threshold: 10th percentile of correct predictions on our test crops
PTHR = np.array([np.percentile(PT[(PT.argmax(1) == c) & (y[te] == c), c], 10) for c in range(3)])
print("probability thresholds per brand:", dict(zip(NAMES, PTHR.round(3))))


def score(which):
    P = np.mean([r["p" + which] for r in res.values()], 0); pred = P.argmax(1)
    agree = np.all([r["p" + which].argmax(1) == pred for r in res.values()], 0)
    ok = {}
    for fam in ("densenet121", "vit_b_16"):
        rs = [r for (s, n), r in res.items() if n == fam]
        sim = np.mean([r["s" + which] for r in rs], 0); vote = np.mean([r["v" + which] for r in rs], 0)
        thr = np.percentile(np.mean([r["st"] for r in rs], 0), 5)
        ok[fam] = dict(sim=sim, thr=thr, vote=vote[np.arange(len(pred)), pred])
    cand = agree & (P.max(1) >= PTHR[pred]) & np.all([(ok[f]["vote"] >= 8) & (ok[f]["sim"] >= ok[f]["thr"]) for f in ok], 0)
    return P, pred, agree, ok, cand


Pt, predt, agt, okt, ct = score("t"); yt = y[te]
print("\nRule on our fold-0 test crops (unseen): kept", int(ct.sum()), "of", len(te))
for c in range(3):
    k = ct & (predt == c)
    print(f"  {NAMES[c]}: candidates {k.sum()}, correct {int((yt[k] == c).sum())} (precision {((yt[k] == c).mean() if k.any() else float('nan')):.3f}), "
          f"recall of true {NAMES[c]} {((ct & (yt == c) & (predt == c)).sum() / (yt == c).sum()):.3f}")
Po, predo, ago, oko, co = score("o")
grp = [re.sub(r"_\d+\.jpg$", "", os.path.basename(f)) for f in files]; pat = [g.split("-")[0] for g in grp]
D = pd.DataFrame(dict(file=[os.path.basename(f) for f in files], ct_image=grp, patient=pat, pred=[NAMES[i] for i in predo], prob=Po.max(1).round(4),
                      p_Bego=Po[:, 0].round(4), p_Bicon=Po[:, 1].round(4), p_ITI=Po[:, 2].round(4), models_agree=ago,
                      knn_dn=oko["densenet121"]["vote"].round(1), knn_vit=oko["vit_b_16"]["vote"].round(1),
                      nn_sim_dn=oko["densenet121"]["sim"].round(4), nn_sim_vit=oko["vit_b_16"]["sim"].round(4),
                      in_distribution=(oko["densenet121"]["sim"] >= oko["densenet121"]["thr"]) & (oko["vit_b_16"]["sim"] >= oko["vit_b_16"]["thr"]), candidate=co))
D.to_csv(out / "oii_scores.csv", index=False)
print(f"\nOII-DS single implants: {len(D)} crops from {D.ct_image.nunique()} CT images / {D.patient.nunique()} patients")
print("OOD thresholds (NN cosine sim):", {f: round(float(oko[f]["thr"]), 3) for f in oko}, "| in-distribution crops:", int(D.in_distribution.sum()))
print("ensemble prediction, all crops:", D.pred.value_counts().to_dict())
print("candidates:", D[D.candidate].pred.value_counts().to_dict(), "| patients per candidate class:", D[D.candidate].groupby("pred").patient.nunique().to_dict())
for c in NAMES:  # contact sheets: top 48 candidates per brand, ranked by probability x mean NN similarity
    d = D[D.candidate & (D.pred == c)].copy(); d["rank"] = d.prob * (d.nn_sim_dn + d.nn_sim_vit) / 2; d = d.sort_values("rank", ascending=False)
    d.to_csv(out / f"candidates_{c}.csv", index=False)
    if len(d) == 0: continue
    tiles = []
    for f, pr in zip(d.file[:48], d.prob[:48]):
        t = cv2.resize(letterbox(cv2.imread(f"{a.oii}/{f}", cv2.IMREAD_GRAYSCALE)), (150, 150)); cv2.putText(t, f"{pr:.2f}", (3, 14), 0, 0.45, 255, 1); tiles.append(t)
    ref = [cv2.resize(X[i], (150, 150)) for i in np.random.default_rng(0).choice(np.where(y[tr] == NAMES.index(c))[0], 8, replace=False)]
    while len(tiles) % 8: tiles.append(np.zeros((150, 150), np.uint8))
    sheet = np.vstack([np.hstack(ref)] + [np.full((6, 1200), 255, np.uint8)] + [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
    cv2.imwrite(str(out / f"sheet_{c}.jpg"), sheet)
print("DONE")
