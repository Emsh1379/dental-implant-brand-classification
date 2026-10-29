#!/usr/bin/env python3
"""Lashaki et al. (BMC Oral Health 2025;25:535) protocol on the same Roboflow dataset, with and without leakage.

Their pipeline (Table 1/2): whole periapical image -> 128 x 128 -> sharpen (0,-1,0;-1,5,-1;0,-1,0) -> median 3x3 -> CLAHE (clip 2, tiles 3x3)
-> implant mask (here: the annotation boxes) -> histogram equalisation -> /255; 10 augmented versions per image (rotation +-20 deg, shifts and
shear 0.2, zoom 0.8-1.2, horizontal flip) generated BEFORE the 5-fold split; VGG16 (ImageNet, all but the last block frozen, GAP, Dense 256),
Adam 1e-4, early stopping (patience 15) on a 10 % validation split, <= 70 epochs; binary pairwise tasks (headline: Bicon vs Bego 98.65 %).
Deviation: batch 32 instead of 8 (speed).

Protocols:  leaky          = their protocol: all images x 10 versions, random stratified 5-fold over the versions
            grouped        = same preprocessing + mask, our source-grouped folds; versions of training images only; test = original primary image
            grouped_nomask = grouped, preprocessing without the annotation mask (the mask needs ground-truth boxes, unavailable at test time)
            grouped_raw    = grouped, resize only
Tasks: bicon_bego (binary, their headline) and 3class.
usage: python lashaki_protocol.py prep --out D ; python lashaki_protocol.py run --out D --jobs "task:protocol:fold" ...
"""
import argparse, json, os, sys, time, zipfile
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--out"); ap.add_argument("--jobs", nargs="*", default=[])
ap.add_argument("--crops", default="/tmp/data/crops"); ap.add_argument("--repo", default="/kaggle/working/repo")
a = ap.parse_args(); D = Path(a.out); D.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, f"{a.repo}/dental_cls")
from common import parse_boxes
NV = 10  # versions per image (original + 9 augmented)


def preprocess(g, boxes, mode):
    x = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    if mode == "raw": return x
    x = cv2.filter2D(x, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
    x = cv2.medianBlur(x, 3); x = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)).apply(x)
    if mode == "mask":
        m = np.zeros_like(x)
        for _, cx, cy, w, h in boxes:
            m[int(max(0, (cy - h / 2) * 128)):int(min(128, (cy + h / 2) * 128)) + 1, int(max(0, (cx - w / 2) * 128)):int(min(128, (cx + w / 2) * 128)) + 1] = 1
        x = x * m
    return cv2.equalizeHist(x)


def augment(x, rng):
    ang = rng.uniform(-20, 20); z = rng.uniform(0.8, 1.2); sh = rng.uniform(-0.2, 0.2); tx, ty = rng.uniform(-0.2, 0.2, 2) * 128
    M = cv2.getRotationMatrix2D((64, 64), ang, z); M[0, 1] += sh; M[:, 2] += (tx, ty)
    y = cv2.warpAffine(x, M, (128, 128), borderMode=cv2.BORDER_REPLICATE)
    return y[:, ::-1].copy() if rng.random() < 0.5 else y


if a.mode == "prep":
    import pandas as pd
    from huggingface_hub import hf_hub_download, HfFileSystem
    rf = "/tmp/data/rf/orig_v1_yolo"
    if not os.path.exists(rf):
        from roboflow import Roboflow
        Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace("dental-implant-nyw2h").project("-dental-implant").version(1).download("yolov8", location=rf)
    lab = Path("/tmp/data/hat_labels")
    if not lab.exists():
        with HfFileSystem().open("datasets/Mesh2001/Dental_implant/Teeth-Segmentation-ToothFairy-1-enahnced.zip", "rb", block_size=2 ** 22) as f:
            z = zipfile.ZipFile(f)
            for n in z.namelist():
                if n.endswith(".txt") and "/labels/" in n:
                    d = lab / n.split("/")[-3]; d.mkdir(parents=True, exist_ok=True); (d / Path(n).name).write_bytes(z.read(n))
    match = pd.read_csv(hf_hub_download("emad2001/dental-implant-brand-classification", "crops/raw_orig_match.csv", repo_type="dataset")).set_index("idx").original
    meta = pd.read_csv(f"{a.repo}/backup/crops/meta_groups.csv"); folds = [f for f in json.load(open(f"{a.crops}/folds.json"))["folds"] if f["rep"] == 0]
    test_fold = {}
    for f in folds:
        for i in f["test"]: test_fold[meta.cgroup[i]] = f["fold"]
    paths = {Path(p).name: str(p) for p in Path(rf).glob("*/images/*")}
    imgs = meta.groupby("image").agg(brand=("brand", "first"), group=("cgroup", "first"), primary=("primary", "first"), split=("split", "first"), idx=("idx", "first")).reset_index()
    imgs["fold"] = imgs.group.map(test_fold); assert imgs.fold.notna().all()
    out = {m: np.zeros((len(imgs), NV, 128, 128), np.uint8) for m in ("mask", "nomask", "raw")}
    rng = np.random.default_rng(0)
    for i, r in enumerate(imgs.itertuples()):
        g = cv2.imread(paths[match[r.idx]], cv2.IMREAD_GRAYSCALE); boxes = parse_boxes(lab / r.split / (Path(r.image).stem + ".txt"))
        for m in out:
            base = preprocess(g, boxes, m); out[m][i, 0] = base
            for v in range(1, NV): out[m][i, v] = augment(base, np.random.default_rng(1000 * i + v))
    for m, X in out.items(): np.save(D / f"img_{m}.npy", X)
    imgs.to_csv(D / "images.csv", index=False); print("images", len(imgs), imgs.brand.value_counts().to_dict(), "groups", imgs.group.nunique()); sys.exit()

imgs = pd.read_csv(D / "images.csv"); dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
cache = {}


def model(nout):
    import torchvision.models as tvm
    m = tvm.vgg16(weights="DEFAULT").features
    for p in m[:24].parameters(): p.requires_grad = False
    return nn.Sequential(m, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, nout))


for job in a.jobs:
    task, proto, k = job.split(":"); k = int(k); path = D / "results" / f"{task}_{proto}_f{k}.json"; path.parent.mkdir(exist_ok=True)
    if path.exists(): continue
    t0 = time.time(); mode = {"leaky": "mask", "grouped": "mask", "grouped_nomask": "nomask", "grouped_raw": "raw"}[proto]
    if mode not in cache: cache[mode] = np.load(D / f"img_{mode}.npy", mmap_mode="r")
    X = cache[mode]
    sel = imgs.brand.isin([1, 0]) if task == "bicon_bego" else imgs.brand.notna()
    I = np.where(sel)[0]; cls = {0: 1, 1: 0} if task == "bicon_bego" else {0: 0, 1: 1, 2: 2}  # binary: Bego = positive class
    ylab = imgs.brand.map(cls).fillna(-1).astype(int).values; nout = 1 if task == "bicon_bego" else 3  # recall order: binary [Bicon, Bego]; 3class [Bego, Bicon, ITI]
    rng = np.random.default_rng(k)
    if proto == "leaky":  # versions are separate samples, split at random
        pairs = np.array([(i, v) for i in I for v in range(NV)])
        skf = StratifiedKFold(5, shuffle=True, random_state=0); tr_te = list(skf.split(pairs, ylab[pairs[:, 0]]))[k]
        trp, tep = pairs[tr_te[0]], pairs[tr_te[1]]; perm = rng.permutation(len(trp)); nv = len(trp) // 10
        vap, trp = trp[perm[:nv]], trp[perm[nv:]]
    else:
        f = imgs.fold.values
        teI = I[(f[I] == k) & imgs.primary.values[I]]; trI = I[f[I] != k]
        gs = imgs.group.values; ug = np.unique(gs[trI]); vg = set(rng.choice(ug, len(ug) // 10, replace=False))
        vaI = trI[np.isin(gs[trI], list(vg))]; trI = trI[~np.isin(gs[trI], list(vg))]
        trp = np.array([(i, v) for i in trI for v in range(NV)]); vap = np.array([(i, 0) for i in vaI]); tep = np.array([(i, 0) for i in teI])
    def batch(p):
        x = torch.from_numpy(np.stack([X[i, v] for i, v in p])).to(dev).float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
        return (x - MEAN) / STD, torch.from_numpy(ylab[p[:, 0]]).to(dev)
    def loss_fn(o, y): return F.binary_cross_entropy_with_logits(o.squeeze(1), y.float()) if nout == 1 else F.cross_entropy(o, y)
    def predict(net, P):
        net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(P), 256):
                x, _ = batch(P[s:s + 256])
                with torch.autocast("cuda", dtype=torch.float16): o = net(x).float()
                out.append(torch.sigmoid(o.squeeze(1)) if nout == 1 else F.softmax(o, 1))
        return torch.cat(out).cpu().numpy()
    torch.manual_seed(k); net = model(nout).to(dev); opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], 1e-4)
    best, best_state, wait, ep_run = 1e9, None, 0, 0
    for ep in range(70):
        net.train(); perm = rng.permutation(len(trp))
        for s in range(0, len(perm), 32):
            x, y = batch(trp[perm[s:s + 32]])
            with torch.autocast("cuda", dtype=torch.float16): l = loss_fn(net(x).float(), y)
            opt.zero_grad(); l.backward(); opt.step()
        pv = predict(net, vap); yv = ylab[vap[:, 0]]
        vl = float(F.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6), torch.tensor(yv).float())) if nout == 1 else \
            float(F.nll_loss(torch.log(torch.tensor(pv).clamp_min(1e-6)), torch.tensor(yv)))
        ep_run = ep + 1
        if vl < best - 1e-4: best, best_state, wait = vl, {n: v.detach().clone() for n, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 15: break
    net.load_state_dict(best_state); P = predict(net, tep); y = ylab[tep[:, 0]]
    pred = (P > 0.5).astype(int) if nout == 1 else P.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in sorted(set(y))]
    auc = roc_auc_score(y, P) if nout == 1 else roc_auc_score(y, P, multi_class="ovr")
    res = dict(task=task, protocol=proto, fold=k, n_train=len(trp), n_test=len(tep), epochs=ep_run, acc=float((pred == y).mean()),
               macro_f1=float(f1_score(y, pred, average="macro")), recall=rec, auc=float(auc), test=tep[:, 0].tolist(), y=y.tolist(), p=np.asarray(P).tolist())
    json.dump(res, open(path, "w"))
    print(f"{task} {proto} f{k}: acc {res['acc']:.4f} macro-F1 {res['macro_f1']:.4f} AUC {res['auc']:.4f} recall {np.round(rec, 3).tolist()} "
          f"(train {len(trp)}, test {len(tep)}, {ep_run} ep, {time.time() - t0:.0f}s)", flush=True)
    del net, opt; torch.cuda.empty_cache()
