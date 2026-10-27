#!/usr/bin/env python3
"""Tiryaki et al. (J Prosthet Dent 2023) pipeline on our dataset.

Brand classification: VGG-16, VGG-19, ResNet-50, ResNet-101, GoogLeNet (ImageNet weights, last layer -> 3 classes), tight crops stretched to
224 x 224 (paper.npy), no augmentation, SGD (lr 1e-3, momentum 0.9, weight decay 1e-4), batch 32, 50 epochs, model after the last epoch
(the paper reports no validation-based selection). Fusion = majority vote of the 5 models (analyze_paper.py).
Angle: ResNet-101 regression. Our periapicals come in every orientation, so the target is (cos 2θ, sin 2θ) and the error is taken modulo 180°;
only crops whose metal is clearly elongated (elong >= 2 in angles.csv) are used.

Splits: --split group  = our fixed patient-grouped folds (rep 0; train + val used for training, test = primary crop of each test group)
        --split random = random stratified 5-fold over all crops (like the paper; crops of the same radiograph and Roboflow copies can fall on both sides)
usage: python train_paper.py --task cls|angle --models vgg16 ... --split group|random --folds 0 1 2 3 4 --data /tmp/data/paper --crops <dir with meta.csv, folds.json> --out <dir>
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="cls"); ap.add_argument("--models", nargs="+"); ap.add_argument("--split", default="group")
ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--data"); ap.add_argument("--crops"); ap.add_argument("--out")
ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
X = torch.from_numpy(np.load(f"{a.data}/paper.npy")); meta = pd.read_csv(f"{a.crops}/meta.csv").iloc[:len(X)]; y_all = meta.brand.values
A = pd.read_csv(f"{a.data}/angles.csv").set_index("idx"); ok_angle = (A.elong >= 2) & A.angle.notna()


def build(name, n):
    w = "DEFAULT"
    if name in ("vgg16", "vgg19"):
        m = getattr(tvm, name)(weights=w); m.classifier[-1] = nn.Linear(4096, n)
    elif name in ("resnet50", "resnet101"):
        m = getattr(tvm, name)(weights=w); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "googlenet":
        m = tvm.googlenet(weights=w); m.fc = nn.Linear(1024, n)
    return m


def splits():
    if a.split == "group":
        F_ = [f for f in json.load(open(f"{a.crops}/folds.json"))["folds"] if f["rep"] == 0]
        for f in F_:
            yield f["fold"], np.array([i for i in f["train"] + f["val"] if i < len(X)]), np.array(f["test"])
    else:
        sk = StratifiedKFold(5, shuffle=True, random_state=0)
        for k, (tr, te) in enumerate(sk.split(np.zeros(len(y_all)), y_all)): yield k, tr, te


def prep(idx):
    x = X[idx].to(dev).float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
    return (x - MEAN) / STD


def target(idx):
    if a.task == "cls": return torch.from_numpy(y_all[idx]).to(dev)
    t = np.radians(A.loc[idx, "angle"].values) * 2
    return torch.tensor(np.stack([np.cos(t), np.sin(t)], 1), dtype=torch.float32, device=dev)


for name in a.models:
    for k, tr, te in splits():
        if k not in a.folds: continue
        tag = f"{a.task}_{a.split}_f{k}_{name}"; path = out / f"{tag}.json"
        if path.exists(): continue
        if a.task == "angle":
            tr = tr[ok_angle.reindex(tr).fillna(False).values]; te = te[ok_angle.reindex(te).fillna(False).values]
        torch.manual_seed(k); np.random.seed(k)
        model = build(name, 3 if a.task == "cls" else 2).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4); scaler = torch.amp.GradScaler(); t0 = time.time()
        for ep in range(a.epochs):
            model.train(); perm = np.random.permutation(tr)
            for i in range(0, len(perm), a.bs):
                b = perm[i:i + a.bs]
                if len(b) < 2: continue
                with torch.autocast("cuda", dtype=torch.float16):
                    o = model(prep(b))
                    if not torch.is_tensor(o): o = o.logits
                    loss = F.cross_entropy(o.float(), target(b)) if a.task == "cls" else F.mse_loss(o.float(), target(b))
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); P = []
        with torch.no_grad():
            for i in range(0, len(te), 128):
                with torch.autocast("cuda", dtype=torch.float16):
                    o = model(prep(te[i:i + 128])).float()
                P.append(F.softmax(o, 1).cpu() if a.task == "cls" else o.cpu())
        P = torch.cat(P).numpy()
        if a.task == "cls":
            y = y_all[te]; pred = P.argmax(1); rec = np.diag(confusion_matrix(y, pred, labels=[0, 1, 2])) / np.maximum(np.bincount(y, minlength=3), 1)
            met = dict(acc=float((pred == y).mean()), bal_acc=float(balanced_accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                       f1=f1_score(y, pred, average=None, labels=[0, 1, 2]).tolist(), recall=rec.tolist())
            res = dict(job=tag, task=a.task, split=a.split, fold=k, model=name, test_idx=te.tolist(), y=y.tolist(), probs=P.tolist(), metrics=met)
        else:
            true = A.loc[te, "angle"].values; pa = np.degrees(np.arctan2(P[:, 1], P[:, 0])) / 2
            err = (pa - true + 90) % 180 - 90
            met = dict(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))), n=len(te))
            res = dict(job=tag, task=a.task, split=a.split, fold=k, model=name, test_idx=te.tolist(), true=true.tolist(), pred=pa.tolist(), metrics=met)
        json.dump(res, open(path, "w")); print(tag, {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in met.items() if k2 not in ("f1", "recall")},
                                              f"({time.time() - t0:.0f}s)", flush=True)
        del model, opt; torch.cuda.empty_cache()
