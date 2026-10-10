#!/usr/bin/env python3
"""Group-stratified k-fold CV for pretrained classifiers on three input versions.
Jobs are (rep, fold, version, model); results are one JSON per job so runs are resumable and shardable."""
import argparse, json, os, time
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

from models import HPARAMS, MODELS, build

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prep(x, dev):  # uint8 [B,224,224] -> normalised float [B,3,224,224]
    x = x.to(dev).float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
    return (x - MEAN.to(dev)) / STD.to(dev)


def augment(x):  # x: normalised [B,3,H,W] on GPU; hflip, small affine, brightness/contrast
    B = x.shape[0]
    dev = x.device
    ang = (torch.rand(B, device=dev) - .5) * 2 * np.deg2rad(10)
    sc = 1 + (torch.rand(B, device=dev) - .5) * 0.2
    tx, ty = [(torch.rand(B, device=dev) - .5) * 0.1 for _ in range(2)]
    flip = (torch.rand(B, device=dev) < .5).float() * 2 - 1
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) - .5) * 0.4
    b = (torch.rand(B, 1, 1, 1, device=dev) - .5) * 0.4
    return x * c + b


@torch.no_grad()
def predict(model, X, idx, dev, bs=64):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(F.softmax(model(prep(X[idx[i:i + bs]], dev)).float(), 1).cpu())
    return torch.cat(out).numpy()


def metrics(y, p):
    pred = p.argmax(1)
    f1s = f1_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)
    rec = np.diag(confusion_matrix(y, pred, labels=[0, 1, 2])) / np.maximum(np.bincount(y, minlength=3), 1)
    return dict(acc=float((pred == y).mean()), bal_acc=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1s.mean()), f1=f1s.tolist(), recall=rec.tolist())


def run_job(job, X, meta, folds, outdir, dev, ckpt_dir):
    rep, fold, ver, name = job
    tag = f"r{rep}_f{fold}_{ver}_{name}"
    path = outdir / f"{tag}.json"
    if path.exists():
        return
    fd = next(f for f in folds if f["rep"] == rep and f["fold"] == fold)
    y_all = torch.tensor(meta.brand.values)
    tr, va, te = (np.array(fd[k]) for k in ("train", "val", "test"))
    torch.manual_seed(1000 * rep + fold); np.random.seed(1000 * rep + fold)
    epochs, lr_b, lr_h = HPARAMS[name]
    model, head = build(name)
    model = model.to(dev).to(memory_format=torch.channels_last)
    hp = {id(p) for p in head.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in hp], "lr": lr_b},
                             {"params": head.parameters(), "lr": lr_h}], weight_decay=1e-2)
    bs = 32
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [lr_b, lr_h], total_steps=steps, pct_start=0.15)
    # class-weighted CE replaces the synthetic oversampling used previously
    cnt = np.bincount(meta.brand.values[tr], minlength=3)
    w = torch.tensor(len(tr) / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    scaler = torch.amp.GradScaler()
    best, best_state, hist, t0 = -1, None, [], time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr); tl = 0
        for i in range(0, len(perm) - bs + 1, bs):
            b = perm[i:i + bs]
            x = augment(prep(X[b], dev)); y = y_all[b].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * bs
        pv = predict(model, X, va, dev)
        m = metrics(y_all[va].numpy(), pv)
        score = m["macro_f1"] + 1e-3 * m["acc"]
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(perm), val_acc=m["acc"], val_macro_f1=m["macro_f1"]))
        if score > best:
            best = score; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pt = predict(model, X, te, dev)
    res = dict(job=tag, rep=rep, fold=fold, version=ver, model=name, test_idx=te.tolist(), y=y_all[te].tolist(),
               probs=pt.round(5).tolist(), metrics=metrics(y_all[te].numpy(), pt), history=hist,
               train_counts=cnt.tolist(), seconds=time.time() - t0)
    if rep == 0 and fold == 0:
        torch.save(best_state, ckpt_dir / f"{tag}.pt")
    json.dump(res, open(path, "w"))
    print(f"{tag}: acc={res['metrics']['acc']:.3f} f1={res['metrics']['macro_f1']:.3f} ({res['seconds']:.0f}s)", flush=True)
    del model, opt; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="/tmp/data/crops")
    ap.add_argument("--out", default="/kaggle/working/results")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--reps", type=int, nargs="+", default=[0])
    ap.add_argument("--versions", nargs="+", default=["raw", "pipeline", "hat"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()
    dev = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES selects the GPU per shard
    torch.backends.cudnn.benchmark = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.out + "_ckpt"); ck.mkdir(exist_ok=True)
    meta = pd.read_csv(f"{a.crops}/meta.csv", index_col="idx")
    folds = json.load(open(f"{a.crops}/folds.json"))["folds"]
    jobs = [(r, f, v, m) for r in a.reps for f in range(5) for v in a.versions for m in a.models]
    jobs = jobs[a.shard::a.nshards]
    cache = {}
    for job in jobs:
        v = job[2]
        if v not in cache:
            cache = {v: torch.from_numpy(np.load(f"{a.crops}/{v}.npy"))}
        run_job(job, cache[v], meta, folds, out, dev, ck)
    print("SHARD DONE", flush=True)
