#!/usr/bin/env python3
"""Grad-CAM for the fold-0 checkpoints: 2 test crops per brand, all versions side by side per model."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import MODELS, build
from train_cv import prep

ap = argparse.ArgumentParser()
ap.add_argument("--crops", default="/tmp/data/crops"); ap.add_argument("--ck", default="/kaggle/working/results_ckpt")
ap.add_argument("--out", default="/kaggle/working/report/gradcam")
a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda:0")
meta = pd.read_csv(f"{a.crops}/meta.csv", index_col="idx")
fd = next(f for f in json.load(open(f"{a.crops}/folds.json"))["folds"] if f["rep"] == 0 and f["fold"] == 0)
te = np.array(fd["test"]); rng = np.random.default_rng(0)
pick = [int(i) for c in range(3) for i in rng.choice(te[meta.brand.values[te] == c], 2, replace=False)]

def target_layer(name, m):
    return {"custom_cnn": lambda: m.features[-4], "vgg16": lambda: m.features[-2], "resnet50": lambda: m.layer4[-1],
            "densenet121": lambda: m.features[-1], "efficientnet_b0": lambda: m.features[-1],
            "vit_b_16": lambda: m.encoder.layers[-1].ln_1}[name]()

def reshape_vit(t):  # [B,197,768] -> [B,768,14,14]
    return t[:, 1:, :].reshape(t.size(0), 14, 14, -1).permute(0, 3, 1, 2)

for name in MODELS:
    fig, axs = plt.subplots(len(pick), 3 * 2, figsize=(12, 2 * len(pick)))
    for vi, ver in enumerate(["raw", "pipeline", "hat"]):
        X = torch.from_numpy(np.load(f"{a.crops}/{ver}.npy", mmap_mode="r")[pick])
        m, _ = build(name, pretrained=False)
        m.load_state_dict(torch.load(f"{a.ck}/r0_f0_{ver}_{name}.pt", map_location="cpu")); m = m.to(dev).eval()
        cam = GradCAM(model=m, target_layers=[target_layer(name, m)], reshape_transform=reshape_vit if name == "vit_b_16" else None)
        x = prep(X, dev)
        with torch.no_grad(): pred = m(x).argmax(1).cpu().numpy()
        heat = cam(input_tensor=x, targets=None)  # explains each prediction
        for r in range(len(pick)):
            g = X[r].numpy().astype(np.float32) / 255
            axs[r, 2 * vi].imshow(g, cmap="gray"); axs[r, 2 * vi + 1].imshow(show_cam_on_image(np.stack([g] * 3, -1), heat[r], use_rgb=True))
            axs[r, 2 * vi].set_ylabel(f"true {['Bego','Bicon','ITI'][meta.brand.values[pick[r]]]}", fontsize=7)
            axs[r, 2 * vi + 1].set_title(f"pred {['Bego','Bicon','ITI'][pred[r]]}", fontsize=7)
            if r == 0: axs[r, 2 * vi].set_title(ver, fontsize=8)
    for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(name); fig.tight_layout(); fig.savefig(f"{a.out}/gradcam_{name}.png", dpi=150); plt.close(fig)
    print("done", name, flush=True)
