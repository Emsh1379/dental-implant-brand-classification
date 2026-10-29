#!/usr/bin/env python3
"""Class-conditional diffusion for implant-brand crops, after ThirdMolarSynthesis (s4nyam, 3m-cdiff/train_cuda.py).

Kept from the repository: 128 x 128 diffusion U-Net (diffusers), class-conditional, DDPM with 1000 steps, squaredcos_cap_v2 schedule, epsilon/MSE loss.
Changed (see ClassConditionedUnet): brand added to the timestep embedding at every level instead of the repository's cross-attention context
(which only reaches the mid block), 4 levels (32-256 channels) with attention at 16 px only (6x faster).
Also: 1 grey channel, 3 brands (+1 "null" label for classifier-free guidance: label dropped 10 % of the time),
batch 32, AdamW 2e-4, fp16 autocast, EMA 0.999 of the weights, horizontal flips; sampling with DDIM (100 steps) and guidance scale --cfg.
No pretrained weights are released with the repository, so the model is trained from scratch on each fold's TRAINING crops only.

usage: python cdiff_brand.py train  --fold K --crops <dir: raw_orig.npy, meta.csv, folds.json> --out <dir> --epochs N
       python cdiff_brand.py sample --fold K --crops ... --out <dir> --n_bego 400 --n_bicon 0 --n_iti 0 [--cfg 2.0]
"""
import argparse, copy, json, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel

ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("--fold", type=int); ap.add_argument("--crops"); ap.add_argument("--out")
ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--n_bego", type=int, default=400); ap.add_argument("--n_bicon", type=int, default=0); ap.add_argument("--n_iti", type=int, default=0)
ap.add_argument("--cfg", type=float, default=2.0); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); NULL = 3


class ClassConditionedUnet(nn.Module):
    """The repository's UNet2DConditionModel uses AttnDownBlock2D/AttnUpBlock2D, which have no cross-attention, so its class context reaches the
    network only through the mid block (weak conditioning), and self-attention at 64 px makes it ~6x slower. Here the brand is added to the
    timestep embedding at every level (diffusers' standard class conditioning, num_class_embeds) and attention is kept at 16 px only."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = UNet2DModel(sample_size=128, in_channels=1, out_channels=1, layers_per_block=2, block_out_channels=(32, 64, 128, 256),
                                 down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                                 up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"), num_class_embeds=num_classes)

    def forward(self, x, t, y):
        return self.model(x, t, class_labels=y).sample


meta = pd.read_csv(f"{a.crops}/meta.csv"); fd = next(f for f in json.load(open(f"{a.crops}/folds.json"))["folds"] if f["rep"] == 0 and f["fold"] == a.fold)
tr = np.array([i for i in fd["train"] if i < 3276])  # original crops of this fold's training set only
torch.manual_seed(a.seed); np.random.seed(a.seed)
ck = out / f"cdiff_f{a.fold}.pt"

if a.mode == "train":
    X = np.load(f"{a.crops}/raw_orig.npy", mmap_mode="r")
    X = torch.from_numpy(np.stack([cv2.resize(np.asarray(X[i]), (128, 128), interpolation=cv2.INTER_AREA) for i in tr])).float().div(127.5).sub(1).unsqueeze(1)
    Y = torch.from_numpy(meta.brand.values[tr]).long()
    print(f"fold {a.fold}: {len(tr)} training crops, per brand {np.bincount(Y.numpy(), minlength=3).tolist()}", flush=True)
    net = ClassConditionedUnet().to(dev); ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    scaler = torch.amp.GradScaler(); steps = a.epochs * int(np.ceil(len(X) / a.bs))
    lrs = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.02); t0 = time.time(); hist = []
    for ep in range(a.epochs):
        perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), a.bs):
            b = perm[i:i + a.bs]; x = X[b].to(dev); y = Y[b].to(dev)
            x = torch.where(torch.rand(len(b), 1, 1, 1, device=dev) < 0.5, x.flip(3), x)
            y = torch.where(torch.rand(len(b), device=dev) < 0.1, torch.full_like(y, NULL), y)
            noise = torch.randn_like(x); t = torch.randint(0, 1000, (len(b),), device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.mse_loss(net(sched.add_noise(x, noise, t), t, y).float(), noise)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); lrs.step(); tot += loss.item() * len(b)
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.999).add_(pn.detach(), alpha=0.001)
        hist.append(tot / len(X))
        if ep % 25 == 0 or ep == a.epochs - 1: print(f"fold {a.fold} epoch {ep} loss {hist[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(dict(ema=ema.state_dict(), hist=hist, epochs=a.epochs), ck); print("saved", ck, flush=True)

else:  # sample
    net = ClassConditionedUnet().to(dev).eval(); net.load_state_dict(torch.load(ck, map_location=dev)["ema"])
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True); ddim.set_timesteps(a.steps)
    want = [(0, a.n_bego), (1, a.n_bicon), (2, a.n_iti)]; imgs, labs = [], []
    g = torch.Generator(device=dev).manual_seed(1000 + a.fold)
    for c, n in want:
        for s in range(0, n, 64):
            m = min(64, n - s); x = torch.randn(m, 1, 128, 128, device=dev, generator=g)
            yc = torch.full((m,), c, device=dev); yn = torch.full((m,), NULL, device=dev)
            for t in ddim.timesteps:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    e = net(torch.cat([x, x]), torch.cat([t.expand(m), t.expand(m)]).to(dev), torch.cat([yc, yn])).float()
                ec, eu = e.chunk(2); x = ddim.step(eu + a.cfg * (ec - eu), t, x).prev_sample
            imgs.append(((x.clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(1).cpu().numpy()); labs += [c] * m
    S = np.concatenate(imgs); S224 = np.stack([cv2.resize(s, (224, 224), interpolation=cv2.INTER_CUBIC) for s in S])
    np.save(out / f"syn_f{a.fold}.npy", S224); np.save(out / f"syn_f{a.fold}_labels.npy", np.array(labs))
    tiles = [cv2.resize(s, (112, 112)) for s in S[:48]]
    cv2.imwrite(str(out / f"syn_f{a.fold}.jpg"), np.vstack([np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)]))
    print(f"fold {a.fold}: generated {len(S)} images, per brand {np.bincount(labs, minlength=3).tolist()}", flush=True)
