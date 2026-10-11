#!/usr/bin/env python3
"""Two-stage system (detector boxes -> crop classifier) and box-jitter robustness of the classifier, on the same folds.

Needs per fold: the detection set from det_prep.py, the detector checkpoint from det_train.py and the classifier
checkpoint from `train_cv.py --save_all`. Per fold writes <out>/two_stage_<detector>_f<k>.json and <out>/jitter_f<k>.json.

Two-stage, GT-matched (same rule as det_train.py): each ground-truth implant takes the highest-scoring detection with
IoU >= 0.5; brand = detector class (one-stage) or classifier on the crop of that detected box (two-stage); no detection = error.
Two-stage, operating point: detections above a score threshold after class-agnostic NMS, matched one-to-one to implants;
per-brand TP / FP / FN with the detector's label and with the classifier's label.
Jitter: ground-truth boxes are scaled, shifted or have their edges perturbed before cropping."""
import argparse, json, os, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import crop_box, letterbox, parse_boxes
from models import build
from train_cv import metrics, predict

ap = argparse.ArgumentParser()
ap.add_argument("--det", default="/tmp/data/det")
ap.add_argument("--src", default="/tmp/data/toothfairy")
ap.add_argument("--crops", default="/tmp/data/crops")
ap.add_argument("--det_ckpt", default="/tmp/data/det_runs")
ap.add_argument("--cls_ckpt", default="/tmp/data/results_ts_ckpt")
ap.add_argument("--out", default="/kaggle/working/ts_results")
ap.add_argument("--version", default="raw384"); ap.add_argument("--model", default="densenet121"); ap.add_argument("--size", type=int, default=384)
ap.add_argument("--detectors", nargs="+", default=["rfdetr_small", "yolo11n"])
ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
ap.add_argument("--thr", type=float, nargs="+", default=[0.3, 0.5])
a = ap.parse_args()
DET, OUT, RES, dev = Path(a.det), Path(a.out), 640, torch.device("cuda:0")
OUT.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(f"{a.crops}/meta.csv", index_col="idx")
folds = {f["fold"]: f for f in json.load(open(f"{a.crops}/folds.json"))["folds"] if f["rep"] == 0}
row_of = {(im, b): i for i, im, b in zip(meta.index, meta.image, meta.box_idx)}


def iou(p, q):  # xywh
    iw = max(0, min(p[0] + p[2], q[0] + q[2]) - max(p[0], q[0])); ih = max(0, min(p[1] + p[3], q[1] + q[3]) - max(p[1], q[1]))
    return iw * ih / (p[2] * p[3] + q[2] * q[3] - iw * ih + 1e-9)


def crop(img, b):  # b: xywh in pixels of the 640px image -> letterboxed crop, same code path as prep_variants.py
    return letterbox(crop_box(img, ((b[0] + b[2] / 2) / RES, (b[1] + b[3] / 2) / RES, b[2] / RES, b[3] / RES)), a.size)


def classify(model, crops):
    return predict(model, torch.from_numpy(np.stack(crops)), np.arange(len(crops)), dev) if len(crops) else np.zeros((0, 3), np.float32)


def detect(name, k, gt):
    cache = OUT / f"dets_{name}_f{k}.json"
    if cache.exists():
        return json.load(open(cache))
    root, dets = DET / f"fold{k}" / "coco" / "test", []
    if name == "rfdetr_small":
        from rfdetr import RFDETRSmall
        m = RFDETRSmall(pretrain_weights=f"{a.det_ckpt}/rfdetr_small_f{k}/checkpoint_best_total.pth")
        for im in gt["images"]:
            d = m.predict(Image.open(root / im["file_name"]).convert("RGB"), threshold=0.001)
            dets += [dict(image_id=im["id"], category_id=int(c), bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s))
                     for (x1, y1, x2, y2), s, c in zip(d.xyxy.tolist(), d.confidence.tolist(), d.class_id.tolist())]
    else:
        from ultralytics import YOLO
        m = YOLO(f"{a.det_ckpt}/{name}_f{k}/weights/best.pt")
        for im in gt["images"]:
            r = m.predict(str(root / im["file_name"]), conf=0.001, iou=0.7, max_det=300, imgsz=640, device=0, verbose=False)[0]
            dets += [dict(image_id=im["id"], category_id=int(c), bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s))
                     for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    del m; torch.cuda.empty_cache()
    json.dump(dets, open(cache, "w"))
    return dets


def prf(tp, fp, fn):
    return [dict(tp=int(t), fp=int(p), fn=int(n)) for t, p, n in zip(tp, fp, fn)]


def two_stage(name, k, model, gt, imgs):
    dets = detect(name, k, gt)
    by_img = {}
    for d in dets: by_img.setdefault(d["image_id"], []).append(d)
    fname = {im["id"]: im["file_name"] for im in gt["images"]}
    # --- GT-matched
    rows, crops = [], []
    seen = {}
    for g in sorted(gt["annotations"], key=lambda g: g["id"]):
        b = seen[g["image_id"]] = seen.get(g["image_id"], -1) + 1  # box_idx = order of the annotation inside its image
        cands = [d for d in by_img.get(g["image_id"], []) if iou(g["bbox"], d["bbox"]) >= 0.5]
        best = max(cands, key=lambda d: d["score"]) if cands else None
        rows.append(dict(idx=int(row_of[(fname[g["image_id"]], b)]), y=g["category_id"], det=best["category_id"] if best else -1,
                         det_score=best["score"] if best else None, iou=iou(g["bbox"], best["bbox"]) if best else None))
        crops.append(crop(imgs[fname[g["image_id"]]], best["bbox"] if best else g["bbox"]))
    p = classify(model, crops)
    for r, pi in zip(rows, p):
        r["two_stage"] = int(pi.argmax()) if r["det"] >= 0 else -1; r["probs"] = pi.round(5).tolist() if r["det"] >= 0 else None
    assert all(meta.brand[r["idx"]] == r["y"] for r in rows)
    y = np.array([r["y"] for r in rows]); out = dict(detector=name, fold=k, n=len(rows), implants=rows, operating=[])
    for key in ("det", "two_stage"):
        yh = np.array([r[key] for r in rows])
        out[f"{key}_acc"] = float((yh == y).mean()); out[f"{key}_recall"] = [float(((yh == c) & (y == c)).sum() / max((y == c).sum(), 1)) for c in range(3)]
    out["missed"] = int(sum(r["det"] < 0 for r in rows)); out["mean_iou_matched"] = float(np.mean([r["iou"] for r in rows if r["iou"] is not None]))
    # --- operating point: class-agnostic NMS, one-to-one matching in score order
    gts = {}
    for g in gt["annotations"]: gts.setdefault(g["image_id"], []).append(g)
    for thr in a.thr:
        kept = []
        for iid, ds in by_img.items():
            keep = []
            for d in sorted((d for d in ds if d["score"] >= thr), key=lambda d: -d["score"]):
                if all(iou(d["bbox"], q["bbox"]) < 0.5 for q in keep): keep.append(d)
            used = set()
            for d in keep:
                c = [(iou(d["bbox"], g["bbox"]), g) for g in gts.get(iid, []) if g["id"] not in used]
                c = [x for x in c if x[0] >= 0.5]
                g = max(c, key=lambda x: x[0])[1] if c else None
                if g: used.add(g["id"])
                kept.append((d, g, iid))
        pc = classify(model, [crop(imgs[fname[iid]], d["bbox"]) for d, _, iid in kept]).argmax(1) if kept else []
        n_gt = np.bincount([g["category_id"] for g in gt["annotations"]], minlength=3)
        res = dict(thr=thr, n_dets=len(kept), n_unmatched_dets=int(sum(g is None for _, g, _ in kept)), n_gt=n_gt.tolist())
        for key, lab in (("det", [d["category_id"] for d, _, _ in kept]), ("two_stage", [int(c) for c in pc])):
            tp, fp = np.zeros(3, int), np.zeros(3, int)
            for (d, g, _), c in zip(kept, lab):
                if g is not None and g["category_id"] == c: tp[c] += 1
                else: fp[c] += 1
            res[key] = prf(tp, fp, n_gt - tp)
        out["operating"].append(res)
    json.dump(out, open(OUT / f"two_stage_{name}_f{k}.json", "w"))
    print(f"{name} fold{k}: one-stage acc={out['det_acc']:.3f} two-stage acc={out['two_stage_acc']:.3f} missed={out['missed']} "
          f"Bego recall {out['det_recall'][0]:.3f} -> {out['two_stage_recall'][0]:.3f}  mean IoU={out['mean_iou_matched']:.3f}", flush=True)


def jitter(k, model, te, gt_box, imgs, X):
    """gt_box[i] = xywh pixels of test crop i (meta row te[i])."""
    y = meta.brand.values[te]; rng = np.random.default_rng(k)
    B = np.array([gt_box[i] for i in te], float); cx, cy, w, h = B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2, B[:, 2], B[:, 3]
    conds = [("scale", s, 0) for s in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)]
    conds += [("shift", s, r) for s in (0.05, 0.1, 0.15, 0.2, 0.3) for r in range(3)]
    conds += [("edges", s, r) for s in (0.03, 0.06, 0.1, 0.15) for r in range(3)]
    out = dict(fold=k, test_idx=[int(i) for i in te], y=y.tolist(), conditions=[])
    for kind, lv, rep in conds:
        if kind == "scale":
            x1, y1, x2, y2 = cx - w * lv / 2, cy - h * lv / 2, cx + w * lv / 2, cy + h * lv / 2
        elif kind == "shift":  # centre moved by lv x box size in a random direction
            t = rng.uniform(0, 2 * np.pi, len(te)); dx, dy = lv * w * np.cos(t), lv * h * np.sin(t)
            x1, y1, x2, y2 = cx - w / 2 + dx, cy - h / 2 + dy, cx + w / 2 + dx, cy + h / 2 + dy
        else:  # every edge moved independently, SD = lv x box size
            e = rng.normal(0, lv, (4, len(te)))
            x1, y1, x2, y2 = cx - w / 2 + e[0] * w, cy - h / 2 + e[1] * h, cx + w / 2 + e[2] * w, cy + h / 2 + e[3] * h
        x1, y1 = np.clip(x1, 0, RES - 2), np.clip(y1, 0, RES - 2); x2, y2 = np.clip(x2, x1 + 1, RES), np.clip(y2, y1 + 1, RES)
        J = np.stack([x1, y1, x2 - x1, y2 - y1], 1)
        p = classify(model, [crop(imgs[meta.image[i]], b) for i, b in zip(te, J)])
        m = metrics(y, p)
        out["conditions"].append(dict(kind=kind, level=lv, rep=rep, mean_iou=float(np.mean([iou(b, j) for b, j in zip(B, J)])),
                                      pred=p.argmax(1).tolist(), acc=m["acc"], macro_f1=m["macro_f1"], recall=m["recall"]))
        if kind == "scale" and lv == 1.0:  # must reproduce the stored crops and therefore the GT-crop result
            same = np.mean([np.array_equal(crop(imgs[meta.image[i]], b), X[i]) for i, b in zip(te, B)])
            out["crops_identical_to_stored"] = float(same); out["gt_probs"] = p.round(5).tolist()
            print(f"fold{k}: GT-box crops identical to {a.version}.npy for {same:.1%}; acc={m['acc']:.3f} macro-F1={m['macro_f1']:.3f}", flush=True)
    json.dump(out, open(OUT / f"jitter_f{k}.json", "w"))
    print(f"fold{k} jitter done: " + "  ".join(f"{c['kind']}{c['level']}:{c['macro_f1']:.3f}" for c in out["conditions"] if c["rep"] == 0), flush=True)


X = np.load(f"{a.crops}/{a.version}.npy", mmap_mode="r")
for k in a.folds:
    gt = json.load(open(DET / f"fold{k}" / "coco" / "test" / "_annotations.coco.json"))
    imgs = {im["file_name"]: cv2.imread(str(DET / "images" / im["file_name"]), cv2.IMREAD_GRAYSCALE) for im in gt["images"]}
    model, _ = build(a.model, pretrained=False)
    model.load_state_dict(torch.load(f"{a.cls_ckpt}/r0_f{k}_{a.version}_{a.model}.pt", map_location="cpu"))
    model = model.to(dev).to(memory_format=torch.channels_last)
    te = np.array(folds[k]["test"])
    # exact label boxes (not the rounded COCO ones) for the jitter test, so that jitter 0 reproduces the stored crops
    gt_box = {}
    for name in imgs:
        sp = meta.split[row_of[(name, 0)]]
        for b, (_, x, y, w, h) in enumerate(parse_boxes(Path(a.src) / sp / "labels" / (Path(name).stem + ".txt"))):
            gt_box[row_of[(name, b)]] = [(x - w / 2) * RES, (y - h / 2) * RES, w * RES, h * RES]
    assert set(te) == set(gt_box), "test crops and detection test implants differ"
    if not (OUT / f"jitter_f{k}.json").exists():
        jitter(k, model, te, gt_box, imgs, X)
    for name in a.detectors:
        if not (OUT / f"two_stage_{name}_f{k}.json").exists():
            two_stage(name, k, model, gt, imgs)
    del model; torch.cuda.empty_cache()
print("TWO-STAGE DONE", flush=True)
