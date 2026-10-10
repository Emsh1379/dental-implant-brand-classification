#!/usr/bin/env python3
"""Train YOLO11n and RF-DETR-small on the 5 group-stratified folds and evaluate both with the same COCO evaluator.

Jobs are claimed through lock files, so two workers (one per GPU) share the queue; finished jobs are skipped.
Per job: /kaggle/working/det_results/<model>_f<k>.json with COCO metrics, per-class AP and brand accuracy
of the detector (each ground-truth implant is assigned the class of the highest-scoring prediction with IoU>=0.5;
an implant with no such prediction counts as missed)."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import f1_score

DET, RUNS, RES = Path("/tmp/data/det"), Path(os.environ.get("DET_RUNS", "/tmp/data/det_runs")), Path(os.environ.get("DET_RES", "/kaggle/working/det_results"))
EPOCHS = {"yolo11n": 30, "yolo11n_gtroi": 30, "rfdetr_small": 30}
if os.environ.get("DET_EPOCHS"):  # smoke test
    EPOCHS = {k: int(os.environ["DET_EPOCHS"]) for k in EPOCHS}


def iou(a, b):  # xywh
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def evaluate(gt_path, dets):
    gt = COCO(str(gt_path))
    out = {"n_images": len(gt.imgs), "n_boxes": len(gt.anns)}
    dt = gt.loadRes(dets) if dets else None
    E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    out.update(mAP=E.stats[0], mAP50=E.stats[1], mAP75=E.stats[2], AR100=E.stats[8])
    prec = E.eval["precision"]  # [T,R,K,A,M]
    for k, c in enumerate(gt.getCatIds()):
        name = gt.cats[c]["name"]
        p = prec[:, :, k, 0, -1]; p50 = prec[0, :, k, 0, -1]
        out[f"AP_{name}"] = float(p[p > -1].mean()); out[f"AP50_{name}"] = float(p50[p50 > -1].mean())
    # brand classification from detections
    by_img = {}
    for d in dets: by_img.setdefault(d["image_id"], []).append(d)
    y, yhat = [], []
    for a in gt.anns.values():
        cands = [d for d in by_img.get(a["image_id"], []) if iou(a["bbox"], d["bbox"]) >= 0.5]
        y.append(a["category_id"]); yhat.append(max(cands, key=lambda d: d["score"])["category_id"] if cands else -1)
    y, yhat = np.array(y), np.array(yhat)
    out.update(det_brand_acc=float((y == yhat).mean()), det_missed=int((yhat == -1).sum()),
               det_brand_macro_f1=float(f1_score(y, yhat, labels=[0, 1, 2], average="macro", zero_division=0)),
               det_brand_recall=[float(((yhat == c) & (y == c)).sum() / max((y == c).sum(), 1)) for c in range(3)])
    return out


def run_yolo(k, dev, det=DET, tag="yolo11n"):
    from ultralytics import YOLO
    root = det / f"fold{k}"
    m = YOLO("yolo11n.pt")
    m.train(data=str(root / "yolo" / "data.yaml"), epochs=EPOCHS[tag], imgsz=640, batch=16, device=dev, seed=k,
            project=str(RUNS), name=f"{tag}_f{k}", exist_ok=True, workers=4, plots=False, verbose=False)
    best = YOLO(str(RUNS / f"{tag}_f{k}" / "weights" / "best.pt"))
    gt = json.load(open(root / "coco" / "test" / "_annotations.coco.json"))
    dets = []
    for im in gt["images"]:
        r = best.predict(str(root / "coco" / "test" / im["file_name"]), conf=0.001, iou=0.7, max_det=300, imgsz=640, device=dev, verbose=False)[0]
        for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            dets.append(dict(image_id=im["id"], category_id=int(c), bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s)))
    return dets


def run_rfdetr(k, dev):
    from rfdetr import RFDETRSmall
    root, out = DET / f"fold{k}", RUNS / f"rfdetr_small_f{k}"
    m = RFDETRSmall()
    m.train(dataset_dir=str(root / "coco"), epochs=EPOCHS["rfdetr_small"], batch_size=8, grad_accum_steps=2,
            lr=1e-4, output_dir=str(out), seed=k, early_stopping=False,
            num_workers=4, tensorboard=False, progress_bar=False)
    best = RFDETRSmall(pretrain_weights=str(out / "checkpoint_best_total.pth"))
    gt = json.load(open(root / "coco" / "test" / "_annotations.coco.json"))
    dets = []
    for im in gt["images"]:
        d = best.predict(Image.open(root / "coco" / "test" / im["file_name"]).convert("RGB"), threshold=0.001)
        for (x1, y1, x2, y2), s, c in zip(d.xyxy.tolist(), d.confidence.tolist(), d.class_id.tolist()):
            dets.append(dict(image_id=im["id"], category_id=int(c), bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s)))
    return dets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", nargs="+", default=["yolo11n", "rfdetr_small"])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
    jobs = [(m, k) for m in a.order for k in a.folds]
    for m, k in jobs:
        res, lock = RES / f"{m}_f{k}.json", RES / f".{m}_f{k}.lock"
        if res.exists():
            continue
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        t0 = time.time()
        det = Path("/tmp/data/det_roi") if m == "yolo11n_gtroi" else DET
        if m == "rfdetr_small":
            dets = run_rfdetr(k, 0)
        else:
            dets = run_yolo(k, 0, det, m)
        r = evaluate(det / f"fold{k}" / "coco" / "test" / "_annotations.coco.json", dets)
        r.update(model=m, fold=k, seconds=time.time() - t0, n_dets=len(dets))
        json.dump(r, open(res, "w"), indent=1)
        print(f"{m} fold{k}: mAP50={r['mAP50']:.3f} mAP={r['mAP']:.3f} brand_acc={r['det_brand_acc']:.3f} ({r['seconds']/60:.1f} min)", flush=True)
    print("WORKER DONE", flush=True)
