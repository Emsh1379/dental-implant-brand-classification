#!/usr/bin/env python3
"""Tekis et al. (Sci Rep 2025;15:33304) protocol: YOLO brand detection on the Roboflow export lamiaa/implant-recognition (pre-augmented copies,
Roboflow's own random train/valid/test split), against a split grouped by source radiograph.

usage: ROBOFLOW_API_KEY=... python tekis_yolo.py prep --out D
       python tekis_yolo.py train --out D --split rf|grouped [--model yolo11m.pt --epochs 100]
* rf      : Roboflow's split as published. Test images are also scored separately by whether their source radiograph has copies in training.
* grouped : the same images; whole source radiographs assigned to train / valid / test (same image counts as rf, stratified by brand, seed 0).
Classes: Bego, Bicon, ITI. Metrics: per-class and mean precision, recall, F1, mAP50, mAP50-95 (Ultralytics val on the test set).
"""
import argparse, json, os, re, shutil
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--out"); ap.add_argument("--split", default="rf")
ap.add_argument("--model", default="yolo11m.pt"); ap.add_argument("--epochs", type=int, default=100)
a = ap.parse_args(); D = Path(a.out); D.mkdir(parents=True, exist_ok=True)
NAMES = ["Bego", "Bicon", "ITI"]; src = lambda n: re.split(r"_(?:jpg|jpeg|png)\.rf\.", n)[0].lower()

if a.mode == "prep":
    L = Path("/tmp/data/rf/lamiaa_v3")
    if not L.exists():
        from roboflow import Roboflow
        Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace("lamiaa").project("implant-recognition").version(3).download("coco", location=str(L))
    rows = []
    for sp in ("train", "valid", "test"):
        c = json.load(open(L / sp / "_annotations.coco.json")); cats = {k["id"]: k["name"] for k in c["categories"]}
        anns = {}
        for an in c["annotations"]:
            if cats[an["category_id"]] in NAMES: anns.setdefault(an["image_id"], []).append(an)
        for im in c["images"]:
            if im["id"] not in anns: continue  # 18 images have no annotation
            W, H = im["width"], im["height"]
            lab = "".join(f"{NAMES.index(cats[an['category_id']])} {(an['bbox'][0] + an['bbox'][2] / 2) / W:.6f} {(an['bbox'][1] + an['bbox'][3] / 2) / H:.6f} "
                          f"{an['bbox'][2] / W:.6f} {an['bbox'][3] / H:.6f}\n" for an in anns[im["id"]])
            rows.append(dict(file=im["file_name"], rf_split=sp, src=src(im["file_name"]), label=lab, brand=NAMES.index(cats[anns[im["id"]][0]["category_id"]])))
    T = pd.DataFrame(rows)
    # grouped split with the same image counts per split, whole sources, stratified by brand
    rng = np.random.default_rng(0); T["grouped_split"] = ""
    for b, tb in T.groupby("brand"):
        srcs = tb.groupby("src").size().sample(frac=1, random_state=int(rng.integers(1e6)))
        want = tb.rf_split.value_counts(); n_te, n_va = want.get("test", 0), want.get("valid", 0); got = {"test": 0, "valid": 0}
        for s, n in srcs.items():
            sp = "test" if got["test"] < n_te else ("valid" if got["valid"] < n_va else "train")
            if sp != "train": got[sp] += n
            T.loc[(T.brand == b) & (T.src == s), "grouped_split"] = sp
    train_src = set(T[T.rf_split == "train"].src)
    T["rf_test_source_in_train"] = T.src.isin(train_src)
    T.to_csv(D / "images.csv", index=False)
    for split_col in ("rf_split", "grouped_split"):
        root = D / split_col
        for r in T.itertuples():
            sp = getattr(r, split_col); (root / sp / "images").mkdir(parents=True, exist_ok=True); (root / sp / "labels").mkdir(parents=True, exist_ok=True)
            dst = root / sp / "images" / r.file
            if not dst.exists(): os.symlink(L / r.rf_split / r.file, dst)
            (root / sp / "labels" / (Path(r.file).stem + ".txt")).write_text(r.label)
        (root / "data.yaml").write_text(f"path: {root}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames: {NAMES}\n")
    for seen in (True, False):  # rf test images split by whether their radiograph has copies in rf training
        root = D / f"rf_test_seen{int(seen)}"
        for sub in ("images", "labels"): (root / "test" / sub).mkdir(parents=True, exist_ok=True)
        for r in T[(T.rf_split == "test") & (T.rf_test_source_in_train == seen)].itertuples():
            dst = root / "test" / "images" / r.file
            if not dst.exists(): os.symlink(L / "test" / r.file, dst)
            (root / "test" / "labels" / (Path(r.file).stem + ".txt")).write_text(r.label)
        (root / "data.yaml").write_text(f"path: {root}\ntrain: test/images\nval: test/images\ntest: test/images\nnames: {NAMES}\n")
    print("images", len(T), "| sources", T.src.nunique(), "| rf split", T.rf_split.value_counts().to_dict(), "| grouped split", T.grouped_split.value_counts().to_dict())
    print("rf test images whose source has copies in rf train:", int(T[T.rf_split == 'test'].rf_test_source_in_train.sum()), "of", int((T.rf_split == 'test').sum()))
    print("sources in more than one rf split:", int((T.groupby('src').rf_split.nunique() > 1).sum()), "| grouped:", int((T.groupby('src').grouped_split.nunique() > 1).sum()))
else:
    from ultralytics import YOLO
    col = "rf_split" if a.split == "rf" else "grouped_split"
    m = YOLO(a.model)
    m.train(data=str(D / col / "data.yaml"), epochs=a.epochs, imgsz=640, batch=16, project=str(D / "runs"), name=a.split, exist_ok=True, verbose=False, plots=False, workers=4)
    best = YOLO(str(D / "runs" / a.split / "weights" / "best.pt")); out = {}
    evals = [("test", D / col / "data.yaml")] + ([("test_seen_source", D / "rf_test_seen1" / "data.yaml"), ("test_unseen_source", D / "rf_test_seen0" / "data.yaml")] if a.split == "rf" else [])
    for name, yml in evals:
        r = best.val(data=str(yml), split="test", imgsz=640, batch=16, verbose=False, plots=False, project=str(D / "runs"), name=f"{a.split}_{name}", exist_ok=True)
        p, rc = r.box.p, r.box.r; f1 = 2 * p * rc / np.maximum(p + rc, 1e-9)
        out[name] = dict(n_images=len(list(Path(yml).parent.joinpath("test/images").iterdir())), mP=float(p.mean()), mR=float(rc.mean()), mF1=float(f1.mean()),
                         mAP50=float(r.box.map50), mAP50_95=float(r.box.map),
                         per_class={NAMES[int(c)]: dict(P=float(p[i]), R=float(rc[i]), F1=float(f1[i]), AP50=float(r.box.ap50[i])) for i, c in enumerate(r.box.ap_class_index)})
        print(a.split, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items() if k != "per_class"}, flush=True)
    json.dump(out, open(D / f"result_{a.split}.json", "w"), indent=1)
