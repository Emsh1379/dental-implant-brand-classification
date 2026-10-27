"""Auto-annotate the 18 unlabelled Bego images of lamiaa/implant-recognition v3 with the 5 YOLO11n fold models (consensus boxes).

usage: python -I auto_annotate.py <lamiaa_v3 dir> <checkpoint dir> <out dir>
A box is kept when >= 3 of the 5 models find it (IoU > 0.5, conf >= 0.25); its coordinates are the confidence-weighted mean.
Brand label = Bego (from the file name); the detectors' own brand vote is recorded as a check.
"""
import json, sys, shutil
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

L, CK, OUT = map(Path, sys.argv[1:4])
NAMES = ["Bego", "Bicon", "ITI"]
models = [YOLO(str(CK / f"det_checkpoints/yolo11n_f{k}/weights/best.pt")) for k in range(5)]
imgs = []
for sp in ("train", "valid", "test"):
    c = json.load(open(L / sp / "_annotations.coco.json"))
    has = {a["image_id"] for a in c["annotations"]}
    imgs += [(sp, i["file_name"]) for i in c["images"] if i["id"] not in has]
print(len(imgs), "unlabelled images")


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


for d in ("images", "labels", "preview"): (OUT / d).mkdir(parents=True, exist_ok=True)
coco = dict(images=[], annotations=[], categories=[{"id": i, "name": n} for i, n in enumerate(NAMES)])
rows, tiles = [], []
for iid, (sp, name) in enumerate(imgs):
    g = cv2.imread(str(L / sp / name), cv2.IMREAD_GRAYSCALE); H, W = g.shape
    im3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    dets = []  # (model, xyxy, conf, cls)
    for k, m in enumerate(models):
        r = m.predict(im3, imgsz=640, conf=0.25, verbose=False)[0]
        for b, s, c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy()):
            dets.append((k, b, float(s), int(c)))
    dets.sort(key=lambda d: -d[2]); clusters = []
    for d in dets:
        for cl in clusters:
            if iou(cl[0][1], d[1]) > 0.5 and d[0] not in {x[0] for x in cl}: cl.append(d); break
        else: clusters.append([d])
    keep = [cl for cl in clusters if len(cl) >= 3]
    coco["images"].append(dict(id=iid, file_name=name, width=W, height=H))
    lines = []
    for cl in keep:
        w = np.array([d[2] for d in cl]); b = (np.stack([d[1] for d in cl]) * w[:, None]).sum(0) / w.sum()
        vote = np.bincount([d[3] for d in cl], minlength=3)
        x, y, bw, bh = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H
        lines.append(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")
        coco["annotations"].append(dict(id=len(coco["annotations"]), image_id=iid, category_id=0, iscrowd=0,
                                        bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], area=float((b[2] - b[0]) * (b[3] - b[1]))))
        rows.append(dict(image=name, split=sp, models=len(cl), mean_conf=round(float(w.mean()), 3), vote_Bego=int(vote[0]), vote_Bicon=int(vote[1]), vote_ITI=int(vote[2]),
                         x1=round(float(b[0]), 1), y1=round(float(b[1]), 1), x2=round(float(b[2]), 1), y2=round(float(b[3]), 1)))
        cv2.rectangle(im3, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
        cv2.putText(im3, f"{len(cl)}/5 {w.mean():.2f}", (int(b[0]), max(20, int(b[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for cl in clusters:
        if len(cl) < 3:  # rejected candidates, for review
            b = cl[0][1]; cv2.rectangle(im3, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 255), 1)
    shutil.copy(L / sp / name, OUT / "images" / name)
    (OUT / "labels" / (Path(name).stem + ".txt")).write_text("".join(lines))
    cv2.putText(im3, f"{name.split('_jpg')[0]} {sp}: {len(keep)} box", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(str(OUT / "preview" / (Path(name).stem + ".jpg")), im3); tiles.append(cv2.resize(im3, (320, 320)))
    print(name[:48], sp, "boxes kept", len(keep), "rejected", len(clusters) - len(keep), [(r["models"], r["mean_conf"], (r["vote_Bego"], r["vote_Bicon"], r["vote_ITI"])) for r in rows if r["image"] == name])
json.dump(coco, open(OUT / "_annotations.coco.json", "w"), indent=1)
import csv
with open(OUT / "auto_boxes.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(OUT / "contact_sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
print("boxes", len(rows))
