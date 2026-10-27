#!/usr/bin/env python3
"""Find implants in unannotated panoramics with the 5 YOLO11n fold detectors and cut crops (GPU, batched).

usage: python detect_crops.py <image dir> <checkpoint dir (det_checkpoints/yolo11n_f*/weights/best.pt)> <out dir> [max images]
Same rule as label_pano.py: square tiles of 1/3 image height (50 % overlap) resized to 640 px; boxes cut by a tile border are dropped;
a box is kept when >= 3 of the 5 models find it (IoU > 0.5); overlapping boxes reduced to the best one; snapped to the implant metal.
Crops: common.crop_box + letterbox from the full-resolution grey image, saved as <image stem>_<k>.jpg; crops.csv lists boxes and votes.
"""
import csv, glob, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import crop_box, letterbox

SRC, CK, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); MAXN = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
OUT.mkdir(parents=True, exist_ok=True)
models = [YOLO(str(CK / f"det_checkpoints/yolo11n_f{k}/weights/best.pt")) for k in range(5)]
src = open(Path(__file__).with_name("label_new.py")).read()
exec(src[src.index("def iou"):src.index("coco = dict(")])  # iou(), snap()
files = sorted(p for p in glob.glob(f"{SRC}/**/*.*", recursive=True) if p.lower().endswith((".png", ".jpg", ".jpeg")))[:MAXN]
rows = []
for n, p in enumerate(files):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None: continue
    H, W = g.shape; s = int(H / 3); step = s // 2
    tiles, offs = [], []
    for y0 in list(range(0, max(1, H - s), step)) + [H - s]:
        for x0 in list(range(0, max(1, W - s), step)) + [W - s]:
            tiles.append(cv2.cvtColor(cv2.resize(g[y0:y0 + s, x0:x0 + s], (640, 640), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)); offs.append((x0, y0))
    dets = []
    for k, m in enumerate(models):
        for r, (x0, y0) in zip(m.predict(tiles, imgsz=640, conf=0.25, verbose=False, batch=32), offs):
            for b, c, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if min(b[0], b[1], 640 - b[2], 640 - b[3]) >= 8:
                    dets.append((k, b * s / 640 + np.array([x0, y0, x0, y0]), float(c), int(cl)))
    dets.sort(key=lambda d: -d[2]); clusters = []
    for d in dets:
        for cl in clusters:
            if iou(cl[0][1], d[1])[0] > 0.5: cl.append(d); break
        else: clusters.append([d])
    cand = sorted([cl for cl in clusters if len({d[0] for d in cl}) >= 3], key=lambda cl: (-len({d[0] for d in cl}), -np.mean([d[2] for d in cl])))
    keep = []
    for cl in cand:
        w = np.array([d[2] for d in cl]); b = (np.stack([d[1] for d in cl]) * w[:, None]).sum(0) / w.sum()
        if all(iou(b, q[0])[0] <= 0.3 and iou(b, q[0])[1] <= 0.7 for q in keep): keep.append((b, len({d[0] for d in cl}), float(w.mean())))
    for j, (b, nm, cf) in enumerate(keep):
        sb, ok = snap(g, b); nb = ((sb[0] + sb[2]) / 2 / W, (sb[1] + sb[3]) / 2 / H, (sb[2] - sb[0]) / W, (sb[3] - sb[1]) / H)
        f = f"{Path(p).stem}_{j}.jpg"; cv2.imwrite(str(OUT / f), letterbox(crop_box(g, nb)))
        rows.append(dict(file=f, image=str(Path(p).relative_to(SRC)), folder=Path(p).parent.name, x1=round(sb[0]), y1=round(sb[1]), x2=round(sb[2]), y2=round(sb[3]),
                         models=nm, conf=round(cf, 3), snapped=ok, W=W, H=H))
    if n % 200 == 0: print(f"{n}/{len(files)} images, {len(rows)} implants", flush=True)
with open(OUT / "crops.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("DONE", len(rows), "implant crops from", len({r['image'] for r in rows}), "of", len(files), "images")
