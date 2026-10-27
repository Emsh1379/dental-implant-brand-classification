"""Auto-label user-supplied Bego radiographs (several implants per image possible).

usage: python -I label_new.py <image dir> <checkpoint dir> <out dir> <file> [<file> ...]
Each image -> grey, stretched to 640x640 (like the Roboflow export). The 5 YOLO11n fold models run on it; a box is kept when >= 3 models agree
(IoU > 0.5, conf >= 0.2); overlapping consensus boxes (IoU > 0.3 or > 70 % contained) are reduced to the best one. Each box is snapped to the
implant metal (as in refine_boxes.py). Label = Bego for every box.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

SRC, CK, OUT = map(Path, sys.argv[1:4]); FILES = sys.argv[4:]
models = [YOLO(str(CK / f"det_checkpoints/yolo11n_f{k}/weights/best.pt")) for k in range(5)]
for d in ("images", "labels", "preview"): (OUT / d).mkdir(parents=True, exist_ok=True)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i), i / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def snap(g, b):
    H, W = g.shape; x1, y1, x2, y2 = b
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.3, (y2 - y1) * 1.3
    X1, Y1, X2, Y2 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(W, cx + bw / 2)), int(min(H, cy + bh / 2))
    roi = cv2.GaussianBlur(g[Y1:Y2, X1:X2], (5, 5), 0)
    t, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (roi > 0.5 * (t + np.percentile(roi, 99.5))).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    det = np.zeros_like(m); det[int(y1) - Y1:int(y2) - Y1, int(x1) - X1:int(x2) - X1] = 1
    ov = [((lab == i) & (det > 0)).sum() for i in range(n)]
    if n > 1:
        i = int(np.argmax(ov[1:]) + 1); sx, sy, sw, sh = st[i, :4]
        nb = (X1 + sx, Y1 + sy, X1 + sx + sw, Y1 + sy + sh)
        if ov[i] > 0.3 * st[i, cv2.CC_STAT_AREA] and 0.5 < sw * sh / ((x2 - x1) * (y2 - y1)) < 2.5: return [float(v) for v in nb], True
    return [float(v) for v in b], False


coco = dict(images=[], annotations=[], categories=[{"id": 0, "name": "Bego"}, {"id": 1, "name": "Bicon"}, {"id": 2, "name": "ITI"}])
tiles = []
for iid, f in enumerate(FILES):
    g = cv2.resize(cv2.imread(str(SRC / f), cv2.IMREAD_GRAYSCALE), (640, 640), interpolation=cv2.INTER_AREA)
    name = f"ubego-{iid:02d}.png"; cv2.imwrite(str(OUT / "images" / name), g)
    im3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); dets = []
    for k, m in enumerate(models):
        r = m.predict(im3, imgsz=640, conf=0.2, verbose=False)[0]
        dets += [(k, b, float(s), int(c)) for b, s, c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy())]
    dets.sort(key=lambda d: -d[2]); clusters = []
    for d in dets:
        for cl in clusters:
            if iou(cl[0][1], d[1])[0] > 0.5 and d[0] not in {x[0] for x in cl}: cl.append(d); break
        else: clusters.append([d])
    cand = sorted([cl for cl in clusters if len(cl) >= 3], key=lambda cl: (-len(cl), -np.mean([d[2] for d in cl])))
    keep = []
    for cl in cand:
        w = np.array([d[2] for d in cl]); b = (np.stack([d[1] for d in cl]) * w[:, None]).sum(0) / w.sum()
        if all(iou(b, k[0])[0] <= 0.3 and iou(b, k[0])[1] <= 0.7 for k in keep):
            keep.append((b, len(cl), float(w.mean()), np.bincount([d[3] for d in cl], minlength=3)))
    coco["images"].append(dict(id=iid, file_name=name, width=640, height=640, source_file=f))
    lines = []
    for b, n, conf, vote in keep:
        sb, ok = snap(g, b)
        lines.append(f"0 {(sb[0] + sb[2]) / 1280:.6f} {(sb[1] + sb[3]) / 1280:.6f} {(sb[2] - sb[0]) / 640:.6f} {(sb[3] - sb[1]) / 640:.6f}\n")
        coco["annotations"].append(dict(id=len(coco["annotations"]), image_id=iid, category_id=0, iscrowd=0, bbox=[sb[0], sb[1], sb[2] - sb[0], sb[3] - sb[1]],
                                        area=(sb[2] - sb[0]) * (sb[3] - sb[1]), models=n, conf=round(conf, 3), vote=vote.tolist(), snapped=ok))
        cv2.rectangle(im3, (int(sb[0]), int(sb[1])), (int(sb[2]), int(sb[3])), (0, 255, 0), 3)
        cv2.putText(im3, f"{n}/5 B{vote[0]} I{vote[2]}", (int(sb[0]), max(20, int(sb[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    (OUT / "labels" / (Path(name).stem + ".txt")).write_text("".join(lines))
    cv2.putText(im3, f"{name} <- {f[:22]}", (8, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / "preview" / name.replace(".png", ".jpg")), im3); tiles.append(cv2.resize(im3, (400, 400)))
    print(name, f, "boxes", len(keep), [(n, round(c, 2), v.tolist()) for _, n, c, v in keep])
json.dump(coco, open(OUT / "_annotations.coco.json", "w"), indent=1)
while len(tiles) % 5: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(OUT / "contact_sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
