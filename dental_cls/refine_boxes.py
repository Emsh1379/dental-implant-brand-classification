"""One implant per image: keep the best consensus box and snap it to the implant metal.

usage: python -I refine_boxes.py <out dir from auto_annotate.py>
The detector box is enlarged by 30 %; inside it the metal is segmented (halfway between the Otsu threshold and the 99.5th percentile), the connected component that
overlaps the detector box most is taken, and its bounding box becomes the label. Falls back to the detector box if segmentation fails.
"""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np

OUT = Path(sys.argv[1])
rows = list(csv.DictReader(open(OUT / "auto_boxes.csv")))
best = {}
for r in rows:  # most models first, then confidence
    k = (int(r["models"]), float(r["mean_conf"]))
    if r["image"] not in best or k > best[r["image"]][0]: best[r["image"]] = (k, r)
coco = dict(images=[], annotations=[], categories=[{"id": 0, "name": "Bego"}, {"id": 1, "name": "Bicon"}, {"id": 2, "name": "ITI"}])
tiles, report = [], []
for iid, (name, (_, r)) in enumerate(sorted(best.items())):
    g = cv2.imread(str(OUT / "images" / name), cv2.IMREAD_GRAYSCALE); H, W = g.shape
    x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.3, (y2 - y1) * 1.3
    X1, Y1, X2, Y2 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(W, cx + bw / 2)), int(min(H, cy + bh / 2))
    roi = cv2.GaussianBlur(g[Y1:Y2, X1:X2], (5, 5), 0)
    t, m = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t2 = 0.5 * (t + np.percentile(roi, 99.5)); m = (roi > t2).astype(np.uint8) * 255  # metal is the brightest tail, above bone
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    det = np.zeros_like(m); det[int(y1) - Y1:int(y2) - Y1, int(x1) - X1:int(x2) - X1] = 1
    ov = [((lab == i) & (det > 0)).sum() for i in range(n)]
    i = int(np.argmax(ov[1:]) + 1) if n > 1 else 0
    ok = n > 1 and ov[i] > 0.3 * st[i, cv2.CC_STAT_AREA]
    if ok:
        sx, sy, sw, sh = st[i, :4]; b = (X1 + sx, Y1 + sy, X1 + sx + sw, Y1 + sy + sh)
        a = (b[2] - b[0]) * (b[3] - b[1]); ok = 0.5 < a / ((x2 - x1) * (y2 - y1)) < 2.5  # reject leaks into neighbouring teeth
    if not ok: b = (x1, y1, x2, y2)
    b = [float(v) for v in b]
    report.append((name[:40], "snapped" if ok else "detector box", int(r["models"]), r["mean_conf"]))
    (OUT / "labels" / (Path(name).stem + ".txt")).write_text(
        f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n")
    coco["images"].append(dict(id=iid, file_name=name, width=W, height=H))
    coco["annotations"].append(dict(id=iid, image_id=iid, category_id=0, iscrowd=0, bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]], area=(b[2] - b[0]) * (b[3] - b[1])))
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(v, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 1)
    cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)
    cv2.putText(v, name.split("_jpg")[0] + (" snapped" if ok else " det"), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(str(OUT / "preview" / (Path(name).stem + ".jpg")), v); tiles.append(cv2.resize(v, (320, 320)))
json.dump(coco, open(OUT / "_annotations.coco.json", "w"), indent=1)
while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(OUT / "contact_sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
for x in report: print(*x)

