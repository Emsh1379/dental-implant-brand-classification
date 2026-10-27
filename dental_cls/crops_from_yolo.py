#!/usr/bin/env python3
"""Cut implant crops from a YOLO-format dataset that has an implant class (e.g. the Zenodo panoramic dataset, 14 classes).

usage: python crops_from_yolo.py <dataset root> <out dir> [class-name substring (default 'implant') or class id]
Finds data.yaml, the class whose name contains the substring, and every image/label pair below the root.
Crops use the same margin and letterboxing as our crops (common.crop_box / letterbox) from the full-resolution grey image;
saved as <image stem>_<k>.jpg, plus crops.csv (file, image, split, box, image size).
"""
import csv, glob, os, sys
from pathlib import Path
import cv2, yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import crop_box, letterbox, parse_boxes

root, out = Path(sys.argv[1]), Path(sys.argv[2]); key = sys.argv[3] if len(sys.argv) > 3 else "implant"
out.mkdir(parents=True, exist_ok=True)
if key.isdigit():  # class id given directly (dataset without data.yaml, e.g. Zenodo: legend.txt says Implant = 0)
    cid = [int(key)]
else:
    ys = glob.glob(f"{root}/**/data.yaml", recursive=True) + glob.glob(f"{root}/**/*.yaml", recursive=True)
    names = yaml.safe_load(open(ys[0]))["names"]; names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
    cid = [i for i, n in enumerate(names) if key in str(n).lower()]
    print("yaml", ys[0], "| classes", names)
print("implant class", cid)
rows = []
for p in sorted(glob.glob(f"{root}/**/*.*", recursive=True)):
    if not p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) or "/labels/" in p: continue
    lab = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
    if not lab.exists(): continue
    boxes = [b for b in parse_boxes(lab) if b[0] in cid]
    if not boxes: continue
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); H, W = g.shape
    for k, b in enumerate(boxes):
        f = f"{Path(p).stem}_{k}.jpg"; cv2.imwrite(str(out / f), letterbox(crop_box(g, b[1:])))
        rows.append(dict(file=f, image=Path(p).name, split=Path(p).parts[-3] if "/images/" in p else "", cx=b[1], cy=b[2], w=b[3], h=b[4], W=W, H=H,
                         box_px=f"{b[3] * W:.0f}x{b[4] * H:.0f}"))
with open(out / "crops.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "implant crops from", len({r['image'] for r in rows}), "images")
