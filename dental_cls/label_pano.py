"""Auto-label implants in panoramic images (user-supplied Bego set).

usage: python -I label_pano.py <image dir> <checkpoint dir> <out dir> <file> [<file> ...]
Implants in a panoramic are small, so the 5 YOLO11n fold models run on overlapping square tiles (side = 1/3 and 1/2 of the image height,
50 % overlap), each resized to 640 px. A box is kept when >= 3 models agree at the same place (IoU > 0.5); overlapping boxes are merged to the
best-supported one. Boxes are snapped to the implant metal on the full-resolution grey image. Label = Bego.
Images are kept at full resolution (crops are cut from them directly); labels are YOLO-normalised to the full image.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

SRC, CK, OUT = map(Path, sys.argv[1:4]); FILES = sys.argv[4:]
models = [YOLO(str(CK / f"det_checkpoints/yolo11n_f{k}/weights/best.pt")) for k in range(5)]
for d in ("images", "labels", "preview"): (OUT / d).mkdir(parents=True, exist_ok=True)
src = open(Path(__file__).with_name("label_new.py")).read()
exec(src[src.index("def iou"):src.index("coco = dict(")])  # iou(), snap()

coco = dict(images=[], annotations=[], categories=[{"id": 0, "name": "Bego"}, {"id": 1, "name": "Bicon"}, {"id": 2, "name": "ITI"}])
for iid, f in enumerate(FILES):
    g = cv2.imread(str(SRC / f), cv2.IMREAD_GRAYSCALE); H, W = g.shape
    dets = []  # (model, box in full-image px, conf, cls)
    for frac in (1 / 3, 1 / 2):
        s = int(H * frac); step = s // 2
        for y0 in list(range(0, max(1, H - s), step)) + [H - s]:
            for x0 in list(range(0, max(1, W - s), step)) + [W - s]:
                tile = cv2.cvtColor(cv2.resize(g[y0:y0 + s, x0:x0 + s], (640, 640), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
                for k, m in enumerate(models):
                    r = m.predict(tile, imgsz=640, conf=0.25, verbose=False)[0]
                    for b, c, cl in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy()):
                        bb = b * s / 640 + np.array([x0, y0, x0, y0])
                        edge = min(b[0], b[1], 640 - b[2], 640 - b[3]) < 8  # cut by the tile border -> unreliable
                        if not edge: dets.append((k, bb, float(c), int(cl)))
    dets.sort(key=lambda d: -d[2]); clusters = []
    for d in dets:
        for cl in clusters:
            if iou(cl[0][1], d[1])[0] > 0.5: cl.append(d); break
        else: clusters.append([d])
    cand = [cl for cl in clusters if len({d[0] for d in cl}) >= 3]
    cand.sort(key=lambda cl: (-len({d[0] for d in cl}), -np.mean([d[2] for d in cl])))
    keep = []
    for cl in cand:
        w = np.array([d[2] for d in cl]); b = (np.stack([d[1] for d in cl]) * w[:, None]).sum(0) / w.sum()
        if all(iou(b, k[0])[0] <= 0.3 and iou(b, k[0])[1] <= 0.7 for k in keep):
            keep.append((b, len({d[0] for d in cl}), float(w.mean()), np.bincount([d[3] for d in cl], minlength=3)))
    name = f"upano-{iid:02d}.png"; cv2.imwrite(str(OUT / "images" / name), g)
    coco["images"].append(dict(id=iid, file_name=name, width=W, height=H, source_file=f))
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); lines = []
    for j, (b, n, conf, vote) in enumerate(keep):
        sb, ok = snap(g, b)
        lines.append(f"0 {(sb[0] + sb[2]) / 2 / W:.6f} {(sb[1] + sb[3]) / 2 / H:.6f} {(sb[2] - sb[0]) / W:.6f} {(sb[3] - sb[1]) / H:.6f}\n")
        coco["annotations"].append(dict(id=len(coco["annotations"]), image_id=iid, category_id=0, iscrowd=0, bbox=[sb[0], sb[1], sb[2] - sb[0], sb[3] - sb[1]],
                                        area=(sb[2] - sb[0]) * (sb[3] - sb[1]), models=n, conf=round(conf, 3), vote=vote.tolist(), snapped=ok))
        cv2.rectangle(v, (int(sb[0]), int(sb[1])), (int(sb[2]), int(sb[3])), (0, 255, 0), 2)
        cv2.putText(v, str(j), (int(sb[0]), max(15, int(sb[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    (OUT / "labels" / (Path(name).stem + ".txt")).write_text("".join(lines))
    cv2.imwrite(str(OUT / "preview" / name.replace(".png", ".jpg")), v)
    print(name, f, W, "x", H, "boxes", len(keep), [(j, n, round(c, 2), vt.tolist()) for j, (_, n, c, vt) in enumerate(keep)])
json.dump(coco, open(OUT / "_annotations.coco.json", "w"), indent=1)
