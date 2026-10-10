"""Shared helpers: label parsing, source-image grouping, letterboxed crops."""
import re
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["Bego", "Bicon", "ITI"]
CROP_SIZE = 224
MARGIN = 0.15


def source_id(filename: str) -> str:
    """Roboflow names files '<source>_<ext>.rf.<hash>.<ext>'; the part before '.rf.' identifies the original image."""
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", filename)[0]


def parse_boxes(label_path: Path):
    """YOLO box or polygon lines -> [(cls, cx, cy, w, h)] normalised."""
    out = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cls, v = int(float(p[0])), list(map(float, p[1:]))
        if len(v) == 4:
            cx, cy, w, h = v
        elif len(v) >= 6:
            xs, ys = v[0::2], v[1::2]
            x0, x1, y0, y1 = max(0, min(xs)), min(1, max(xs)), max(0, min(ys)), min(1, max(ys))
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        else:
            continue
        if w > 0 and h > 0:
            out.append((cls, cx, cy, w, h))
    return out


def crop_box(img: np.ndarray, box, margin=MARGIN):
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    x1 = max(0, int(round((cx - bw / 2) * w - margin * bw * w)))
    y1 = max(0, int(round((cy - bh / 2) * h - margin * bh * h)))
    x2 = min(w - 1, int(round((cx + bw / 2) * w + margin * bw * w)))
    y2 = min(h - 1, int(round((cy + bh / 2) * h + margin * bh * h)))
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    return img[y1:y2 + 1, x1:x2 + 1]


def letterbox(img: np.ndarray, size=CROP_SIZE):
    """Pad to square (edge value = border median) then resize; keeps implant aspect ratio."""
    h, w = img.shape[:2]
    s = max(h, w)
    pad = int(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))
    canvas = np.full((s, s), pad, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    interp = cv2.INTER_AREA if s > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp)
