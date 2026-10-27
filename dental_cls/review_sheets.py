#!/usr/bin/env python3
"""Numbered review sheets of ALL candidates per brand (number = review_no in candidates_<brand>.csv).

usage: python review_sheets.py <match output dir> <crop dir> [<crop dir> is where the scored .jpg crops are]
"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import letterbox

O, S = Path(sys.argv[1]), Path(sys.argv[2])
for b in ("Bego", "Bicon", "ITI"):
    p = O / f"candidates_{b}.csv"
    if not p.exists(): continue
    d = pd.read_csv(p).reset_index(drop=True)
    if "review_no" not in d: d.insert(0, "review_no", d.index)
    d.to_csv(p, index=False)
    print(b, len(d), "candidates,", d.patient.nunique() if len(d) else 0, "patients/images")
    if not len(d): continue
    tiles = []
    for i, f in zip(d.review_no, d.file):
        t = cv2.resize(letterbox(cv2.imread(str(S / f), cv2.IMREAD_GRAYSCALE)), (130, 130))
        cv2.rectangle(t, (0, 0), (34, 16), 0, -1); cv2.putText(t, str(i), (2, 13), 0, 0.45, 255, 1); tiles.append(t)
    while len(tiles) % 10: tiles.append(np.zeros((130, 130), np.uint8))
    cv2.imwrite(str(O / f"review_{b}.jpg"), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
