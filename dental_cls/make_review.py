"""Clinician review pack: the 22 crops most models get wrong + the empty box (idx 2882).
One PDF page per case: whole original radiograph with the implant marked (red = this case, yellow = other implants in the image), the crop the
models see, the dataset label and the usual model prediction. Plus review_form.csv to fill in.
usage: python make_review.py <orig_v1_yolo dir> <out dir>"""
import glob, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

V1, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(exist_ok=True)
NAMES = ["Bego", "Bicon", "ITI"]
X = np.load("hf_crops/crops/raw_orig.npy", mmap_mode="r"); meta = pd.read_csv("ref/crops/meta.csv")
match = pd.read_csv("hf_crops/crops/raw_orig_match.csv").set_index("idx").original
paths = {Path(p).name: p for p in glob.glob(f"{V1}/*/images/*")}
E = pd.read_csv("consistent_errors.csv")
import json; LABELS = json.load(open("review_labels.json"))  # dataset label files of these images (Mesh2001/Dental_implant zip)
cases = [dict(idx=r.idx, label=r.label, pred=r.usual_prediction, wrong=f"{r.n_wrong} of {r.n_configs} model configurations wrong",
              p=f"mean probability: dataset label {r.mean_p_label:.2f}, usual prediction {r.mean_p_prediction:.2f}") for r in E.itertuples()]
m = meta.set_index("idx").loc[2882]
cases.append(dict(idx=2882, label=NAMES[int(m.brand)], pred="-", wrong="suspected EMPTY BOX (no implant inside the annotated box)", p=""))


def boxes_of(p):
    out = []
    for ln in open(p.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\").rsplit(".", 1)[0] + ".txt").read().split("\n"):
        v = ln.split()
        if len(v) >= 5:
            c, xy = int(v[0]), list(map(float, v[1:]))
            if len(xy) == 4: cx, cy, w, h = xy
            else: xs, ys = xy[0::2], xy[1::2]; cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
            out.append((cx, cy, w, h))
    return out


def z(a): a = a.astype(np.float32).ravel(); a -= a.mean(); return a / (np.linalg.norm(a) + 1e-6)


rows = []
with PdfPages(OUT / "review_cases.pdf") as pdf:
    for n, c in enumerate(cases, 1):
        idx = int(c["idx"]); r = meta.set_index("idx").loc[idx]; p = paths[match[idx]]
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); H, W = g.shape; crop = np.asarray(X[idx])
        # boxes from the dataset's own label file (the ones the crops were cut from); this case = box number box_idx
        bs = []
        for ln in LABELS[Path(r.image).stem].split("\n"):
            v = ln.split()
            if len(v) >= 5:
                xy = list(map(float, v[1:]))
                if len(xy) == 4: bs.append(tuple(xy))
                else: xs, ys = xy[0::2], xy[1::2]; bs.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)))
        best = bs[int(r.box_idx)]
        v = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        for b in bs:
            cx, cy, w, h = b; col = (255, 0, 0) if b == best else (255, 210, 0)
            cv2.rectangle(v, (int((cx - w / 2) * W), int((cy - h / 2) * H)), (int((cx + w / 2) * W), int((cy + h / 2) * H)), col, 3 if b == best else 2)
        fig = plt.figure(figsize=(11.7, 6.2))
        ax1 = fig.add_axes([0.02, 0.05, 0.55, 0.78]); ax1.imshow(v); ax1.axis("off"); ax1.set_title("whole radiograph (red = this case)", fontsize=10)
        ax2 = fig.add_axes([0.60, 0.20, 0.38, 0.63]); ax2.imshow(crop, cmap="gray", vmin=0, vmax=255); ax2.axis("off"); ax2.set_title("crop the models see", fontsize=10)
        fig.text(0.02, 0.93, f"Case {n}   (crop idx {idx}, source image {r.group})", fontsize=14, weight="bold")
        fig.text(0.02, 0.875, f"Dataset label: {c['label']}    |    Models usually predict: {c['pred']}    |    {c['wrong']}", fontsize=11, color="darkred")
        fig.text(0.60, 0.12, c["p"], fontsize=9); fig.text(0.60, 0.07, f"file: {r.image[:60]}", fontsize=7, color="gray")
        pdf.savefig(fig); fig.savefig(OUT / f"case_{n:02d}.png", dpi=110); plt.close(fig)
        rows.append(dict(case=n, crop_idx=idx, source_image=r.group, dataset_label=c["label"], model_prediction=c["pred"],
                         your_verdict="", correct_brand="", image_quality_ok="", notes=""))
pd.DataFrame(rows).to_csv(OUT / "review_form.csv", index=False)
print(len(rows), "cases ->", OUT)
