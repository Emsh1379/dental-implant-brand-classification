# Published protocols reproduced on the same dataset – with and without leakage

Two 2025 papers use exactly our data source (Roboflow `dental-implant-nyw2h/-dental-implant`, Bicon / Bego / ITI periapical radiographs), and a third
(Tiryaki et al. 2023) uses the same three-brand idea on private panoramics. Each protocol was re-run on our data twice: as published, and with
splits grouped by source radiograph (our leakage-free folds). Same images, same models, only the split differs.

## Summary

| study | their headline | their protocol on our data | same protocol, leakage-free | drop |
|---|---|---|---|---|
| Lashaki et al., BMC Oral Health 2025 (VGG16, Bicon vs Bego) | 98.65 % acc. | 99.3 % acc. / macro-F1 0.982 / Bego recall 0.955 | 97.8 % / 0.926 / 0.838 | −1.5 points acc., −0.056 F1, −0.12 Bego recall |
| Lashaki protocol, all 3 brands (not reported by them) | – | 89.8 % / 0.874 / 0.848 | 84.1 % / 0.753 / 0.488 | **−5.7 points, −0.121 F1, −0.36 Bego recall (all 5 folds, p ≤ 0.015)** |
| Tekis et al., Sci Rep 2025 (YOLO detection) | F1 0.914, precision 0.931 | YOLO11m: F1 0.956, mAP50 0.966 | F1 0.857, mAP50 0.872 | **−0.10 F1, −0.09 mAP50; Bego F1 0.95 → 0.76** |
| Tiryaki et al., J Prosthet Dent 2023 (5 CNNs + vote) | 98.9 % acc. | 98.4 % / 0.960 | 96.6 % / 0.916 | −1.8 points, −0.044 F1, Bego sens. 0.88 → 0.76 |
| **Ours** (ViT-B/16, crops, 3 seeds) | – | – | **98.6 % / 0.954 / Bego recall 0.915** | – |

All three published protocols reproduce their reported numbers on our data when their split is used, and all lose clearly – most on Bego – when
copies of the same radiograph are kept out of the test set. Under the leakage-free split our pipeline is better than every reproduced protocol.

## 1. Lashaki et al. (BMC Oral Health 2025;25:535)
Their protocol (`lashaki_protocol.py`): whole image → 128 px → sharpening → median 3×3 → CLAHE (2.0, 3×3) → implant mask → histogram equalisation;
10 versions per image (rotation ±20°, shift/shear 0.2, zoom 0.8–1.2, flip) **created before the 5-fold split**; VGG16 (ImageNet, last block trainable,
GAP, Dense 256), Adam 1e-4, early stopping (patience 15) on a 10 % validation split. Batch 32 instead of 8. Their mask needs the annotated boxes; here
the boxes of the dataset were used (unavailable at test time in practice).

Protocols: **leaky** = theirs (random split over the 10 versions of all 2,064 images); **grouped** = same preprocessing + mask, source-grouped folds,
versions of training images only, test on the original primary image; **grouped_nomask** = preprocessing without the mask; **grouped_raw** = resize only.

| task | protocol | test images per fold | accuracy | macro-F1 | ROC-AUC | Bego recall |
|---|---|---|---|---|---|---|
| Bicon vs Bego | leaky (theirs) | 2,546 | 0.993 ± 0.001 | 0.982 ± 0.003 | 0.9995 | 0.955 |
| Bicon vs Bego | grouped | 125 | 0.978 ± 0.013 | 0.926 ± 0.062 | 0.986 | 0.838 |
| Bicon vs Bego | grouped, no mask | 125 | 0.987 ± 0.004 | 0.963 ± 0.013 | 0.991 | 0.904 |
| Bicon vs Bego | grouped, raw (no preprocessing) | 125 | **0.997 ± 0.004** | **0.992 ± 0.011** | 0.996 | **0.971** |
| 3 brands | leaky (theirs) | 4,128 | 0.898 ± 0.005 | 0.874 ± 0.012 | 0.979 | 0.848 |
| 3 brands | grouped | 202 | 0.841 ± 0.031 | 0.753 ± 0.063 | 0.937 | 0.488 |
| 3 brands | grouped, no mask | 202 | 0.779 ± 0.020 | 0.685 ± 0.063 | 0.914 | 0.510 |
| 3 brands | grouped, raw | 202 | 0.802 ± 0.031 | 0.738 ± 0.047 | 0.928 | 0.603 |

* **Leakage:** the 3-brand task loses 5.7 accuracy points, 0.121 macro-F1 and 0.36 Bego recall in all 5 folds (paired t p = 0.014 / 0.011 / 0.015).
  For their headline Bicon-vs-Bego the effect is smaller because the two brands look very different (−1.5 points, 0 of 5 folds better).
* **Their preprocessing does not help.** For Bicon vs Bego the raw images are best (raw vs preprocessed without mask: better in 5 of 5 folds,
  accuracy p = 0.004, macro-F1 p = 0.011, Bego recall p = 0.018); raw vs their full pipeline: accuracy +1.9 points (p = 0.043).
  Their claimed gain from preprocessing is not reproduced once the split is fair.
* **The mask helps only in the 3-brand task (+0.068 macro-F1 vs no mask, p = 0.027)** – because it is made from the ground-truth boxes, i.e. it tells
  the model where the implant is. The legitimate equivalent is cropping with a detector, which is what our pipeline does (macro-F1 0.954).
* Whole images at 128 px with a mostly frozen VGG16 are a weak design for three brands (macro-F1 ≤ 0.75 leakage-free).

## 2. Tekis et al. (Sci Rep 2025;15:33304)
`tekis_yolo.py`: the Roboflow export `lamiaa/implant-recognition` v3 (2,778 annotated images from 515 source radiographs, each 3–9 augmented copies;
classes Bego / Bicon / ITI), YOLO11m (their best family: v8–v11 m/l), 100 epochs, 640 px.
* **Roboflow's split leaks:** 233 of the 515 radiographs have copies in more than one split; **86 of the 116 test images** are copies of radiographs that are also in training.
* Grouped split: same images, whole radiographs assigned to train / valid / test with the same image counts (no radiograph in two splits).

| split | test images | precision | recall | F1 | mAP50 | mAP50-95 | F1 Bego / Bicon / ITI |
|---|---|---|---|---|---|---|---|
| Roboflow split (as in the paper) | 116 | 0.944 | 0.968 | **0.956** | 0.966 | 0.503 | 0.95 / 0.94 / 0.98 |
| grouped by radiograph | 125 | 0.814 | 0.910 | **0.857** | 0.872 | 0.438 | 0.76 / 0.85 / 0.96 |

* Leakage inflates F1 by ~0.10 and mAP50 by ~0.09; every brand is lower without it, Bego most (0.95 → 0.76).
* The grouped result agrees with our own 5-fold leakage-free detector results (YOLO11n mAP50 0.887, RF-DETR 0.900; `RESULTS_DETECTION.md`).
* Limits: one split per protocol (YOLO11m takes 2 h per run) and the two test sets contain different images. Within the Roboflow test set, the 30 images
  whose radiograph is not in training scored F1 0.968 vs 0.936 for the 86 seen ones – not interpretable: 30 images, and the brand mix differs (ITI only in the unseen part).

## 3. Tiryaki et al. (J Prosthet Dent 2023)
See `RESULTS_PAPER_PIPELINE.md`: 5-model majority vote 98.4 % (random crop-level split) vs 96.6 % (grouped); Bego sensitivity 0.88 → 0.76.

## What this means for the paper
* The same public dataset gives "98–99 %" under the published protocols and 84–97 % (depending on task and method) once a radiograph cannot appear in
  both training and test. The inflation comes from (a) augmenting before splitting, (b) Roboflow's pre-augmented export with its random split,
  (c) random crop-level splits; and it is largest for the rare brand, Bego.
* Preprocessing (sharpening / CLAHE / histogram equalisation) does not help under a fair split – consistent with our own ablations – and a ground-truth mask
  is a form of label leakage at test time.
* Files on Hugging Face: `protocols/lashaki/` (40 result JSONs), `protocols/tekis/` (results, image table with splits), `report_protocols/`.
  Scripts: `lashaki_protocol.py`, `tekis_yolo.py`, `analyze_lashaki.py`, `train_paper.py` (Tiryaki).
