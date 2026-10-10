# Previous runs (backup_all.zip) vs new leakage-free CV

## Previous runs – original Roboflow split, synthetic class-balanced training set
| Run | Task | Split evaluated | Result |
|---|---|---|---|
| VGG16 | crop classification | valid (321 crops, 12 Bego) | acc 0.866, Bego F1 0.48 (7/12 Bego correct) |
| VGG16 | crop classification | test (153 crops, **4 Bego**) | acc 0.935, macro-F1 0.907, Bego 3/4 |
| RF-DETR small | detection + brand | valid (204 images) | mAP50 0.977, mAP50-95 0.922 |
| RF-DETR small | detection + brand | test (111 images, **4 Bego boxes**) | mAP50 0.999, mAP50-95 0.942, Bego AP 1.0 |
| YOLO11n | detection + brand | valid | mAP50 0.970, mAP50-95 0.905 |
| YOLO12n | detection + brand | valid | mAP50 0.954, mAP50-95 0.869 |
| YOLO11m (stopped at 10 ep) | detection + brand | valid | mAP50 0.858, mAP50-95 0.685 |
| Faster R-CNN | detection + brand | test | mAP50 0.962, mAP 0.837 |

## New – 5-fold group-stratified CV, 1,590 test crops incl. ~76 Bego
| Model (best input) | acc | macro-F1 | Bego recall |
|---|---|---|---|
| VGG16 (raw) | 0.976 | 0.937 | 0.885 |
| DenseNet121 (raw) | 0.983 | 0.950 | 0.914 |
| ViT-B/16 (hat) | 0.984 | 0.954 | 0.914 |

## Why the numbers are not directly comparable
* **Classification improved, not worsened.** The same VGG16 goes from 0.935 test accuracy / 0.907 macro-F1 (previous) to 0.976 / 0.937 (new).
* **RF-DETR mAP is a detection metric** (box ranking, IoU), not classification accuracy/F1; 0.999 mAP50 cannot be compared with 0.95 macro-F1.
* **The previous test set is tiny for Bego:** 4 boxes. 4/4 correct has a 95% Clopper–Pearson interval of 0.40–1.00, so Bego AP = 1.0 carries almost no evidence. The new CV tests ~76 Bego crops.
* **Single split vs 5 folds:** one lucky/easy split can look better than the average of 5; the new numbers are averages with SDs.
* **Leakage in the previous split:** Bicon-289 (train) and Bicon-291 (test) are near-duplicate radiographs.
* RF-DETR reports recall 0.91 for every class on valid and 1.00 for every class on test – identical per-class values suggest these are recall at a fixed operating point, not per-class recalls; use mAP.

## To compare like with like
Retrain RF-DETR / YOLO11n on the same 5 group-stratified folds (detection version of `folds.json`) and report mAP50, mAP50-95 and per-class AP as mean ± SD.
