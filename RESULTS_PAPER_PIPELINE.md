# The Tiryaki et al. pipeline on our dataset

Reference: Tiryaki B, Ozdogan A, Guller MT, Miloglu O, Oral EA, Ozbek IY. *Dental implant brand and angle identification using deep neural networks.*
J Prosthet Dent 2023. doi:10.1016/j.prosdent.2023.07.022. 2,634 panoramic radiographs, 11,904 implant crops, 5 brands (incl. Bego);
5 transfer-learning CNNs + majority-vote fusion: 98.3 % (VGG-19) and 98.9 % (fusion) accuracy with 5-fold CV; implant angle by ResNet-101 regression, RMSE 2.91°.

## What was reproduced
| step | paper | here |
|---|---|---|
| crops | labelled boxes cropped, resized to 224 × 224 | our boxes on the **original** Roboflow images, no margin, stretched to 224 × 224 (`prep_paper.py`) |
| models | VGG-16, VGG-19, ResNet-50, ResNet-101, GoogLeNet, ImageNet weights, last layer replaced | same (torchvision) |
| training | SGD, minibatches, 50 epochs; no augmentation or validation selection reported | SGD lr 1e-3, momentum 0.9, wd 1e-4, batch 32, 50 epochs, no augmentation, last-epoch model |
| fusion | majority vote of the 5 models (≥ 3) | same (ties: largest summed probability) |
| evaluation | 5-fold CV, split not described as patient-grouped | **(a)** our patient-grouped folds (leakage-free, same test crops as all other results) **(b)** random stratified 5-fold over all crops, like the paper |
| angle | hand-measured tilt (protractor), maxillary flipped to mandibular, ResNet-101 regression, 10-fold CV | **automatic** tilt labels (principal axis of the implant metal), crops with a clearly elongated metal outline only (2,582 of 3,276); ResNet-101 predicting (cos 2θ, sin 2θ) because our periapicals have every orientation; error modulo 180°; 5-fold CV, both splits |

## Brand classification (mean ± SD over 5 folds)

| model | grouped: accuracy | grouped: macro-F1 | grouped: Bego sens. / spec. | random: accuracy | random: macro-F1 |
|---|---|---|---|---|---|
| VGG-16 | 0.963 ± 0.006 | **0.927 ± 0.034** | 0.812 / 0.996 | 0.976 | 0.953 |
| VGG-19 | 0.963 ± 0.016 | 0.912 ± 0.042 | 0.771 / 0.993 | 0.979 | 0.958 |
| ResNet-50 | 0.945 ± 0.012 | 0.878 ± 0.033 | 0.717 / 0.989 | 0.964 | 0.922 |
| ResNet-101 | 0.943 ± 0.014 | 0.889 ± 0.022 | 0.798 / 0.988 | 0.963 | 0.925 |
| GoogLeNet | 0.964 ± 0.007 | 0.908 ± 0.016 | 0.735 / 0.994 | 0.979 | 0.951 |
| **Fusion (majority vote)** | **0.966 ± 0.011** | 0.916 ± 0.025 | 0.760 / 0.995 | **0.984 ± 0.005** | **0.960 ± 0.021** |
| *our recipe, ViT-B/16 on original crops (3 seeds, `RESULTS_ORIGINAL.md`)* | *0.986* | *0.954* | *0.915 (recall)* | – | – |
| *our recipe, DenseNet121 on original crops (3 seeds)* | *0.976* | *0.936* | *0.935 (recall)* | – | – |

Fusion confusion matrices (pooled over folds):

| grouped split | pred Bego | pred Bicon | pred ITI | | random split | pred Bego | pred Bicon | pred ITI |
|---|---|---|---|---|---|---|---|---|
| true Bego (76) | 57 | 0 | 19 | | true Bego (171) | 151 | 0 | 20 |
| true Bicon (934) | 0 | 928 | 6 | | true Bicon (1,879) | 1 | 1,874 | 4 |
| true ITI (580) | 8 | 20 | 552 | | true ITI (1,226) | 10 | 17 | 1,199 |

### Reading
1. **The random split inflates every model.** With the paper's kind of split (random over crops, so crops of the same radiograph and its Roboflow
   copies land in both training and test) the fusion reaches **98.4 % accuracy / 0.960 macro-F1** – close to the paper's 98.9 %. With patient-grouped folds the
   same pipeline gives **96.6 % / 0.916**: −1.8 points accuracy, −0.044 macro-F1, and Bego sensitivity drops from 0.88 to 0.76. The paper's numbers are
   therefore likely optimistic; its split is not described as patient-level.
2. **Under a fair split our recipe is clearly better.** ViT-B/16 with our training (augmentation, class weights, label smoothing, validation-based
   epoch selection, letterboxed crops with context) reaches 0.954 macro-F1 and Bego recall 0.915, against 0.927 / 0.81 for the paper's best single model
   and 0.916 / 0.76 for its fusion.
3. **Majority-vote fusion does not help on grouped folds** (0.916 vs 0.927 for VGG-16 alone); it helps a little on the random split (0.960 vs 0.958).
4. **Bego is again the weak class** (as in the paper: 95.7 %, their lowest), and its errors go to ITI, never to Bicon.

## Implant angle (ResNet-101 regression)

| split | crops | RMSE | MAE | median abs. error | within 5° | within 10° | > 20° | baseline RMSE (always predict the mean angle) |
|---|---|---|---|---|---|---|---|---|
| grouped | 1,253 test | 7.9° | 4.4° | 3.1° | 71 % | 93 % | 1.0 % | 60.2° |
| random | 2,582 test | 8.1° | 4.5° | 3.1° | 69 % | 94 % | 1.3 % | 55.9° |

* The model estimates the tilt to a median of ~3° and within 10° for 93 % of implants. The RMSE (7.9°) is higher than the paper's 2.91° mainly because
  of a few large errors (1 % > 20°); these are probably wrong automatic labels (crowns or neighbouring metal included in the outline, short wide implants).
* Not directly comparable with the paper: our labels are automatic, not hand-measured; our periapicals come in every orientation (the paper's panoramic
  implants are all near-vertical after flipping maxillary to mandibular), so our task covers −90° to +90° instead of a few tens of degrees.
* Grouped and random splits give the same angle error – unlike brand, tilt does not depend on recognising the same radiograph.
* A paper-grade angle result would need hand-checked angle labels (e.g. a clinician correcting the automatic axis on `angle_check.jpg`-style sheets).

Files: `dental_cls/prep_paper.py`, `train_paper.py`, `analyze_paper.py`; results `results_paper_pipeline/` (60 jobs), report `report_paper_pipeline/`
(`summary.csv`, `per_fold.csv`, `angle.csv`); crops and angle labels `paper_pipeline/` (`paper.npy`, `angles.csv`, `angle_check.jpg`) on Hugging Face.
