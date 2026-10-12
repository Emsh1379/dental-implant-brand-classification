# Final summary – dental implant brand classification benchmark

Details: `RESULTS.md` (round 1), `RESULTS_ROUND2.md` (ablation, resolution), `RESULTS_AUC.md` (per-brand AUC), `RESULTS_SEEDS.md` (3 training seeds – read this before quoting single-seed numbers),
`RESULTS_DETECTION.md` (detectors), `RESULTS_TWO_STAGE.md` (detector → classifier, box jitter), `COMPARISON_WITH_PAPER.md`, `PREVIOUS_vs_NEW.md`.

## Setup
* 3 brands: Bego 171 / Bicon 1,879 / ITI 1,226 implant crops from 1,009 source-image groups (near-duplicates merged).
* 5-fold group-stratified CV, one fixed split (`backup/crops/folds.json`). Augmented copies are used for training only.
  Every one of the 1,590 test implants (76 Bego) is tested exactly once; detectors and classifiers use the same folds and the same implants.
* 180 classifier jobs (6 models × 3 inputs in round 1; ablations, revised pipelines, 3-channel and 384 px in round 2) and 15 detector jobs.
* Primary metrics: macro-F1 and balanced accuracy (Bego is rare). Tests: McNemar on pooled out-of-fold predictions with Holm correction, paired t / Wilcoxon on folds, bootstrap CIs.

## Headline results

| system | accuracy | macro-F1 | Bego recall | macro ROC-AUC | Bego PR-AUC |
|---|---|---|---|---|---|
| **DenseNet121, raw crops 384 px** (best) | 0.982 | **0.955 ± 0.024** | **0.950** | 0.998 | 0.951 |
| DenseNet121, raw 224 px | 0.983 | 0.950 ± 0.030 | 0.914 | 0.992 | 0.907 |
| ViT-B/16, raw 224 px | 0.984 | 0.950 ± 0.028 | 0.869 | 0.988 | 0.904 |
| DenseNet121, original preprocessing pipeline | 0.968 | 0.917 ± 0.026 | 0.876 | 0.995 | 0.888 |
| DenseNet121, HAT super-resolution 224 px | 0.978 | 0.945 ± 0.039 | 0.916 | 0.995 | 0.908 |
| custom CNN from scratch, raw | 0.922 | 0.822 ± 0.025 | 0.789 | 0.975 | 0.740 |
| **RF-DETR Small → DenseNet121 (two-stage)** | 0.974 | 0.948 ± 0.028 | 0.960 | – | – |
| **YOLO11n → DenseNet121 (two-stage)** | 0.976 | 0.956 ± 0.035 | 0.960 | – | – |
| RF-DETR Small (detector, brand from boxes) | 0.968 | 0.917 ± 0.026 | 0.749 | – | – |
| YOLO11n (detector, brand from boxes) | 0.963 | 0.905 ± 0.040 | 0.690 | – | – |

The classifier rows are the original single seed. With 3 seeds (`RESULTS_SEEDS.md`) the macro-F1 means are: DenseNet121 raw 0.944, pipeline 0.923, HAT 0.945;
ViT-B/16 raw 0.951, pipeline 0.936, HAT 0.951. Raw 384 px has one seed plus one retraining (0.955 and 0.950).
Classifier accuracy / macro-F1 / Bego recall are fold means (scratch CNN accuracy and Bego recall: pooled); AUCs are pooled out-of-fold.
Classifiers are given ground-truth boxes; detectors and two-stage systems must find the implant first, and a missed implant counts as an error.
The two-stage rows use DenseNet121 raw 384 px retrained per fold (0.950 ± 0.027 on ground-truth boxes, against 0.955 ± 0.024 in the first run).

Detection quality: mAP50 0.887 ± 0.033 (YOLO11n) and 0.900 ± 0.020 (RF-DETR Small); mAP50-95 0.451 ± 0.042 and 0.479 ± 0.035.

## What the paper can claim
1. **A fine-tuned pretrained classifier on plain crops reaches macro-F1 0.955 / accuracy 0.982 on the 3-class task** under leakage-free CV.
   Pretrained models are all within 0.92–0.955 macro-F1; the scratch CNN is clearly worse (0.78–0.82).
2. **Hand-crafted enhancement does not help.** The sharpen / CLAHE / unsharp pipeline is never better than raw crops and gives
   lower macro-F1 in every one of 3 training seeds for both DenseNet121 and ViT-B/16. Seed-averaged, the loss is significant for DenseNet121
   (−0.021, all 5 folds, ensemble McNemar Holm p = 0.002) but **not** for ViT-B/16 (−0.014, p = 0.24; the single-seed Holm p = 0.044 did not hold up). Ablation points to the strong
   sharpening step; a gentler pipeline only gets back to the raw level. Threshold-free metrics agree that there is no benefit
   (Bego PR-AUC lower for all five pretrained models), but the AUC loss is small and mostly not significant – word this as "no benefit, lower macro-F1".
3. **HAT super-resolution gives no consistent benefit** at 224 or 384 px. One exception to report honestly: EfficientNet-B0 at 224 px
   has higher Bego PR-AUC with HAT (+0.074, p < 0.001), not reproduced at 384 px or in any other model.
4. **Higher resolution is at least as good, not yet a proven gain.** Raw 384 px crops gave the best single-seed macro-F1, Bego recall (0.950) and AUCs,
   but its margin over the 3-seed raw 224 px mean (+0.006 to +0.011) is the size of the seed noise; seeds 1 and 2 at 384 px are still to run.
5. **Preprocessing that uses the annotated implant region leaks the label location.** With GT-ROI preprocessing YOLO11n mAP50-95 goes
   from 0.451 to 0.799 (all 5 folds, p = 0.0003) with no gain in mAP50, brand accuracy or Bego recall. This explains very high mAP50-95 values from such pipelines.
6. **Detectors are slightly below the crop classifier on brand, and clearly below on Bego** (recall 0.69–0.75 vs 0.91–0.95).
   RF-DETR Small is ahead of YOLO11n on every summary metric but no difference is significant with 5 folds.
7. **A two-stage system keeps the classifier's quality without ground-truth boxes.** Detector box → DenseNet121 gives macro-F1 0.948 (RF-DETR)
   and 0.956 (YOLO11n), not significantly different from the classifier on ground-truth boxes (0.950; McNemar p = 0.15 and 0.46), and raises
   Bego recall from 0.69–0.75 to 0.96. At a fixed score threshold it is also better end-to-end (macro-F1 +0.02 to +0.03) but Bego precision drops (0.86 → 0.74 for RF-DETR).
8. **The classifier is robust to detector-sized box errors, less to size errors.** Macro-F1 stays within 0.02 down to IoU ≈ 0.75–0.8;
   tight boxes lose Bego, loose boxes push predictions towards Bego (ITI recall 0.82 at 1.5× box size).
9. **Bego is the limiting class everywhere.** Bicon and ITI ROC-AUC ≥ 0.989 for every pretrained model; the most consistent errors are ITI↔Bego confusions.

## Limits to state in the paper
* One repetition of the 5-fold split; 3 training seeds only for DenseNet121 / ViT-B/16 on raw, pipeline and HAT, one seed for everything else.
  Seed noise: ≈ 0.005–0.012 on a 5-fold mean macro-F1, up to 0.07 on Bego recall. Fold-level tests have little power (minimum Wilcoxon p = 0.0625).
* Only 76 Bego test implants: Bego recall and PR-AUC have wide CIs (≈ ±0.06), and a handful of crops move them by several points.
* "Raw" images are the HAT images area-downsampled to 640 px, a proxy for the original Roboflow export.
* The two-stage classifier was trained on ground-truth crops only, it cannot reject false detections, and the detector thresholds (0.3 / 0.5) were not tuned.
* 22 test crops are misclassified by most of the 36 classifier configurations (two by all 36: ITI-178, Bicon-285); these look like
  label or image problems and should be checked by a clinician (`backup/report3/consistent_errors.csv`).
* Single data source (Roboflow dataset); no external test set.

## Still open (needs a GPU)
1. DenseNet121 raw 384 px seeds 1 and 2 (10 jobs, ≈ 1 h on one T4) – the Kaggle session ended before they ran.
1b. Optional: train the two-stage classifier on jittered / detector boxes and add a "no implant" class; tune the detector threshold on validation folds.
2. Optional: 3 seeds / repeated CV for the top models, original Roboflow images for "raw", YOLO11-OBB/seg at 1024 px.
