# Final summary – dental implant brand classification benchmark

Details: `RESULTS_ORIGINAL.md` (original Roboflow images – the raw baseline), `RESULTS_PAPER_PIPELINE.md` (Tiryaki et al. 2023 pipeline reproduced; random vs grouped split), `RESULTS_EXTRA_BEGO.md` and `RESULTS_EXTERNAL.md` (extra / external training data), `RESULTS.md` (round 1), `RESULTS_ROUND2.md` (ablation, resolution), `RESULTS_AUC.md` (per-brand AUC), `RESULTS_SEEDS.md` (3 training seeds – read this before quoting single-seed numbers),
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
| DenseNet121, raw crops 384 px (best single run; 3-seed mean 0.936, see below) | 0.982 | 0.955 ± 0.024 | 0.950 | 0.998 | 0.951 |
| **ViT-B/16, original images 224 px (3-seed mean)** | **0.986** | **0.954** | 0.915 | – | – |
| DenseNet121, original images 224 px (3-seed mean) | 0.976 | 0.936 | 0.935 | – | – |
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
ViT-B/16 raw 0.951, pipeline 0.936, HAT 0.951; DenseNet121 raw 384 px **0.936** (0.955 / 0.929 / 0.925) – the 384 px row above is its most favourable run,
and 384 px is not better than 224 px. Quote the 3-seed means in the paper: best single models are ViT-B/16 raw 0.951 and DenseNet121 raw 0.944 at 224 px.
"Raw" in the single-seed rows is the HAT image downsampled to 640 px; the "original images" rows use the original Roboflow export (`RESULTS_ORIGINAL.md`),
which agrees with that proxy within noise. Classifier accuracy / macro-F1 / Bego recall are fold means (scratch CNN accuracy and Bego recall: pooled); AUCs are pooled out-of-fold.
Classifiers are given ground-truth boxes; detectors and two-stage systems must find the implant first, and a missed implant counts as an error.
The two-stage rows use DenseNet121 raw 384 px retrained per fold (0.950 ± 0.027 on ground-truth boxes, against 0.955 ± 0.024 in the first run).

Detection quality: mAP50 0.887 ± 0.033 (YOLO11n) and 0.900 ± 0.020 (RF-DETR Small); mAP50-95 0.451 ± 0.042 and 0.479 ± 0.035.

## What the paper can claim
1. **A fine-tuned pretrained classifier on plain crops reaches macro-F1 ≈ 0.94–0.95 / accuracy ≈ 0.98 on the 3-class task** under leakage-free CV
   (3-seed means: DenseNet121 0.944 / 0.981, ViT-B/16 0.951 / 0.984; single runs range 0.925–0.958). Do not quote 0.955 as the result – it is the best of 5 runs at 384 px.
   Pretrained models are all within 0.92–0.955 macro-F1; the scratch CNN is clearly worse (0.78–0.82).
2. **Hand-crafted enhancement does not help.** The sharpen / CLAHE / unsharp pipeline is never better than raw crops and gives
   lower macro-F1 in every one of 3 training seeds for both DenseNet121 and ViT-B/16. Seed-averaged, the loss is significant for DenseNet121
   (−0.021, all 5 folds, ensemble McNemar Holm p = 0.002) but **not** for ViT-B/16 (−0.014, p = 0.24; the single-seed Holm p = 0.044 did not hold up).
   On the **original images** it is the other way round: ViT-B/16 −0.031 (all 3 seeds, all 5 folds, Holm p = 0.001), DenseNet121 −0.008 (not significant).
   Overall the pipeline is below raw in 10 of 12 model × raw-source × seed runs and never better on average; the size of the loss depends on the model. Ablation points to the strong
   sharpening step; a gentler pipeline only gets back to the raw level. Threshold-free metrics agree that there is no benefit
   (Bego PR-AUC lower for all five pretrained models), but the AUC loss is small and mostly not significant – word this as "no benefit, lower macro-F1".
3. **HAT super-resolution gives no consistent benefit** at 224 or 384 px, also when compared directly with the original images it was made from
   (DenseNet121 +0.009, ViT-B/16 −0.004 macro-F1, 3 seeds, not significant). One exception to report honestly: EfficientNet-B0 at 224 px
   has higher Bego PR-AUC with HAT (+0.074, p < 0.001), not reproduced at 384 px or in any other model.
4. **Higher resolution (384 px) gives no gain.** Raw 384 px crops gave the best macro-F1, Bego recall (0.950) and AUCs in the first run, but seeds 1 and 2
   came out at 0.929 and 0.925: 3-seed mean 0.936 vs 0.944 at 224 px (Δ −0.008, paired t p = 0.14, 384 px better in 1 of 5 folds), same seed-averaged Bego recall (0.909 vs 0.908).
   The round-2 AUC numbers for 384 px (macro ROC-AUC 0.998, Bego PR-AUC 0.951) are from that first run only and should be quoted with this caveat.
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

10. **Published high accuracies can come from the split.** Reproducing Tiryaki et al. (J Prosthet Dent 2023; 5 CNNs + majority vote, 98.9 %) on our data:
   98.4 % accuracy / 0.960 macro-F1 with a random crop-level split like theirs, but 96.6 % / 0.916 (Bego sensitivity 0.88 → 0.76) with patient-grouped folds.
   Under the grouped split our recipe is clearly better (ViT-B/16 0.954 macro-F1, Bego recall 0.915 vs 0.927 / 0.81 for their best model).
   Their implant-angle regression also transfers: median error 3.1°, 93 % within 10° (automatic angle labels; RMSE 7.9° vs their 2.91° with hand labels).
11. **More data from other sources did not help.** Auto-labelled Bego images (internet periapicals, panoramics) and pseudo-labelled candidates from three public
   datasets (OII-DS, Zenodo, Mendeley) never improved the best model; external candidates made ViT-B/16 significantly worse (ensemble McNemar p = 0.039).
   Panoramic / CT implant crops are mostly out of distribution for periapical models.

## Limits to state in the paper
* One repetition of the 5-fold split; 3 training seeds only for DenseNet121 (raw, pipeline, HAT, raw 384 px) and ViT-B/16 (raw, pipeline, HAT), one seed for everything else.
  Training noise: SD ≈ 0.005–0.017 on a 5-fold mean macro-F1, up to 0.10 on Bego recall. Training is not deterministic: repeating a run with the *same* seed moved the
  5-fold mean by 0.006–0.027 (`RESULTS_SEEDS.md`, "Run-to-run variation"), so single-run differences below ≈ 0.02 macro-F1 mean nothing. Fold-level tests have little power (minimum Wilcoxon p = 0.0625).
* Only 76 Bego test implants: Bego recall and PR-AUC have wide CIs (≈ ±0.06), and a handful of crops move them by several points.
* Round 1 and 2 used the HAT images area-downsampled to 640 px as "raw"; the original export (Roboflow `dental-implant-nyw2h/-dental-implant` v1, CC BY 4.0)
  was added later for DenseNet121 and ViT-B/16 and gives the same conclusions (`RESULTS_ORIGINAL.md`). The other four models were not re-run on it.
* The two-stage classifier was trained on ground-truth crops only, it cannot reject false detections, and the detector thresholds (0.3 / 0.5) were not tuned.
* 22 test crops are misclassified by most of the 36 classifier configurations (two by all 36: ITI-178, Bicon-285); these look like
  label or image problems and should be checked by a clinician (`backup/report3/consistent_errors.csv`). Crop idx 2882 is a box with no implant in it.
* Single data source (Roboflow dataset); no external test set.

## Still open (needs a GPU)
1. Done (2026-10-12): DenseNet121 raw 384 px seeds 1 and 2, plus 3 same-seed control re-runs – see `RESULTS_SEEDS.md`.
1b. Optional: train the two-stage classifier on jittered / detector boxes and add a "no implant" class; tune the detector threshold on validation folds.
1c. Done (2026-10-25): DenseNet121 and ViT-B/16 on the original Roboflow images, plain and with the pipeline, 3 seeds – `RESULTS_ORIGINAL.md`.
2. Optional: seeds for the other four models and for the round-2 ablation (all single-run, so their differences below ≈ 0.02 are untested), AUCs averaged over seeds,
   repeated CV with new fold splits, the other four models and the detectors on the original images, YOLO11-OBB/seg at 1024 px.
