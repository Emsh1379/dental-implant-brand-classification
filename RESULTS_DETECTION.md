# Detection – YOLO11n and RF-DETR Small on the same 5 folds

Same group-stratified folds as the classifiers; every test image is used once (1,590 ground-truth implants in total, the same
implants as the 1,590 classifier test crops). 30 epochs, 640 px, both detectors scored with the same COCO evaluator.
Brand accuracy of a detector: each ground-truth implant takes the class of the highest-scoring prediction with IoU ≥ 0.5;
an implant with no such prediction counts as an error ("missed"). Mean ± SD over 5 folds.

| metric | YOLO11n | RF-DETR Small | YOLO11n, GT-ROI preprocessed images |
|---|---|---|---|
| mAP50 | 0.887 ± 0.033 | **0.900 ± 0.020** | 0.863 ± 0.026 |
| mAP50-95 | 0.451 ± 0.042 | **0.479 ± 0.035** | 0.799 ± 0.029 (leak) |
| mAP75 | 0.414 ± 0.068 | 0.470 ± 0.069 | 0.837 ± 0.028 (leak) |
| AP50-95 Bego / Bicon / ITI | 0.377 / 0.459 / 0.518 | 0.445 / 0.455 / 0.537 | 0.678 / 0.859 / 0.860 |
| AP50 Bego / Bicon / ITI | 0.831 / 0.911 / 0.919 | 0.824 / 0.919 / 0.957 | 0.688 / 0.964 / 0.936 |
| brand accuracy | 0.963 ± 0.009 | **0.968 ± 0.014** | 0.962 ± 0.004 |
| brand macro-F1 | 0.905 ± 0.040 | **0.917 ± 0.026** | 0.895 ± 0.017 |
| brand recall Bego / Bicon / ITI | 0.690 / 0.987 / 0.959 | 0.749 / 0.982 / 0.976 | 0.644 / 0.985 / 0.967 |
| implants missed (of 1,590) | 11 | 9 | 5 |

Per-fold values: `backup/det_report/per_fold.csv`; paired tests: `backup/det_report/comparisons.csv` (`dental_cls/analyze_det.py`).

## Findings
1. **RF-DETR Small is slightly ahead of YOLO11n on every summary metric, but not significantly.** mAP50-95 +0.028 (better in 4 of 5
   folds, paired t p = 0.15), mAP50 +0.013 (p = 0.49), brand accuracy +0.005 (p = 0.38). With the 5th fold included the RF-DETR
   means moved down a little from the 4-fold values reported earlier (0.901 / 0.482 / 0.973 / 0.920 → 0.900 / 0.479 / 0.968 / 0.917).
2. **Both detectors localise loosely.** mAP50 is ≈ 0.89–0.90 but mAP50-95 is only 0.45–0.48: implants are found, box edges are imprecise.
3. **Bego is the weak class for detectors, more than for classifiers.** Detector Bego recall is 0.69 (YOLO11n) and 0.75 (RF-DETR),
   against 0.91–0.95 for DenseNet121 on ground-truth crops. Bego recall varies strongly between folds (YOLO11n 0.40–0.89, RF-DETR 0.71–0.80; few Bego implants per fold),
   so the detector-vs-detector Bego difference is not reliable (p = 0.50).
4. **Detectors are below the best crop classifier on brand.** Brand macro-F1 0.905 / 0.917 vs 0.955 (DenseNet121, raw 384 px);
   brand accuracy 0.963 / 0.968 vs 0.982. The classifier gets ground-truth boxes, so this is an upper bound for a two-stage system,
   not a like-for-like comparison – the two-stage experiment (detector boxes → classifier) is still open.
5. **GT-ROI preprocessing leak, now with all folds.** Running the ROI preprocessing with ground-truth boxes before detection raises
   YOLO11n mAP50-95 from 0.451 to 0.799 (+0.348, better in all 5 folds, paired t p = 0.0003) while mAP50 (−0.025), brand accuracy
   (−0.001), brand macro-F1 (−0.011) and Bego recall (−0.046) do not improve. The gain is box-edge precision copied from the
   annotation, not better recognition; it explains the earlier ≈ 0.90 mAP50-95.

## Caveats
* 5 folds: paired tests have little power (minimum possible Wilcoxon p = 0.0625); read the deltas with their SDs.
* One seed per fold, small models only (YOLO11n, RF-DETR Small), 30 epochs, 640 px.
* Brand accuracy of a detector depends on the IoU ≥ 0.5 matching rule; a missed implant is counted as a brand error.
