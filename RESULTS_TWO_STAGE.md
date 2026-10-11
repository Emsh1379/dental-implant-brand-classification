# Two-stage system (detector box → DenseNet121) and box-jitter robustness

Same 5 folds and the same 1,590 test implants (76 Bego) as everything else. Detectors: the stored RF-DETR Small and YOLO11n
fold checkpoints. Classifier: DenseNet121 on raw 384 px crops, retrained per fold with `train_cv.py --save_all` so that every
fold has a checkpoint (the first run kept only fold 0). The retrained models match the first run but are not bit-identical:
macro-F1 0.950 ± 0.027 / accuracy 0.980 / Bego recall 0.952, against 0.955 ± 0.024 / 0.982 / 0.950. All numbers below use the retrained models.
Checks: crops rebuilt from the ground-truth boxes are pixel-identical to the stored `raw384` crops in all 5 folds, and the
detectors' one-stage brand accuracy reproduces the stored detection results exactly.

## 1. Brand of each implant (GT-matched)
Each ground-truth implant takes the highest-scoring detection with IoU ≥ 0.5 (same rule as `RESULTS_DETECTION.md`).
One-stage = the detector's class. Two-stage = the detected box is cropped (15% margin, letterbox 384) and classified by DenseNet121.
An implant with no detection is an error in both. Mean ± SD over 5 folds.

| system | accuracy | macro-F1 | recall Bego / Bicon / ITI | missed |
|---|---|---|---|---|
| RF-DETR Small, one-stage | 0.968 ± 0.014 | 0.917 ± 0.026 | 0.749 / 0.982 / 0.976 | 9 |
| **RF-DETR Small → DenseNet121** | 0.974 ± 0.009 | 0.948 ± 0.028 | **0.960** / 0.981 / 0.963 | 9 |
| YOLO11n, one-stage | 0.963 ± 0.009 | 0.905 ± 0.040 | 0.690 / 0.987 / 0.959 | 11 |
| **YOLO11n → DenseNet121** | **0.976 ± 0.014** | **0.956 ± 0.035** | **0.960** / 0.983 / 0.968 | 11 |
| DenseNet121 on ground-truth boxes (upper reference) | 0.980 ± 0.010 | 0.950 ± 0.027 | 0.952 / 0.985 / 0.975 | 0 |

McNemar on the 1,590 implants (`backup/ts_report/gt_matched_mcnemar.csv`):

| comparison | only first correct | only second correct | p |
|---|---|---|---|
| RF-DETR one-stage vs two-stage | 23 | 32 | 0.28 |
| YOLO11n one-stage vs two-stage | 19 | 40 | 0.009 |
| RF-DETR two-stage vs ground-truth boxes | 11 | 20 | 0.15 |
| YOLO11n two-stage vs ground-truth boxes | 12 | 17 | 0.46 |

* **The two-stage system reaches the level of the classifier on ground-truth boxes.** Macro-F1 0.948 / 0.956 against 0.950; the
  remaining gap in accuracy (0.974–0.976 vs 0.980) is not significant and is about the size of the missed implants (9 and 11 of 1,590).
* **The gain is almost entirely Bego.** Bego recall goes from 0.75 to 0.96 (RF-DETR) and from 0.69 to 0.96 (YOLO11n). With RF-DETR,
  the detector calls 18 of the 76 Bego implants ITI and 1 Bicon; after re-classification 3 are wrong. The classifier fixes 18 (RF-DETR)
  and 20 (YOLO11n) Bego implants and breaks 2 and 0.
* **It costs a little ITI recall** (RF-DETR 0.976 → 0.963), so overall accuracy rises only by 0.006–0.013; the improvement is
  significant for YOLO11n (p = 0.009) and not for RF-DETR (p = 0.28).
* **The choice of detector does not matter for the two-stage result** (0.948 vs 0.956 macro-F1, same Bego recall).
* Detector boxes are loose: mean IoU with the ground truth is 0.76 for both detectors (RF-DETR: median 0.78, 5th percentile 0.56).

## 2. End-to-end at a score threshold
Detections above the threshold, class-agnostic NMS (IoU 0.5), matched one-to-one to implants; a detection is correct when it
matches an implant and has its brand. False and duplicate detections count against precision. Counts pooled over 5 folds.

| detector | threshold | labels | Bego precision / recall / F1 | Bicon F1 | ITI F1 | macro-F1 |
|---|---|---|---|---|---|---|
| RF-DETR Small | 0.3 | detector | 0.855 / 0.697 / 0.768 | 0.925 | 0.944 | 0.879 |
| RF-DETR Small | 0.3 | classifier | 0.742 / 0.908 / 0.817 | 0.923 | 0.948 | **0.896** |
| RF-DETR Small | 0.5 | detector | 0.906 / 0.632 / 0.744 | 0.929 | 0.947 | 0.873 |
| RF-DETR Small | 0.5 | classifier | 0.771 / 0.842 / 0.805 | 0.926 | 0.950 | 0.893 |
| YOLO11n | 0.3 | detector | 0.912 / 0.684 / 0.782 | 0.922 | 0.925 | 0.876 |
| YOLO11n | 0.3 | classifier | 0.807 / 0.934 / 0.866 | 0.924 | 0.939 | **0.910** |
| YOLO11n | 0.5 | detector | 0.930 / 0.526 / 0.672 | 0.815 | 0.886 | 0.791 |
| YOLO11n | 0.5 | classifier | 0.806 / 0.711 / 0.755 | 0.815 | 0.889 | 0.820 |

* The two-stage system is better at every threshold (macro-F1 +0.017 to +0.034), again through Bego recall.
* **Bego precision drops** (RF-DETR 0.86 → 0.74 at 0.3): the class-weighted classifier calls more boxes Bego, including some of the
  ≈ 100 false detections, which it has no way to reject. A deployed system needs a detector threshold tuned on validation data, or a "no implant" class.
* YOLO11n loses many implants at 0.5 (Bicon recall 0.73), so its threshold must be lower than RF-DETR's. The thresholds here are fixed, not tuned.
* These numbers are lower than section 1 because every extra box is an error here; section 1 only asks "given the implant, is the brand right?".

## 3. Box-jitter robustness of the classifier
Ground-truth boxes are changed before cropping; no detector involved. Mean over 5 folds (random jitter: 3 draws per fold).
`backup/ts_report/jitter_summary.csv`, figure `backup/ts_report/jitter.png`.

| change to the box | mean IoU with GT | accuracy | macro-F1 | recall Bego / ITI |
|---|---|---|---|---|
| none | 1.00 | 0.980 | 0.950 | 0.952 / 0.975 |
| scale × 0.9 (tighter) | 0.81 | 0.978 | 0.939 | 0.899 / 0.972 |
| scale × 0.8 | 0.64 | 0.969 | 0.918 | 0.852 / 0.955 |
| scale × 0.7 | 0.49 | 0.929 | 0.837 | 0.773 / 0.865 |
| scale × 1.1 (looser) | 0.83 | 0.976 | 0.943 | 0.975 / 0.961 |
| scale × 1.2 | 0.70 | 0.966 | 0.917 | 0.988 / 0.942 |
| scale × 1.35 | 0.55 | 0.943 | 0.862 | 0.989 / 0.887 |
| scale × 1.5 | 0.45 | 0.910 | 0.807 | 1.000 / 0.819 |
| centre shift 10% of box size | 0.78 | 0.977 | 0.945 | 0.947 / 0.970 |
| centre shift 20% | 0.61 | 0.972 | 0.929 | 0.931 / 0.962 |
| centre shift 30% | 0.48 | 0.965 | 0.910 | 0.924 / 0.943 |
| edge noise SD 6% of box size | 0.83 | 0.977 | 0.942 | 0.938 / 0.968 |
| edge noise SD 10% | 0.74 | 0.972 | 0.930 | 0.946 / 0.956 |
| edge noise SD 15% | 0.64 | 0.959 | 0.905 | 0.928 / 0.925 |

* **The classifier tolerates detector-sized errors.** Down to a mean IoU of about 0.75–0.8 (the detectors' level) macro-F1 stays
  within 0.02 of the clean value, which agrees with the two-stage result.
* **Size errors matter more than position errors.** A shift giving IoU 0.48 still leaves macro-F1 at 0.910; a scale change giving
  IoU 0.45–0.49 drops it to 0.81–0.84.
* **The direction of the size error decides which brand suffers.** Tight boxes lose Bego (recall 0.95 → 0.85 at × 0.8). Loose boxes
  push predictions towards Bego: Bego recall rises to 1.0 while ITI recall falls to 0.82 at × 1.5. So the model uses how much of the
  crop the implant fills, or the surrounding context, as a cue for Bego – a reason to keep the crop margin fixed and to report it.

## Caveats
* Section 1 uses the ground truth to pick the detection for each implant (same as the earlier detector brand accuracy); section 2 is the deployment-like view.
* The two-stage classifier was trained on ground-truth crops only; training it on jittered or detector boxes might close the small remaining gap.
* 76 Bego implants: Bego recall SD over folds is 0.04–0.18, and one implant is ≈ 1.3 points of pooled Bego recall.
* One seed, one detector checkpoint per fold, thresholds 0.3 / 0.5 not tuned.

Files: `dental_cls/two_stage.py`, `dental_cls/analyze_ts.py`, `backup/ts_results/` (per-implant predictions), `backup/ts_report/`, `backup/results_ts/` (retrained classifier jobs).
