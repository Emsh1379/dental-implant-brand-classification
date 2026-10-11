# Handoff – dental implant brand classification benchmark

## Where everything is
| What | Location |
|---|---|
| Code, results JSONs, reports, summaries | GitHub `emsh1379/dental-implant-brand-classification`, branch `claude/eager-heisenberg-be64qy` |
| Same + large files (crop arrays, checkpoints, detection weights) | Hugging Face dataset `emad2001/dental-implant-brand-classification` (private; needs your HF token) |
| Original data (HAT-enhanced ToothFairy zip, old backup) | Hugging Face dataset `Mesh2001/Dental_implant` (public) |

### GitHub layout
* `dental_cls/` – all scripts (`prep_crops.py`, `prep_variants.py`, `make_folds.py`, `train_cv.py`, `analyze.py`, `analyze2.py`, `analyze_auc.py`, `analyze_det.py`, `hard_cases.py`, `two_stage.py`, `analyze_ts.py`, `gradcam.py`, `det_prep.py`, `det_train.py`, `models.py`, `common.py`)
* `backup/crops/` – `folds.json` (the fixed 5-fold group split – never regenerate), `meta.csv`, `meta_groups.csv`, `dhash.npy`
* `backup/results/` – round 1: 90 classifier jobs (6 models × raw/pipeline/hat × 5 folds)
* `backup/report/` – round 1 tables, confusion matrices, Grad-CAM
* `backup/results2/`, `backup/report2/` – round 2: pipeline ablation, fixed pipeline, 3-channel, 384 px
* `backup/report3/` – per-brand ROC-AUC / PR-AUC (`auc_per_class.csv`, `auc_vs_raw.csv`, Bego curves) and `consistent_errors.csv` (label-review candidates)
* `backup/det_results/` – detection: YOLO11n, YOLO11n on GT-ROI pipeline images, RF-DETR Small – 5 folds each, complete
* `backup/ts_results/`, `backup/ts_report/`, `backup/results_ts/` – two-stage and box-jitter results, and the 5 retrained DenseNet121 raw384 jobs
* `backup/det_report/` – detection summary (`summary.csv`, `per_fold.csv`, `comparisons.csv`)
* `FINAL_SUMMARY.md` (start here for the paper), `RESULTS.md`, `RESULTS_ROUND2.md`, `RESULTS_AUC.md`, `RESULTS_DETECTION.md`, `RESULTS_TWO_STAGE.md`, `PREVIOUS_vs_NEW.md`, `COMPARISON_WITH_PAPER.md`

### Hugging Face layout (emad2001/dental-implant-brand-classification)
`code/`, `crops/` (raw/pipeline/hat/ablation/fixed/ch3/raw384/hat384 `.npy` + folds/meta), `results/`, `results2/`, `report/`, `report2/`,
`checkpoints/` + `checkpoints_round2/` (fold-0 classifier weights), `det_results/`, `det_checkpoints/` (YOLO best.pt, RF-DETR checkpoint_best_total.pth), `logs/`

## Key results (5-fold group-stratified CV, 1,590 test crops, ~76 Bego)
* Best classifier: **DenseNet121, raw crops 384 px** – macro-F1 0.955, balanced acc 0.972, acc 0.982, Bego recall 0.950
* Preprocessing pipeline never beats raw; significantly worse for DenseNet121/ViT; strong sharpening is the most harmful step; HAT gives no gain
* Detection, 5 folds (mAP50 / mAP50-95 / brand acc / brand macro-F1): YOLO11n 0.887/0.451/0.963/0.905; RF-DETR Small 0.900/0.479/0.968/0.917 (no significant difference)
* Per-brand AUC: best model macro ROC-AUC 0.998, Bego PR-AUC 0.951; Bicon/ITI ROC-AUC ≥ 0.989 for all pretrained models; pipeline shows no AUC benefit, but its AUC loss is small (the significant loss is in macro-F1)
* 22 test crops are wrong in most of the 36 classifier configurations (ITI-178 and Bicon-285 in all 36) – candidates for label review
* GT-ROI preprocessing leak: YOLO11n mAP50-95 0.45 → 0.80 with no gain in brand accuracy – explains the old ~0.90 mAP50-95

## Open next steps
1. Done (2026-10-11): RF-DETR fold 4 fetched, detection summary in `RESULTS_DETECTION.md`
2. Done (2026-10-11): per-class PR-AUC / ROC-AUC in `RESULTS_AUC.md`
3. Done (2026-10-11): two-stage (RF-DETR / YOLO11n boxes → DenseNet121) and box-jitter test in `RESULTS_TWO_STAGE.md`. Two-stage macro-F1 0.948 / 0.956, Bego recall 0.96, equal to the classifier on ground-truth boxes (0.950).
   To re-run: `python train_cv.py --out /tmp/data/results_ts --versions raw384 --models densenet121 --save_all`, `python det_prep.py`, link the detector checkpoints to `/tmp/data/det_runs`, `python two_stage.py`, `python analyze_ts.py` (about 30 min on 2× T4).
   The 5 classifier checkpoints (`results_ts_ckpt/`, 135 MB) and the raw detections (`dets_*.json`) are not on GitHub.
4. Optional: YOLO11-OBB/seg at 1024 px; repeat CV with 3 seeds; use original Roboflow images for "raw"
5. Done (2026-10-11): `FINAL_SUMMARY.md`
7. In progress (started 2026-10-11, Kaggle): seed repeats – seeds 1 and 2 for DenseNet121 (raw, pipeline, hat, raw384) and ViT-B/16 (raw, pipeline, hat), same folds.
   Result JSONs sync to Hugging Face `results_seeds/s1`, `results_seeds/s2` (35 jobs each when complete). Missing jobs can be re-run with
   `python train_cv.py --seed S --out <dir>/sS --versions raw pipeline hat --models densenet121 vit_b_16` and `... --versions raw384 --models densenet121` (finished jobs are skipped).
   Then: `python dental_cls/analyze_seeds.py --results backup/results backup/results2 backup/results_seeds/s1 backup/results_seeds/s2 --out backup/report_seeds`
6. Ask a clinician to check the images in `backup/report3/consistent_errors.csv`

CPU-only analysis (needs numpy, pandas, scipy, scikit-learn, matplotlib):
`python dental_cls/analyze_det.py --results backup/det_results --out backup/det_report`
`python dental_cls/analyze_auc.py --results backup/results backup/results2 --out backup/report3`
`python dental_cls/hard_cases.py --results backup/results backup/results2 --meta backup/crops/meta.csv --out backup/report3`

## Restoring on a new Kaggle GPU server
1. `snapshot_download('emad2001/dental-implant-brand-classification', repo_type='dataset', token=HF_TOKEN, allow_patterns=['code/*','crops/*','det_results/*'])`
2. Copy `crops/*` to `/tmp/data/crops`, `code/*.py` to `/kaggle/working/dental_cls`
3. For detection: download `Teeth-Segmentation-ToothFairy-1-enahnced.zip` from `Mesh2001/Dental_implant`, unzip to `/tmp/data/toothfairy`, `pip install ultralytics "rfdetr[train]"`, run `det_prep.py`
4. Keep outputs out of `/kaggle/working` when large (20 GB limit) – use `/tmp/data`

## Security
Rotate the Hugging Face token and the Roboflow API key that were shared in chat / in the notebook.
