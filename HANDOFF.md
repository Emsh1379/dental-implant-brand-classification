# Handoff – dental implant brand classification benchmark

## Where everything is
| What | Location |
|---|---|
| Code, results JSONs, reports, summaries | GitHub `emsh1379/dental-implant-brand-classification`, branch `claude/eager-heisenberg-be64qy` |
| Same + large files (crop arrays, checkpoints, detection weights) | Hugging Face dataset `emad2001/dental-implant-brand-classification` (private; needs your HF token) |
| Original data (HAT-enhanced ToothFairy zip, old backup) | Hugging Face dataset `Mesh2001/Dental_implant` (public) |

### GitHub layout
* `dental_cls/` – all scripts (`prep_crops.py`, `prep_variants.py`, `make_folds.py`, `train_cv.py`, `analyze.py`, `analyze2.py`, `gradcam.py`, `det_prep.py`, `det_train.py`, `models.py`, `common.py`)
* `backup/crops/` – `folds.json` (the fixed 5-fold group split – never regenerate), `meta.csv`, `meta_groups.csv`, `dhash.npy`
* `backup/results/` – round 1: 90 classifier jobs (6 models × raw/pipeline/hat × 5 folds)
* `backup/report/` – round 1 tables, confusion matrices, Grad-CAM
* `backup/results2/`, `backup/report2/` – round 2: pipeline ablation, fixed pipeline, 3-channel, 384 px
* `backup/det_results/` – detection: YOLO11n (5 folds), YOLO11n on GT-ROI pipeline images (5), RF-DETR Small (folds 0–3; fold 4 uploads to HF when done)
* `RESULTS.md`, `RESULTS_ROUND2.md`, `PREVIOUS_vs_NEW.md`, `COMPARISON_WITH_PAPER.md`

### Hugging Face layout (emad2001/dental-implant-brand-classification)
`code/`, `crops/` (raw/pipeline/hat/ablation/fixed/ch3/raw384/hat384 `.npy` + folds/meta), `results/`, `results2/`, `report/`, `report2/`,
`checkpoints/` + `checkpoints_round2/` (fold-0 classifier weights), `det_results/`, `det_checkpoints/` (YOLO best.pt, RF-DETR checkpoint_best_total.pth), `logs/`

## Key results (5-fold group-stratified CV, 1,590 test crops, ~76 Bego)
* Best classifier: **DenseNet121, raw crops 384 px** – macro-F1 0.955, balanced acc 0.972, acc 0.982, Bego recall 0.950
* Preprocessing pipeline never beats raw; significantly worse for DenseNet121/ViT; strong sharpening is the most harmful step; HAT gives no gain
* Detection (mAP50 / mAP50-95 / brand acc / brand macro-F1): YOLO11n 0.887/0.451/0.963/0.905; RF-DETR (4 folds) 0.901/0.482/0.973/0.920
* GT-ROI preprocessing leak: YOLO11n mAP50-95 0.45 → 0.80 with no gain in brand accuracy – explains the old ~0.90 mAP50-95

## Open next steps
1. Fetch RF-DETR fold 4 from HF (`det_results/rfdetr_small_f4.json`); re-run it if missing:
   `python det_prep.py && python det_train.py --order rfdetr_small --folds 4`
2. Per-class PR-AUC / ROC-AUC for classifiers from saved probabilities (CPU only)
3. Two-stage pipeline: RF-DETR boxes → DenseNet121 brand; box-jitter robustness test
4. Optional: YOLO11-OBB/seg at 1024 px; repeat CV with 3 seeds; use original Roboflow images for "raw"
5. Write combined final summary for the paper

## Restoring on a new Kaggle GPU server
1. `snapshot_download('emad2001/dental-implant-brand-classification', repo_type='dataset', token=HF_TOKEN, allow_patterns=['code/*','crops/*','det_results/*'])`
2. Copy `crops/*` to `/tmp/data/crops`, `code/*.py` to `/kaggle/working/dental_cls`
3. For detection: download `Teeth-Segmentation-ToothFairy-1-enahnced.zip` from `Mesh2001/Dental_implant`, unzip to `/tmp/data/toothfairy`, `pip install ultralytics "rfdetr[train]"`, run `det_prep.py`
4. Keep outputs out of `/kaggle/working` when large (20 GB limit) – use `/tmp/data`

## Security
Rotate the Hugging Face token and the Roboflow API key that were shared in chat / in the notebook.
