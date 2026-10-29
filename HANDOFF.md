# Handoff – dental implant brand classification benchmark

## Where everything is
| What | Location |
|---|---|
| Code, results JSONs, reports, summaries | GitHub `emsh1379/dental-implant-brand-classification`, branch `claude/eager-heisenberg-be64qy` |
| Same + large files (crop arrays, checkpoints, detection weights) | Hugging Face dataset `emad2001/dental-implant-brand-classification` (private; needs your HF token) |
| Original data (HAT-enhanced ToothFairy zip, old backup) | Hugging Face dataset `Mesh2001/Dental_implant` (public) |
| Original images without super-resolution | Roboflow `dental-implant-nyw2h/-dental-implant`, **version 1** (CC BY 4.0); `prep_orig.py` downloads and aligns it |

### GitHub layout
* `dental_cls/` – all scripts (`prep_crops.py`, `prep_variants.py`, `make_folds.py`, `train_cv.py`, `analyze.py`, `analyze2.py`, `analyze_auc.py`, `analyze_det.py`, `hard_cases.py`, `two_stage.py`, `analyze_ts.py`, `gradcam.py`, `det_prep.py`, `det_train.py`, `models.py`, `common.py`)
* `backup/crops/` – `folds.json` (the fixed 5-fold group split – never regenerate), `meta.csv`, `meta_groups.csv`, `dhash.npy`
* `backup/results/` – round 1: 90 classifier jobs (6 models × raw/pipeline/hat × 5 folds)
* `backup/report/` – round 1 tables, confusion matrices, Grad-CAM
* `backup/results2/`, `backup/report2/` – round 2: pipeline ablation, fixed pipeline, 3-channel, 384 px
* `backup/report3/` – per-brand ROC-AUC / PR-AUC (`auc_per_class.csv`, `auc_vs_raw.csv`, Bego curves) and `consistent_errors.csv` (label-review candidates)
* `backup/det_results/` – detection: YOLO11n, YOLO11n on GT-ROI pipeline images, RF-DETR Small – 5 folds each, complete
* `backup/ts_results/`, `backup/ts_report/`, `backup/results_ts/` – two-stage and box-jitter results, and the 5 retrained DenseNet121 raw384 jobs
* `backup/results_seeds/`, `backup/report_seeds/` – seeds 1 and 2 (35 jobs each), same-seed control re-runs (`s0_rerun384`, `s1_rerun224`, `s2_rerun224`) and the 3-seed analysis
* `backup/results_orig/`, `backup/report_orig/` – DenseNet121 / ViT-B/16 on the original images (raw_orig, pipeline_orig), 3 seeds
* `backup/results_trainplus*/`, `backup/report_trainplus*/` – extra Bego training data (user images; + panoramics)
* `backup/results_external/`, `backup/report_external/`, `backup/external_match/` – OII-DS / Zenodo / Mendeley candidate search and retraining (CSV + review sheets)
* `backup/results_paper_pipeline/`, `backup/report_paper_pipeline/` – Tiryaki et al. 2023 pipeline (5 CNNs + fusion, angle regression), grouped and random split
* `backup/extra_labels/` – auto-labels (YOLO txt + COCO) of the extra Bego images; the images themselves are only on Hugging Face (internet-sourced)
* `backup/det_report/` – detection summary (`summary.csv`, `per_fold.csv`, `comparisons.csv`)
* `FINAL_SUMMARY.md` (start here for the paper), `RESULTS_SEEDS.md`, `RESULTS.md`, `RESULTS_ROUND2.md`, `RESULTS_AUC.md`, `RESULTS_DETECTION.md`, `RESULTS_TWO_STAGE.md`, `PREVIOUS_vs_NEW.md`, `COMPARISON_WITH_PAPER.md`

### Hugging Face layout (emad2001/dental-implant-brand-classification)
`code/`, `crops/` (raw/pipeline/hat/ablation/fixed/ch3/raw384/hat384 `.npy` + folds/meta), `results/`, `results2/`, `report/`, `report2/`,
`checkpoints/` + `checkpoints_round2/` (fold-0 classifier weights), `det_results/`, `det_checkpoints/` (YOLO best.pt, RF-DETR checkpoint_best_total.pth), `logs/`

## Key results (5-fold group-stratified CV, 1,590 test crops, ~76 Bego)
* Best classifier: **ViT-B/16 on the original images**, macro-F1 0.954, acc 0.986 (3-seed mean; 3-seed ensemble 0.972, Bego recall 0.961)
* On the HAT-derived proxy raw (3-seed means): ViT-B/16 macro-F1 0.951, DenseNet121 0.944 (acc 0.984 / 0.981, Bego recall 0.912 / 0.908).
  DenseNet121 raw 384 px was 0.955 in its first run (Bego recall 0.950) but 0.936 over 3 seeds – no gain over 224 px
* Preprocessing pipeline never beats raw; significantly worse for DenseNet121/ViT; strong sharpening is the most harmful step; HAT gives no gain
* Detection, 5 folds (mAP50 / mAP50-95 / brand acc / brand macro-F1): YOLO11n 0.887/0.451/0.963/0.905; RF-DETR Small 0.900/0.479/0.968/0.917 (no significant difference)
* Per-brand AUC: best model macro ROC-AUC 0.998, Bego PR-AUC 0.951; Bicon/ITI ROC-AUC ≥ 0.989 for all pretrained models; pipeline shows no AUC benefit, but its AUC loss is small (the significant loss is in macro-F1)
* 3 seeds: pipeline below raw in every seed; significant for DenseNet121 (−0.021), not for ViT-B/16 (−0.014); HAT = raw; 384 px = 224 px (−0.008, p = 0.14)
* Original images vs proxy raw vs HAT: all within noise. Pipeline on the original images: ViT-B/16 −0.031 (Holm p = 0.001), DenseNet121 −0.008 (n.s.)
* Training is not deterministic: the same seed re-run moves the 5-fold mean macro-F1 by 0.006–0.027, so single-run differences below ≈ 0.02 are noise
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
7. Done (2026-10-12), see `RESULTS_SEEDS.md`: seed repeats – seeds 1 and 2 for DenseNet121 (raw, pipeline, hat, raw384) and ViT-B/16 (raw, pipeline, hat), same folds,
   35 of 35 jobs each, plus 15 same-seed control jobs (DenseNet121 raw384 seed 0, raw seeds 1 and 2). Result JSONs are on GitHub and on Hugging Face under `results_seeds/`. To re-run:
   `python train_cv.py --seed S --out <dir>/sS --versions raw pipeline hat --models densenet121 vit_b_16` and `... --versions raw384 --models densenet121` (finished jobs are skipped).
   Then: `python dental_cls/analyze_seeds.py --results backup/results backup/results2 backup/results_seeds/s1 backup/results_seeds/s2 --out backup/report_seeds`
8. Done (2026-10-25), see `RESULTS_ORIGINAL.md`: original Roboflow images (no super-resolution) for DenseNet121 and ViT-B/16, raw and pipeline, 3 seeds (60 jobs).
   `python prep_orig.py` (needs `ROBOFLOW_API_KEY`, `raw.npy`, `meta.csv`), then `train_cv.py --versions raw_orig pipeline_orig`, then `analyze_seeds.py --set orig`.
9. Done (2026-10-27), `RESULTS_EXTRA_BEGO.md`: 18 lamiaa images + 10 user periapicals + 3 user panoramics (auto-labelled Bego) as training-only data – no reliable gain.
10. Done (2026-10-27), `RESULTS_EXTERNAL.md`: pseudo-label search in OII-DS, Zenodo 15487430, Mendeley x4gr6mmwy4 (`match_oii.py`); retraining with 179 candidates – worse.
    OII-DS candidates are in numbered review sheets (Hugging Face `oii_match/review_*.jpg`) if a clinician wants to confirm them.
11. Done (2026-10-27), `RESULTS_PAPER_PIPELINE.md`: Tiryaki et al. pipeline (`prep_paper.py`, `train_paper.py`, `analyze_paper.py`).
    Hand-checked angle labels would be needed for a paper-grade angle result.
12. Done (2026-10-28), `RESULTS_SYNTHETIC.md`: per-fold class-conditional diffusion (`cdiff_brand.py`, `qc_syn.py`, `build_syn.py`, `syn_pipeline.py`).
13. Done (2026-10-29), `RESULTS_PROTOCOLS.md` and `RESULTS_OPENSET_CALIBRATION.md` (`lashaki_protocol.py`, `tekis_yolo.py`, `openset.py`, `calibration.py`, `pipeline3.py`).
14. Done (2026-10-29): benchmark release – `BENCHMARK.md`, `benchmark/` (crops.csv, folds.json, build_crops.py, evaluate.py, baselines/), public data on
    Hugging Face `emad2001/dental-implant-brand-benchmark`. Clinician review pack: `make_review.py` -> 23 cases (22 persistent errors + the tiny empty box).
6. Ask a clinician to check the images in `backup/report3/consistent_errors.csv`, and crop idx 2882 (box with no implant)

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
Rotate the Hugging Face token, the GitHub token and the Roboflow API key that were shared in chat / in the notebook.
