# Leakage-free implant-brand classification benchmark

Pipeline (run in this order, see script docstrings):

1. `prep_crops.py` – crops every annotated implant from the *original* Roboflow images (no synthetic class-balancing copies) in three input versions: `raw` (640px), `pipeline` (raw + sharpen/CLAHE-in-ROI/unsharp/percentile normalisation) and `hat` (HAT x4 super-resolved 2560px image).
2. `make_folds.py` – stratified, group-aware k-fold. Groups are Roboflow source images, merged when their dHash shows they are near-duplicates. Evaluation uses one variant per group; Roboflow augmented variants are used for training only.
3. `train_cv.py` – 6 classifiers (custom CNN, VGG16, ResNet50, DenseNet121, EfficientNet-B0, ViT-B/16) x 3 versions x folds. Class-weighted loss instead of oversampling; test fold untouched by model selection.
4. `analyze.py` – mean ± SD over folds, pooled out-of-fold metrics with bootstrap CIs, confusion matrices, paired t / Wilcoxon / McNemar tests (Holm-corrected).
5. `gradcam.py` – Grad-CAM panels from the fold-0 checkpoints.
6. `analyze_auc.py` – per-brand ROC-AUC / PR-AUC from the saved probabilities (fold mean ± SD, pooled with bootstrap CI, paired bootstrap vs raw); `hard_cases.py` – crops that most configurations get wrong.
7. `det_prep.py`, `det_train.py`, `analyze_det.py` – YOLO11n / RF-DETR Small on the same folds and their summary.
8. `two_stage.py`, `analyze_ts.py` – detector box → classifier on the same folds (needs `train_cv.py --save_all`) and the box-jitter robustness test.

Caveat: the `raw` version is the HAT image area-downsampled to 640px, a proxy for the original Roboflow download.

## Backup / resuming after a Kaggle disconnect

`backup/` holds everything needed to continue without recomputing finished work: per-job result JSONs
(`backup/results`, one per rep/fold/version/model, with test probabilities), the fold definitions and crop
metadata (`backup/crops/folds.json`, `meta.csv`, `meta_groups.csv`, `dhash.npy`) and the training logs.
Large, rebuildable artefacts (crops `.npy`, checkpoints, raw dataset) are not stored in git.

To resume on a fresh server:
1. Download `Mesh2001/Dental_implant` from Hugging Face, unzip `Teeth-Segmentation-ToothFairy-1-enahnced.zip` to `/tmp/data/toothfairy`.
2. `python prep_crops.py` (rebuilds the crop arrays; row order matches `backup/crops/meta.csv`).
3. Copy `backup/crops/folds.json` to `/tmp/data/crops/` (do not re-run `make_folds.py`) and `backup/results/*.json` to `/kaggle/working/results/`.
4. `python train_cv.py ...` – finished jobs are skipped automatically.
