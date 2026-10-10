# Leakage-free implant-brand classification benchmark

Pipeline (run in this order, see script docstrings):

1. `prep_crops.py` – crops every annotated implant from the *original* Roboflow images (no synthetic class-balancing copies) in three input versions: `raw` (640px), `pipeline` (raw + sharpen/CLAHE-in-ROI/unsharp/percentile normalisation) and `hat` (HAT x4 super-resolved 2560px image).
2. `make_folds.py` – stratified, group-aware k-fold. Groups are Roboflow source images, merged when their dHash shows they are near-duplicates. Evaluation uses one variant per group; Roboflow augmented variants are used for training only.
3. `train_cv.py` – 6 classifiers (custom CNN, VGG16, ResNet50, DenseNet121, EfficientNet-B0, ViT-B/16) x 3 versions x folds. Class-weighted loss instead of oversampling; test fold untouched by model selection.
4. `analyze.py` – mean ± SD over folds, pooled out-of-fold metrics with bootstrap CIs, confusion matrices, paired t / Wilcoxon / McNemar tests (Holm-corrected).
5. `gradcam.py` – Grad-CAM panels from the fold-0 checkpoints.

Caveat: the `raw` version is the HAT image area-downsampled to 640px, a proxy for the original Roboflow download.
