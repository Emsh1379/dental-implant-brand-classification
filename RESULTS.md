# Results – leakage-free 5-fold group-stratified CV (rep 0)

3 brands (Bego 171 / Bicon 1,879 / ITI 1,226 crops from 1,009 source-image groups), 6 pretrained/from-scratch classifiers x 3 input versions.
Evaluation uses 1,590 crops (one variant per source image), every crop tested exactly once. Macro-F1 / balanced accuracy are the primary metrics because Bego is rare.

Macro-F1 (mean ± SD over 5 folds)

| model | raw | pipeline | hat |
|---|---|---|---|
| custom CNN (scratch) | 0.822 ± 0.025 | 0.783 ± 0.064 | 0.803 ± 0.032 |
| VGG16 | 0.937 ± 0.032 | 0.925 ± 0.041 | 0.940 ± 0.040 |
| ResNet50 | 0.935 ± 0.041 | 0.923 ± 0.028 | 0.933 ± 0.036 |
| DenseNet121 | 0.950 ± 0.030 | 0.917 ± 0.026 | 0.945 ± 0.039 |
| EfficientNet-B0 | 0.918 ± 0.030 | 0.920 ± 0.035 | 0.937 ± 0.025 |
| ViT-B/16 | 0.950 ± 0.028 | 0.923 ± 0.037 | 0.954 ± 0.028 |

Findings
* Pretrained models reach 0.92–0.95 macro-F1 (97–98% accuracy); the scratch CNN is clearly worse (0.78–0.82).
* The authors' sharpen/CLAHE/unsharp pipeline did **not** improve classification: it is never significantly better than raw or HAT and is slightly worse for DenseNet121 and ViT-B/16 (McNemar, Holm-corrected p = 0.003 and 0.044 vs raw).
* HAT super-resolution vs raw: no significant difference for any model (all Holm-adjusted p = 1.0), so HAT's benefit is not demonstrated either.
* Bego recall (the rare class) is 0.80–0.91 for the best models, vs ≥0.97 for ITI/Bicon; Bego is the main error source (see confusion matrices).
* With 5 folds the paired fold-level tests have little power (minimum Wilcoxon p = 0.0625); pooled McNemar tests are the more informative ones.

Caveats
* `raw` = HAT image area-downsampled to 640 px (proxy, not the original Roboflow file).
* The `pipeline` ROI step uses ground-truth implant boxes at test time; a deployed system would need a detector first.
* Single repetition of the fold split; repeating with other seeds would tighten the estimates.
* Near-duplicate check found one pair (Bicon-289 / Bicon-291) straddling the original train/test split, now merged into one group.

Files: `backup/report/` (tables, confusion matrices, Grad-CAM), `backup/results/` (per-job predictions).
