# Per-brand ROC-AUC and PR-AUC (classifiers)

Computed from the saved test probabilities of all 180 classifier jobs (no retraining). One-vs-rest per brand.
"Pooled" = the 1,590 out-of-fold crops together (76 Bego, prevalence 4.8%), with a bootstrap 95% CI (2,000 resamples).
Fold mean ± SD is in `backup/report3/auc_per_class.csv`. PR-AUC = average precision; for Bego a random ranking scores 0.048.

## Main table (pooled out-of-fold)

| model | input | macro ROC-AUC | macro PR-AUC | Bego ROC-AUC [95% CI] | Bego PR-AUC [95% CI] |
|---|---|---|---|---|---|
| custom CNN | raw | 0.975 | 0.904 | 0.949 [0.919, 0.975] | 0.740 [0.644, 0.822] |
| VGG16 | raw | 0.986 | 0.965 | 0.966 [0.933, 0.992] | 0.905 [0.837, 0.962] |
| ResNet50 | raw | 0.989 | 0.968 | 0.973 [0.944, 0.997] | 0.913 [0.848, 0.964] |
| DenseNet121 | raw | 0.992 | 0.966 | 0.982 [0.958, 0.998] | 0.907 [0.841, 0.959] |
| EfficientNet-B0 | raw | 0.986 | 0.951 | 0.970 [0.940, 0.992] | 0.865 [0.793, 0.925] |
| ViT-B/16 | raw | 0.988 | 0.966 | 0.970 [0.937, 0.995] | 0.904 [0.831, 0.961] |
| DenseNet121 | pipeline | 0.995 | 0.961 | 0.990 [0.981, 0.997] | 0.888 [0.803, 0.957] |
| DenseNet121 | hat | 0.995 | 0.967 | 0.991 [0.981, 0.998] | 0.908 [0.837, 0.962] |
| DenseNet121 | fixed | 0.991 | 0.954 | 0.976 [0.953, 0.994] | 0.866 [0.772, 0.953] |
| DenseNet121 | 3-channel | 0.992 | 0.969 | 0.982 [0.960, 0.997] | 0.913 [0.851, 0.962] |
| **DenseNet121** | **raw 384 px** | **0.998** | **0.982** | **0.998 [0.995, 1.000]** | **0.951 [0.884, 0.992]** |
| DenseNet121 | hat 384 px | 0.995 | 0.971 | 0.989 [0.978, 0.998] | 0.919 [0.854, 0.969] |
| EfficientNet-B0 | hat | 0.995 | 0.976 | 0.996 [0.994, 0.999] | 0.939 [0.895, 0.972] |
| EfficientNet-B0 | raw 384 px | 0.983 | 0.964 | 0.959 [0.917, 0.992] | 0.908 [0.845, 0.964] |
| ViT-B/16 | hat | 0.989 | 0.973 | 0.975 [0.943, 0.997] | 0.927 [0.870, 0.974] |

For every pretrained model and input, Bicon ROC-AUC is ≥ 0.996 and ITI ROC-AUC is ≥ 0.989 (PR-AUC ≥ 0.992 and ≥ 0.980).
All 36 model × input rows: `backup/report3/auc_per_class.csv`. Bego curves for DenseNet121: `backup/report3/bego_curves_densenet121.png`.

## Findings
1. **Bicon and ITI are separated almost perfectly; Bego is the only brand with room to improve.** Bego PR-AUC is 0.83–0.95 for
   pretrained models (0.66–0.78 for the scratch CNN), and its CIs are wide (≈ ±0.06) because there are only 76 Bego test crops.
2. **Best model is the same as by macro-F1:** DenseNet121 on raw 384 px crops – macro ROC-AUC 0.998, macro PR-AUC 0.982,
   Bego PR-AUC 0.951. Against raw 224 px: macro ROC-AUC +0.006 (paired bootstrap p = 0.014), Bego PR-AUC +0.044 (CI −0.003 to +0.097, p = 0.07).
3. **The preprocessing pipeline does not improve ranking either, but the AUC differences are smaller than the macro-F1 differences.**
   Bego PR-AUC is lower with the pipeline for all five pretrained models (−0.012 to −0.081), significant only for ResNet50
   (−0.081, p = 0.033 uncorrected). For DenseNet121 the pipeline's ROC-AUC is not lower than raw (macro 0.995 vs 0.992) even though its
   macro-F1 was significantly lower (0.917 vs 0.950, Holm p = 0.003). So for DenseNet121 the pipeline's loss shows at the argmax
   decision, not in the ranking of probabilities – the paper should say "no benefit, lower macro-F1" rather than "worse on every metric".
4. **HAT: no consistent effect.** Bego PR-AUC changes by −0.003 to +0.024 for VGG16, ResNet50, DenseNet121 and ViT-B/16 (all n.s.).
   EfficientNet-B0 is the exception: +0.074 (CI +0.029 to +0.131, p < 0.001) at 224 px, but the gain is not there at 384 px
   (hat384 0.874 vs raw384 0.908). The scratch CNN gets worse with HAT (−0.077, p = 0.017).
5. **No ablation, revised pipeline or 3-channel input is significantly better than raw** on Bego PR-AUC for DenseNet121 or ViT-B/16 (largest gain +0.015, n.s.) (`backup/report3/auc_vs_raw.csv`).

## Crops that most models get wrong (candidates for label review)
22 of 1,590 test crops are misclassified by at least half of the 36 model × input configurations (`backup/report3/consistent_errors.csv`).
Two are wrong in all 36: **ITI-178** (labelled ITI, predicted Bego with mean probability 0.96) and **Bicon-285** (labelled Bicon,
predicted ITI, 0.71). Others near the top: ITI-432, ITI-10, ITI-288 (→ Bego) and bego-11, bego-27 (→ ITI). Errors that are this
consistent across architectures and inputs are more likely to be label or image problems than model weaknesses; a clinician should check these images.
Because Bego has only 76 test crops, a few such cases move Bego recall by several points.

## Caveats
* Pooled AUC mixes probabilities from 5 different fold models; fold mean ± SD is reported next to it in the CSV and agrees to within ≈ 0.03 for Bego PR-AUC.
* Paired bootstrap p-values are **not** corrected for multiple comparisons (30 version-vs-raw comparisons × 4 metrics); treat p ≈ 0.01–0.05 as suggestive.
* The bootstrap resamples crops, not source images; several crops can come from one image.
* Saved probabilities are rounded to 5 decimals, which creates a few ties (handled as ties).
