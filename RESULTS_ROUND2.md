# Round 2 – preprocessing ablation, revised pipelines, 3-channel input, resolution

Same 5 group-stratified folds as round 1 (1,590 test crops, ~76 Bego). Macro-F1 mean ± SD over folds;
Holm-corrected McNemar p-values on pooled out-of-fold predictions (`backup/report2/comparisons.csv`).

| config | DenseNet121 | ViT-B/16 | EfficientNet-B0 |
|---|---|---|---|
| raw (224) | 0.950 ± 0.030 | 0.950 ± 0.028 | 0.918 ± 0.030 |
| hat (224) | 0.945 ± 0.039 | **0.954 ± 0.028** | 0.937 ± 0.025 |
| original pipeline | 0.917 ± 0.026 | 0.923 ± 0.037 | 0.920 ± 0.035 |
| − strong sharpen | 0.935 ± 0.033 | 0.945 ± 0.011 | |
| − 3×3 blur | 0.927 ± 0.035 | 0.927 ± 0.036 | |
| − ROI CLAHE | 0.922 ± 0.013 | 0.929 ± 0.030 | |
| − unsharp mask | 0.918 ± 0.052 | 0.923 ± 0.035 | |
| − ROI percentile rescale | 0.935 ± 0.047 | 0.918 ± 0.039 | |
| fixed (NLM → global CLAHE → mild unsharp) | 0.940 ± 0.036 | 0.944 ± 0.031 | |
| 3-channel (raw / CLAHE / detail) | 0.947 ± 0.045 | 0.925 ± 0.023 | |
| raw 384 px | **0.955 ± 0.024** | | **0.940 ± 0.027** |
| hat 384 px | 0.929 ± 0.027 | | 0.926 ± 0.028 |

Best overall: **DenseNet121, raw crops at 384 px** – macro-F1 0.955, balanced accuracy 0.972, accuracy 0.982, **Bego recall 0.950** (highest of all runs).

## Findings
1. **The strong sharpening step (×1.9) is the most harmful step.** Removing it recovers most of the loss
   (DenseNet121 +0.019, paired t p = 0.034; ViT-B/16 +0.022). Removing any other single step changes little,
   so the remaining loss comes from the steps together (stacked sharpening + ROI-only contrast).
2. **The fixed pipeline repairs the damage but does not beat raw.** It is significantly better than the original
   pipeline for DenseNet121 (Holm p = 0.040) and statistically equal to raw for both models.
3. **3-channel input did not help:** equal to raw for DenseNet121, worse for ViT-B/16 (−0.025, Holm p = 0.087).
4. **Resolution helps more than enhancement.** Raw 384 px gave the best DenseNet121 and EfficientNet-B0 results
   (EfficientNet +0.022 vs 224 px; McNemar p = 0.036 uncorrected, not significant after Holm) and the best Bego recall.
5. **HAT does not help even at 384 px:** hat384 is below raw384 for both models (DenseNet121 −0.027, McNemar p = 0.020 uncorrected).

## Conclusion for the paper
On this dataset hand-crafted enhancement and super-resolution do not improve brand classification by pretrained CNNs/ViTs; the
original pipeline is harmful mainly because of strong sharpening. Plain crops at higher input resolution are the best choice.
Note: with 5 folds and Holm correction across 25 comparisons, only large effects reach significance; differences of 0.01–0.02 are within noise.
