# Seed repeats – how stable are the classifier comparisons?

Same 5 folds and test crops as before; only the training seed changes (`train_cv.py --seed 1` and `--seed 2`, seed 0 = the original runs).
DenseNet121 and ViT-B/16 on raw, pipeline and HAT crops (224 px): 3 seeds × 5 folds = 15 runs per configuration.
**Not finished:** DenseNet121 raw 384 px seeds 1 and 2 – the Kaggle session ended before these 10 jobs ran (see the end of this file).

## Seed-averaged results

| model | input | macro-F1, mean of 15 runs | per seed (0 / 1 / 2) | SD between seeds | accuracy | Bego recall (range over seeds) |
|---|---|---|---|---|---|---|
| DenseNet121 | raw | 0.944 | 0.950 / 0.942 / 0.941 | 0.005 | 0.981 | 0.908 (0.874–0.936) |
| DenseNet121 | pipeline | 0.923 | 0.917 / 0.930 / 0.923 | 0.006 | 0.972 | 0.880 (0.866–0.898) |
| DenseNet121 | HAT | 0.945 | 0.945 / 0.948 / 0.942 | 0.003 | 0.980 | 0.933 (0.916–0.965) |
| ViT-B/16 | raw | 0.951 | 0.950 / 0.952 / 0.951 | 0.001 | 0.984 | 0.912 (0.869–0.942) |
| ViT-B/16 | pipeline | 0.936 | 0.923 / 0.945 / 0.941 | 0.012 | 0.977 | 0.894 (0.846–0.924) |
| ViT-B/16 | HAT | 0.951 | 0.954 / 0.944 / 0.954 | 0.006 | 0.982 | 0.927 (0.906–0.962) |

Noise levels: the 5-fold mean macro-F1 moves by 0.001–0.012 (SD) between seeds; a single fold moves by ≈ 0.016 (SD) between seeds;
the SD between folds is ≈ 0.03. Bego recall of one configuration changes by up to 0.07 between seeds.

## Paired comparisons (same folds; difference averaged over seeds within each fold)

| model | comparison | Δ macro-F1 | per seed | folds where second is better | paired t (5 folds) | McNemar on 3-seed ensemble (Holm) |
|---|---|---|---|---|---|---|
| DenseNet121 | raw → pipeline | **−0.021** | −0.033 / −0.013 / −0.018 | 0 of 5 | p = 0.025 | 23 vs 4, p = 0.0003 (0.002) |
| DenseNet121 | raw → HAT | +0.001 | −0.004 / +0.006 / +0.001 | 3 of 5 | p = 0.88 | 12 vs 3, p = 0.035 (0.18) |
| DenseNet121 | HAT → pipeline | −0.022 | −0.029 / −0.019 / −0.019 | 1 of 5 | p = 0.067 | 17 vs 7, p = 0.064 (0.21) |
| ViT-B/16 | raw → pipeline | −0.014 | −0.026 / −0.006 / −0.010 | 2 of 5 | p = 0.24 | 11 vs 7, p = 0.48 (0.48) |
| ViT-B/16 | raw → HAT | 0.000 | +0.004 / −0.008 / +0.003 | 3 of 5 | p = 1.00 | 3 vs 9, p = 0.15 (0.29) |
| ViT-B/16 | HAT → pipeline | −0.014 | −0.031 / +0.002 / −0.013 | 0 of 5 | p = 0.030 | 16 vs 6, p = 0.053 (0.21) |

"Ensemble" = the class probabilities of the 3 seeds averaged per crop; "23 vs 4" = crops only the first / only the second input gets right.

## What changes for the paper
1. **"The pipeline does not help" holds in every seed.** The pipeline is below raw in all 3 seeds for both models (6 of 6) and never above it.
2. **"The pipeline is significantly worse" holds for DenseNet121 only.** For DenseNet121 the loss is −0.021 macro-F1 (the first seed's −0.033 was
   the largest), present in all 5 folds and significant on the ensemble (Holm p = 0.002). For ViT-B/16 the loss shrinks to −0.014 and is
   **not significant** (paired t p = 0.24, McNemar p = 0.48); the round-1 result (Holm p = 0.044) came from the least favourable seed. Round-1 text that calls the
   pipeline significantly worse for ViT-B/16 should be changed to "lower in every seed, not significant".
3. **HAT = raw.** Seed-averaged differences are +0.001 (DenseNet121) and 0.000 (ViT-B/16). Nothing to claim for super-resolution at 224 px.
4. **Rankings between architectures are within noise.** DenseNet121 raw 0.944 vs ViT-B/16 raw 0.951; the earlier tie at 0.950 was one seed.
5. **Single-seed Bego recall is not reliable to better than ≈ ±0.04.** Report seed-averaged Bego recall.
6. **Seed ensembling helps a little** (pooled macro-F1 of the 3-seed ensemble: DenseNet121 raw 0.963, ViT-B/16 HAT 0.967 – the highest numbers
   in the project), but this is three models, not one, and it was not a planned comparison.

## Raw 384 px – still one seed
DenseNet121 raw 384 px has seed 0 (macro-F1 0.955, Bego recall 0.950) and the independent retraining done for the two-stage experiment
(0.950, 0.952; same nominal seed, different result because GPU training is not deterministic). Against the 3-seed raw 224 px mean (0.944,
Bego recall 0.908) that is +0.006 to +0.011 macro-F1 – the same size as the seed noise. So "384 px is best" should be written as
"384 px was at least as good as 224 px and gave the highest Bego recall in both runs", not as a demonstrated gain, until seeds 1 and 2 are run:

`python train_cv.py --seed 1 --out <dir>/s1 --versions raw384 --models densenet121` (and `--seed 2`, ≈ 30 min each on one T4), then `analyze_seeds.py`.

Files: `backup/results_seeds/s1`, `s2` (30 jobs each), `backup/report_seeds/` (`summary.csv`, `comparisons.csv`, `per_run.csv`), `dental_cls/analyze_seeds.py`.
