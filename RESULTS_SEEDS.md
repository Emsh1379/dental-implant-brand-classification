# Seed repeats – how stable are the classifier comparisons?

Same 5 folds and test crops as before; only the training seed changes (`train_cv.py --seed 1` and `--seed 2`, seed 0 = the original runs).
DenseNet121 and ViT-B/16 on raw, pipeline and HAT crops (224 px), and DenseNet121 on raw 384 px crops: 3 seeds × 5 folds = 15 runs per configuration.
All 7 configurations are complete (raw 384 px seeds 1 and 2 were added on 2026-10-12). Three control re-runs with an unchanged seed are in
"Run-to-run variation" below – read that section before interpreting the "SD between seeds" column.

## Seed-averaged results

| model | input | macro-F1, mean of 15 runs | per seed (0 / 1 / 2) | SD between seeds | accuracy | Bego recall (range over seeds) |
|---|---|---|---|---|---|---|
| DenseNet121 | raw | 0.944 | 0.950 / 0.942 / 0.941 | 0.005 | 0.981 | 0.908 (0.874–0.936) |
| DenseNet121 | pipeline | 0.923 | 0.917 / 0.930 / 0.923 | 0.006 | 0.972 | 0.880 (0.866–0.898) |
| DenseNet121 | HAT | 0.945 | 0.945 / 0.948 / 0.942 | 0.003 | 0.980 | 0.933 (0.916–0.965) |
| DenseNet121 | raw 384 px | 0.936 | 0.955 / 0.929 / 0.925 | 0.017 | 0.976 | 0.909 (0.855–0.950) |
| ViT-B/16 | raw | 0.951 | 0.950 / 0.952 / 0.951 | 0.001 | 0.984 | 0.912 (0.869–0.942) |
| ViT-B/16 | pipeline | 0.936 | 0.923 / 0.945 / 0.941 | 0.012 | 0.977 | 0.894 (0.846–0.924) |
| ViT-B/16 | HAT | 0.951 | 0.954 / 0.944 / 0.954 | 0.006 | 0.982 | 0.927 (0.906–0.962) |

Noise levels: the 5-fold mean macro-F1 moves by 0.001–0.017 (SD) between seeds; a single fold moves by ≈ 0.012–0.024 (SD) between seeds;
the SD between folds is ≈ 0.03. Bego recall of one configuration changes by up to 0.10 between seeds.

## Paired comparisons (same folds; difference averaged over seeds within each fold)

| model | comparison | Δ macro-F1 | per seed | folds where second is better | paired t (5 folds) | McNemar on 3-seed ensemble (Holm) |
|---|---|---|---|---|---|---|
| DenseNet121 | raw → pipeline | **−0.021** | −0.033 / −0.013 / −0.018 | 0 of 5 | p = 0.025 | 23 vs 4, p = 0.0003 (0.002) |
| DenseNet121 | raw → HAT | +0.001 | −0.004 / +0.006 / +0.001 | 3 of 5 | p = 0.88 | 12 vs 3, p = 0.035 (0.21) |
| DenseNet121 | raw → raw 384 px | −0.008 | +0.006 / −0.013 / −0.016 | 1 of 5 | p = 0.14 | 12 vs 7, p = 0.36 (0.72) |
| DenseNet121 | HAT → pipeline | −0.022 | −0.029 / −0.019 / −0.019 | 1 of 5 | p = 0.067 | 17 vs 7, p = 0.064 (0.26) |
| ViT-B/16 | raw → pipeline | −0.014 | −0.026 / −0.006 / −0.010 | 2 of 5 | p = 0.24 | 11 vs 7, p = 0.48 (0.72) |
| ViT-B/16 | raw → HAT | 0.000 | +0.004 / −0.008 / +0.003 | 3 of 5 | p = 1.00 | 3 vs 9, p = 0.15 (0.44) |
| ViT-B/16 | HAT → pipeline | −0.014 | −0.031 / +0.002 / −0.013 | 0 of 5 | p = 0.030 | 16 vs 6, p = 0.053 (0.26) |

"Ensemble" = the class probabilities of the 3 seeds averaged per crop; "23 vs 4" = crops only the first / only the second input gets right.
Holm correction is now over 7 comparisons (it was 6 before the 384 px row was added), so the unadjusted p-values are unchanged and the Holm values are slightly larger.

## What changes for the paper
1. **"The pipeline does not help" holds in every seed.** The pipeline is below raw in all 3 seeds for both models (6 of 6) and never above it.
2. **"The pipeline is significantly worse" holds for DenseNet121 only.** For DenseNet121 the loss is −0.021 macro-F1 (the first seed's −0.033 was
   the largest), present in all 5 folds and significant on the ensemble (Holm p = 0.002). For ViT-B/16 the loss shrinks to −0.014 and is
   **not significant** (paired t p = 0.24, McNemar p = 0.48); the round-1 result (Holm p = 0.044) came from the least favourable seed. Round-1 text that calls the
   pipeline significantly worse for ViT-B/16 should be changed to "lower in every seed, not significant".
3. **HAT = raw.** Seed-averaged differences are +0.001 (DenseNet121) and 0.000 (ViT-B/16). Nothing to claim for super-resolution at 224 px.
4. **Rankings between architectures are within noise.** DenseNet121 raw 0.944 vs ViT-B/16 raw 0.951; the earlier tie at 0.950 was one seed.
   The same holds for resolution: **raw 384 px is not better than raw 224 px** (see below).
5. **Single-seed Bego recall is not reliable to better than ≈ ±0.05.** Report seed-averaged Bego recall.
6. **Seed ensembling helps a little** (pooled macro-F1 of the 3-seed ensemble: DenseNet121 raw 0.963, raw 384 px 0.964, ViT-B/16 HAT 0.967 – the highest numbers
   in the project), but this is three models, not one, and it was not a planned comparison. The 384 px ensemble has the highest Bego recall (0.974, 74 of 76).

## Raw 384 px – three seeds: no gain over 224 px
The round-2 result that made DenseNet121 raw 384 px the "best classifier" (macro-F1 0.955, Bego recall 0.950) was the most favourable run.
Seeds 1 and 2 give 0.929 and 0.925 (Bego recall 0.855 and 0.921). The 3-seed mean is **0.936 against 0.944 for raw 224 px**: Δ = −0.008,
384 px better in 1 of 3 seeds and 1 of 5 folds, paired t p = 0.14, ensemble McNemar 12 vs 7, p = 0.36. Seed-averaged Bego recall is the same (0.909 vs 0.908),
and 384 px has the largest spread between seeds of all 7 configurations (SD 0.017).

So the paper should say: **"increasing the crop resolution from 224 to 384 px did not improve DenseNet121 (macro-F1 0.936 vs 0.944 over 3 seeds, not significant)"**,
and the headline single-model number should be the 3-seed 224 px value (DenseNet121 raw 0.944, ViT-B/16 raw 0.951), not 0.955.
The two-stage results in `RESULTS_TWO_STAGE.md` are unaffected: they compare each detector → classifier system with the *same* 384 px checkpoints on ground-truth boxes.

## Run-to-run variation – the same seed does not give the same result
`train_cv.py` fixes the seed of the weight initialisation and batch order, but GPU training is not deterministic (cuDNN, mixed precision), so a job
repeated with the same seed is a new sample. Five DenseNet121 runs per input are now available (`backup/report_seeds/replicates_densenet121.csv`):

| input | run | macro-F1 (5-fold mean) | accuracy | Bego recall | Bego missed (of 76) |
|---|---|---|---|---|---|
| raw 224 px | seed 0 (round 1) | 0.950 | 0.983 | 0.914 | 7 |
| raw 224 px | seed 1 (10-11) | 0.942 | 0.982 | 0.874 | 10 |
| raw 224 px | seed 2 (10-11) | 0.941 | 0.979 | 0.936 | 5 |
| raw 224 px | seed 1, re-run (10-12) | 0.953 | 0.983 | 0.933 | 5 |
| raw 224 px | seed 2, re-run (10-12) | 0.958 | 0.985 | 0.950 | 4 |
| raw 384 px | seed 0 (round 2) | 0.955 | 0.982 | 0.950 | 4 |
| raw 384 px | seed 0, re-run (two-stage) | 0.950 | 0.980 | 0.952 | 4 |
| raw 384 px | seed 0, re-run (10-12) | 0.929 | 0.975 | 0.902 | 8 |
| raw 384 px | seed 1 (10-12) | 0.929 | 0.974 | 0.855 | 11 |
| raw 384 px | seed 2 (10-12) | 0.925 | 0.971 | 0.921 | 6 |
| **raw 224 px, 5 runs** | mean (SD, range) | **0.949** (0.007, 0.941–0.958) | 0.982 | 0.922 (0.874–0.950) | |
| **raw 384 px, 5 runs** | mean (SD, range) | **0.938** (0.014, 0.925–0.955) | 0.976 | 0.916 (0.855–0.952) | |

* **Re-running with the same seed changes the 5-fold mean macro-F1 by 0.006–0.027 (mean 0.016)** and a single fold by ≈ 0.018 (SD). This is as large as the
  change between different seeds, so the seed number carries no information: every run is an independent replicate, and the "SD between seeds" column above
  measures general training noise. With only 3 runs it is itself imprecise (e.g. 0.001 for ViT-B/16 raw is a lucky small value, not a property of ViT).
* Using all 5 runs per input, 384 px vs 224 px is −0.011 macro-F1 (Welch t on run means p = 0.16; paired over folds p = 0.15; 384 px lower in 4 of 5 folds). Same conclusion as the 3-seed table: no gain, possibly a small loss.
* The control re-runs were done because all three 384 px runs of 2026-10-12 came out low (0.925–0.929) while both earlier ones were high (0.950–0.955).
  A session effect is unlikely: the 224 px re-runs made in the same session, on the same server and library versions (torch 2.11, 2× T4), came out *higher* than the day before.
* The re-runs are **not** included in the seed-averaged and paired tables above (those use exactly one run per seed, as planned), to avoid choosing runs after seeing them.
* Practical rule for the paper: a difference between two configurations below ≈ 0.02 macro-F1 on a single run is not evidence of anything. Only raw → pipeline
  for DenseNet121 (−0.021 over 3 seeds, all 5 folds, Holm p = 0.002) clears this bar among the comparisons tested.

To reproduce the 384 px seeds: `python train_cv.py --seed 1 --out <dir>/s1 --versions raw384 --models densenet121` (and `--seed 2`; ≈ 30 min each on one T4), then `analyze_seeds.py`.

Files: `backup/results_seeds/s1`, `s2` (35 jobs each), `backup/results_seeds/s0_rerun384`, `s1_rerun224`, `s2_rerun224` (5 control jobs each),
`backup/report_seeds/` (`summary.csv`, `comparisons.csv`, `per_run.csv`, `replicates_densenet121.csv`), `dental_cls/analyze_seeds.py`.
