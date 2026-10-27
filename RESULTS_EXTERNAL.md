# External datasets – pseudo-labelled candidates as extra training data

## Candidates (automatic, not reviewed by a clinician)
Pipeline `match_oii.py`: 6 classifiers (DenseNet121 + ViT-B/16, 3 seeds, fold-0 checkpoints) score each external implant crop; a crop is a candidate for a brand
when all 6 agree, the mean probability reaches that brand's 10th percentile of correct predictions on our test crops, ≥ 8 of 10 nearest neighbours
(in both feature spaces) are that brand, and its nearest-neighbour similarity is within our own test distribution. On our unseen fold-0 test crops this rule
kept 270 of 317 with 100 % precision.

| dataset | image type | implant crops | in-distribution | candidates ITI / Bego / Bicon | used |
|---|---|---|---|---|---|
| OII-DS (figshare 22608790, CC BY 4.0) | mostly CT views | 6,616 | 467 | 170 / 16 / 0 | 158 ITI (12 rejected on sight: CT slices with crosshairs, two-implant crops) + 16 Bego |
| Zenodo 15487430 (CC BY 4.0) | panoramic | 522 (hand boxes) | 6 | 4 / 1 / 0 | 4 ITI + 1 Bego |
| Mendeley x4gr6mmwy4 | panoramic | 26 (our detectors, 65 patients) | 0 | 0 / 0 / 0 | – (most "endosteal" images show no implant) |

**Added: 162 ITI + 17 Bego = 179 crops**, to the training set of every fold; validation and test unchanged (same 1,590 test implants, 76 Bego).
Original-image crops (`raw_orig`), DenseNet121 and ViT-B/16, 3 seeds × 5 folds. Baseline = the `raw_orig` runs of `RESULTS_ORIGINAL.md`.

**Selection caveat:** candidates were chosen by similarity to fold-0 *training* crops, which are test crops in folds 1–4, so folds 1–4 could be slightly
favoured. Fold 0's test crops played no part in the selection: fold 0 alone is the leak-free check.

## Results (mean of 15 runs)

| model | set | macro-F1 | accuracy | Bego recall | Bego F1 | ITI recall | ITI F1 |
|---|---|---|---|---|---|---|---|
| DenseNet121 | baseline | 0.936 | 0.976 | 0.935 | 0.847 | 0.967 | 0.972 |
| DenseNet121 | + external | 0.935 | 0.978 | 0.896 | 0.840 | 0.965 | 0.971 |
| ViT-B/16 | baseline | **0.954** | **0.986** | **0.915** | **0.884** | **0.981** | **0.983** |
| ViT-B/16 | + external | 0.945 | 0.983 | 0.882 | 0.862 | 0.975 | 0.977 |

Paired differences (Δ = with − without):

| model | metric | Δ (per seed) | folds better | paired t | fold 0 only, Δ (per seed) |
|---|---|---|---|---|---|
| DenseNet121 | macro-F1 | −0.001 (+0.012 / −0.013 / −0.002) | 3 of 5 | p = 0.88 | +0.008 (+0.056 / −0.010 / −0.023) |
| DenseNet121 | Bego recall | −0.039 (−0.024 / −0.044 / −0.047) | 0 of 5 | p = 0.18 | 0.000 |
| ViT-B/16 | macro-F1 | −0.010 (+0.001 / +0.001 / −0.031) | 2 of 5 | p = 0.34 | **−0.043 (−0.045 / −0.044 / −0.041)** |
| ViT-B/16 | Bego recall | −0.033 (+0.050 / −0.026 / −0.123) | 1 of 5 | p = 0.23 | −0.067 (0.000 / −0.067 / −0.133) |
| ViT-B/16 | ITI recall | −0.006 | 1 of 5 | p = 0.43 | −0.031 |

3-seed ensembles, all folds: DenseNet121 macro-F1 0.953 → 0.950 (McNemar 7 vs 10, p = 0.63); **ViT-B/16 0.972 → 0.959, Bego recall 0.961 → 0.934
(McNemar 10 vs 2, p = 0.039 – significantly worse)**. Fold 0 alone: ViT-B/16 0.961 → 0.923.

## Reading
1. **The external candidates do not help; they make the best model worse.** ViT-B/16 loses 0.010 macro-F1 overall and 0.043 on the leak-free fold 0
   (lower in all 3 seeds), its seed ensemble is significantly worse (p = 0.039), and Bego recall drops for both models. DenseNet121 is unchanged.
2. Even ITI recall does not improve although 162 ITI-like crops were added. The candidates are from a different image type (CT / panoramic, low resolution,
   no 15 % context margin) and their brand is unverified; many may be other brands that resemble ITI.
3. Together with `RESULTS_EXTRA_BEGO.md`: none of the extra-data experiments (user Bego periapicals, panoramics, public-dataset candidates) improved the best model.
   Recommendation: keep the benchmark on the original 1,011 radiographs. If external data is used at all, only clinician-confirmed periapical images.

Files on Hugging Face: `results_external/s0–s2` (30 jobs), `report_external/` (`comparison.csv`, `per_run.csv`), `oii_match/`, `zenodo_match/`, `mendeley_match/`.
Scripts: `match_oii.py`, `crops_from_yolo.py`, `detect_crops.py`, `review_sheets.py`, `build_ext.py`.
