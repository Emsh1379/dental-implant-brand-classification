# Extra Bego training data – does it help?

## Data added (training only)
| set | source | Bego crops |
|---|---|---|
| **plus** | 18 images of 3 Bego radiographs from `lamiaa/implant-recognition` v3 (`bego_extra/`) + 16 implants in 10 user-supplied periapical radiographs (`bego_user/`) | +34 |
| **plus2** | plus + 26 implants in 3 user-supplied panoramics (`bego_user_pano/`: one panoramic, one panoramic screenshot, one phone photo of a panoramic film) | +60 |

Bego crops in the whole set: 171 → 205 (plus) → 231 (plus2). All boxes are auto-labels (5-model YOLO11n consensus – on 640 px tiles for the panoramics –
snapped to the implant metal), checked visually; 3 boxes were placed by hand where the detectors found nothing. Not used: phone photos of films with
heavy colour cast and social-media collages with text overlays, and a journal figure (copyrighted, brand not verifiable, poor boxes).
The detectors called most of the user's periapical Bego implants ITI (majority ITI on 12 of 14): they look like a different Bego implant line from the dataset's Bego.

The new crops are added to the **training set of every fold**; validation and test sets are unchanged, so all runs are scored on the same 1,590 test implants
(76 Bego). Input: original-image crops (`raw_orig`), 224 px; DenseNet121 and ViT-B/16, 3 seeds × 5 folds. Baseline = the `raw_orig` runs of `RESULTS_ORIGINAL.md`.

## Results (mean of 15 runs)

| model | training set | macro-F1 | accuracy | Bego recall | Bego F1 |
|---|---|---|---|---|---|
| DenseNet121 | baseline | 0.936 | 0.976 | 0.935 | 0.847 |
| DenseNet121 | plus (+34 periapical) | **0.952** | 0.982 | **0.950** | **0.887** |
| DenseNet121 | plus2 (+34 periapical +26 panoramic) | 0.946 | 0.981 | 0.898 | 0.869 |
| ViT-B/16 | baseline | **0.954** | **0.986** | **0.915** | **0.884** |
| ViT-B/16 | plus | 0.947 | 0.982 | 0.905 | 0.871 |
| ViT-B/16 | plus2 | 0.946 | 0.982 | 0.892 | 0.869 |

Paired differences (same folds; difference averaged over seeds per fold):

| model | comparison | Δ macro-F1 (per seed) | folds better | p | Δ Bego recall (per seed) | folds better | p |
|---|---|---|---|---|---|---|---|
| DenseNet121 | baseline → plus | +0.015 (+0.009 / +0.020 / +0.017) | 5 of 5 | 0.047 | +0.015 (−0.013 / +0.037 / +0.022) | 3 of 5 | 0.26 |
| DenseNet121 | plus → plus2 | −0.006 (−0.003 / +0.002 / −0.018) | 1 of 5 | 0.25 | **−0.052 (−0.048 / −0.035 / −0.072)** | **0 of 5** | 0.057 |
| DenseNet121 | baseline → plus2 | +0.009 (+0.007 / +0.022 / −0.001) | 4 of 5 | 0.20 | −0.036 (−0.062 / +0.002 / −0.050) | 1 of 5 | 0.092 |
| ViT-B/16 | baseline → plus | −0.008 (−0.009 / +0.010 / −0.024) | 2 of 5 | 0.30 | −0.010 (−0.001 / +0.042 / −0.072) | 2 of 5 | 0.74 |
| ViT-B/16 | plus → plus2 | −0.001 (−0.007 / −0.004 / +0.010) | 2 of 5 | 0.94 | −0.013 (−0.055 / −0.018 / +0.035) | 1 of 5 | 0.69 |
| ViT-B/16 | baseline → plus2 | −0.008 (−0.016 / +0.005 / −0.014) | 2 of 5 | 0.27 | −0.023 (−0.056 / +0.024 / −0.036) | 1 of 5 | 0.23 |

3-seed ensembles (probabilities averaged per crop), baseline → plus → plus2:
DenseNet121 macro-F1 0.953 → 0.962 → 0.959, Bego recall 0.934 → 0.947 → 0.921; ViT-B/16 macro-F1 0.972 → 0.967 → 0.967, Bego recall 0.961 → 0.934 → 0.934.
No ensemble McNemar test is significant (p ≥ 0.23).

(The ViT-B/16 "plus" row moved slightly from the first report, 0.947 / Bego recall 0.901 → 0.947 / 0.905, because one job ran on both GPUs and the later copy replaced the first.)

## Reading
1. **Panoramic Bego crops do not help and probably hurt Bego recall.** Adding them lowers DenseNet121 Bego recall by 0.05 versus the periapical-only extra set
   (lower in all 3 seeds and all 5 folds, p = 0.057) and leaves ViT-B/16 unchanged. Panoramic implants are small and blurry, a different image type from the
   periapical test images, so they add little usable Bego information.
2. **Periapical extras give DenseNet121 a likely small gain** (+0.015 macro-F1, all seeds and folds, mostly Bego precision); this is the size of the run-to-run noise.
3. **ViT-B/16 – the best model – gains nothing from either extra set**; its best result is still without extra data (macro-F1 0.954, ensemble 0.972 / Bego recall 0.961).
4. Conclusion: auto-labelled, internet-sourced Bego images (34–60 crops, mostly of a different-looking Bego line, partly panoramic) do not reliably improve
   Bego recognition. Keep them out of the main benchmark; report this as a negative add-on result if at all. What would help is more *periapical* radiographs of the
   same Bego implant type as in the dataset, with clinician-confirmed labels.

Files on Hugging Face: `results_trainplus/` and `results_trainplus2/` (30 jobs each), `report_trainplus/`, `report_trainplus2/` (`comparison.csv`, `per_run.csv`),
`crops_trainplus/`, `crops_trainplus2/` (meta, folds; `raw_orig.npy` = `crops/raw_orig.npy` + the extra rows), `bego_extra/`, `bego_user/`, `bego_user_pano/`.
Scripts: `label_new.py`, `label_pano.py`, `build_trainplus.py`; train with `train_cv.py --crops <crops_trainplus2> --versions raw_orig --models densenet121 vit_b_16 --seed S`.
