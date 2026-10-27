# Original Roboflow images – does "raw" have to be the real raw?

Until now "raw" meant the HAT super-resolved image area-downsampled back to 640 px, a proxy for the original export. This experiment uses the
**original images**: Roboflow project `dental-implant-nyw2h/-dental-implant`, **version 1** (2,064 images, 640 × 640 stretch, no augmentation, CC BY 4.0;
universe.roboflow.com/dental-implant-nyw2h/-dental-implant). Version 2 of the same project adds augmented training copies (2,507 images) and was not used.

## Data check
* Version 1 is the same dataset as the HAT set: 2,064 images from 1,011 source images, identical train / valid / test counts (1,754 / 201 / 109),
  identical number of boxes per image and identical brand for every box. Only the Roboflow file hashes differ.
* Each of our images was matched to its original inside the same (source image, split) group by crop correlation, with one-to-one assignment
  (`dental_cls/prep_orig.py`). All 2,064 images matched; every crop correlates ≥ 0.947 with its proxy crop (median 0.995).
* Crops use **the same boxes** (the HAT set's label files) and the same folds, so only the pixels differ. Two new inputs:
  `raw_orig` (original pixels) and `pipeline_orig` (the sharpen / CLAHE / unsharp pipeline applied to the original 640 px image).
* The proxy was not quite raw: it is sharper (Laplacian variance 26.5 vs 22.1 on the original crops), i.e. it keeps some HAT detail.
* One label problem found: crop 2882 (`backup/crops/meta.csv` idx) is a box with no implant in it (IoU 0.06 with the box in the original export). Add it to the label-review list.
* The Roboflow project `lamiaa/implant-recognition` (cited by Tekis et al., Sci Rep 2025) is a different, smaller export of the same source images
  (515 of our 1,011 source images, 3–9 augmented copies each, 233 source images in more than one of its train / valid / test splits). It adds no new radiographs.

## Results – DenseNet121 and ViT-B/16, 3 seeds × 5 folds = 15 runs each (224 px)

| model | input | macro-F1, mean of 15 | per seed (0 / 1 / 2) | accuracy | Bego recall | 3-seed ensemble macro-F1 / Bego recall |
|---|---|---|---|---|---|---|
| DenseNet121 | **original** | 0.936 | 0.936 / 0.927 / 0.947 | 0.976 | 0.935 | 0.953 / 0.934 |
| DenseNet121 | pipeline on original | 0.929 | 0.937 / 0.942 / 0.907 | 0.974 | 0.897 | 0.947 / 0.921 |
| DenseNet121 | proxy raw (HAT ↓640) | 0.944 | 0.950 / 0.942 / 0.941 | 0.981 | 0.908 | 0.963 / 0.934 |
| DenseNet121 | HAT | 0.945 | 0.945 / 0.948 / 0.942 | 0.980 | 0.933 | 0.952 / 0.921 |
| ViT-B/16 | **original** | **0.954** | 0.952 / 0.940 / 0.971 | **0.986** | 0.915 | **0.972 / 0.961** |
| ViT-B/16 | pipeline on original | 0.923 | 0.930 / 0.906 / 0.933 | 0.974 | 0.872 | 0.939 / 0.895 |
| ViT-B/16 | proxy raw (HAT ↓640) | 0.951 | 0.950 / 0.952 / 0.951 | 0.984 | 0.912 | 0.954 / 0.921 |
| ViT-B/16 | HAT | 0.951 | 0.954 / 0.944 / 0.954 | 0.982 | 0.927 | 0.967 / 0.961 |

The proxy-raw, pipeline and HAT rows are the runs from `RESULTS_SEEDS.md` (same folds, seeds 0–2).

## Paired comparisons (difference averaged over seeds within each fold; Holm over the 8 ensemble McNemar tests)

| model | comparison | Δ macro-F1 | per seed | folds where second is better | paired t (5 folds) | McNemar, 3-seed ensemble (Holm) |
|---|---|---|---|---|---|---|
| DenseNet121 | original → pipeline on original | −0.008 | +0.001 / +0.016 / −0.040 | 3 of 5 | p = 0.44 | 14 vs 9, p = 0.40 (1.00) |
| DenseNet121 | original → proxy raw | +0.008 | +0.014 / +0.015 / −0.006 | 3 of 5 | p = 0.31 | 6 vs 14, p = 0.12 (0.69) |
| DenseNet121 | original → HAT | +0.009 | +0.010 / +0.021 / −0.005 | 4 of 5 | p = 0.21 | 10 vs 9, p = 1.00 (1.00) |
| ViT-B/16 | original → pipeline on original | **−0.031** | −0.022 / −0.034 / −0.038 | **0 of 5** | p = 0.061 | **22 vs 3, p = 0.0002 (0.001)** |
| ViT-B/16 | original → proxy raw | −0.004 | −0.003 / +0.012 / −0.021 | 2 of 5 | p = 0.60 | 14 vs 4, p = 0.031 (0.22) |
| ViT-B/16 | original → HAT | −0.004 | +0.001 / +0.004 / −0.017 | 2 of 5 | p = 0.65 | 9 vs 5, p = 0.42 (1.00) |

## What changes for the paper
1. **The "raw is only a proxy" limitation can be removed.** Results on the original images agree with the proxy within training noise
   (DenseNet121 −0.008, ViT-B/16 +0.004, both not significant). Report the original images as the raw baseline from now on.
2. **Super-resolution gives no benefit over the real originals** (DenseNet121 +0.009, ViT-B/16 −0.004, not significant). This is now a direct
   test of HAT against the images it was made from, not against a HAT-derived proxy.
3. **The enhancement pipeline does not help, and significantly hurts ViT-B/16 on the original images**: −0.031 macro-F1, lower in all 3 seeds and all 5 folds,
   ensemble McNemar Holm p = 0.001, Bego recall 0.915 → 0.872. For DenseNet121 on the originals the loss is small and not significant (−0.008; one seed −0.040).
   Together with `RESULTS_SEEDS.md`: across 2 models × 2 raw sources × 3 seeds the pipeline is below raw in 10 of 12 runs and significantly worse in one model per raw source
   (DenseNet121 on the proxy, ViT-B/16 on the originals). Write it as "never better than raw; significantly worse in some settings; the size of the loss depends on the model".
4. **Best single model: ViT-B/16 on original crops**, macro-F1 0.954, accuracy 0.986 (3-seed mean); its 3-seed ensemble reaches 0.972 / Bego recall 0.961 –
   the highest numbers in the project. Differences between architectures and raw sources stay within noise (≈ 0.02).
5. Error pattern unchanged: missed Bego go to ITI (never to Bicon); the most frequent single error is ITI predicted as Bego.

Runs were spread over three Kaggle sessions (two ended mid-run; finished jobs were kept and the rest resumed). Since a same-seed re-run already varies by
≈ 0.016 (`RESULTS_SEEDS.md`), this adds no noise beyond what the seed columns show.

Files: `dental_cls/prep_orig.py` (download, matching, crops), `dental_cls/analyze_seeds.py --set orig`, `backup/results_orig/s0–s2` (60 jobs),
`backup/report_orig/` (`summary.csv`, `comparisons.csv`, `per_run.csv`). Crops on Hugging Face: `crops/raw_orig.npy`, `raw_orig384.npy`, `pipeline_orig.npy`, `raw_orig_match.csv`.
To reproduce: `ROBOFLOW_API_KEY=… python prep_orig.py`, then
`python train_cv.py --seed S --out <dir>/sS --versions raw_orig pipeline_orig --models densenet121 vit_b_16` for S = 0, 1, 2.
