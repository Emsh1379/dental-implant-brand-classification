# Clinical reliability: unknown brands, calibration and deferral

All published models on this dataset always answer Bego, Bicon or ITI. In practice a radiograph can show any of hundreds of implant systems, and a
dentist needs to know when the answer is uncertain.

## 1. Open-set rejection – "not one of these three brands" (`openset.py`)
* Models: DenseNet121 and ViT-B/16 trained on original crops, one per fold (seed 0).
* In-distribution (ID): each fold's real test crops (1,590), scored by the fold model that never saw them.
* Unknown implants (OOD): 6,616 implant crops from OII-DS (mostly CT views) and 522 implant crops from the Zenodo panoramic dataset. Brands are unknown
  and mostly other systems; some may be ITI, Bego or Bicon, so the accept rates below are upper bounds.
* Rejection scores: maximum softmax probability (MSP), energy (log-sum-exp of logits), and kNN = cosine similarity of the image's feature vector to the
  nearest training crop. The threshold keeps 95 % of real test implants (chosen on ID data only).

| model | rejection score | AUROC vs OII-DS | OII-DS accepted at 95 % ID | AUROC vs Zenodo | Zenodo accepted | accuracy on the accepted 95 % ID |
|---|---|---|---|---|---|---|
| ViT-B/16 | softmax confidence (MSP) | 0.589 | 80 % | 0.799 | 48 % | 0.985 |
| ViT-B/16 | energy | 0.420 | 94 % | 0.729 | 79 % | 0.980 |
| ViT-B/16 | **kNN feature similarity** | **0.945** | **32 %** | **0.977** | **12 %** | **0.989** |
| DenseNet121 | softmax confidence (MSP) | 0.751 | 65 % | 0.862 | 43 % | 0.990 |
| DenseNet121 | energy | 0.658 | 91 % | 0.838 | 77 % | 0.982 |
| DenseNet121 | **kNN feature similarity** | **0.952** | **27 %** | **0.987** | **5 %** | **0.987** |

* **The classifier's own confidence does not reveal unknown implants.** With MSP or energy, 65–94 % of the OII-DS implants are still accepted and given
  a brand; ViT-B/16 calls 74 % of them "ITI" and 22 % "Bego". A three-brand model used as is would confidently mislabel most foreign implants.
* **Distance to the training images works:** kNN similarity in the model's feature space reaches AUROC 0.95–0.99 and cuts accepted unknown implants to
  27–32 % (OII-DS) and 5–12 % (Zenodo), while the accepted real implants keep 98.7–98.9 % accuracy. It needs no retraining – only the stored training features.
* Limits: the unknown sets are mostly other image types (CT, panoramic), so they are easier than other-brand periapicals would be; a periapical
  other-brand test set is the next step.

## 2. Calibration (`calibration.py`; DenseNet121 / ViT-B/16 on original crops, 3 seeds, 5 folds, pooled out-of-fold predictions)

| model | run | ECE raw | ECE after temperature scaling | Brier raw → scaled | NLL raw → scaled | mean confidence raw → scaled (accuracy) |
|---|---|---|---|---|---|---|
| DenseNet121 | 3-seed ensemble | 0.253 | **0.012** | 0.152 → 0.030 | 0.339 → 0.096 | 0.73 → 0.98 (0.982) |
| ViT-B/16 | 3-seed ensemble | 0.264 | **0.011** | 0.147 → 0.016 | 0.333 → 0.083 | 0.73 → 0.98 (0.991) |
| both | single seeds | 0.240–0.259 | 0.011–0.019 | | | |

* The raw probabilities are strongly **under-confident** (mean confidence 0.73 at 98–99 % accuracy) – a side effect of label smoothing and class weights.
* Temperature scaling (T ≈ 0.23–0.31), fitted on the other folds only (cross-fitted, no crop calibrated with its own label), fixes it: ECE ≈ 0.01.
  Reliability diagrams and risk-coverage curves: `report_calibration/calibration.png`.

## 3. Selective prediction – let the model defer its least certain cases

| model (3-seed ensemble) | coverage | accuracy | macro-F1 | Bego recall | errors removed |
|---|---|---|---|---|---|
| ViT-B/16 | 100 % | 0.991 | 0.972 | 0.961 | – |
| ViT-B/16 | **95 %** | **0.996** | **0.986** | **0.986** | **57 %** |
| ViT-B/16 | 90 % | 0.996 | 0.986 | 0.986 | 57 % |
| DenseNet121 | 100 % | 0.982 | 0.953 | 0.934 | – |
| DenseNet121 | 95 % | 0.993 | 0.975 | 0.971 | 61 % |

* Deferring the 5 % least confident implants to a dentist removes 57–61 % of all errors and lifts Bego recall to 0.97–0.99. AURC 0.012 (ViT-B/16) and 0.013 (DenseNet121).
* Single seeds: 29–65 % of errors removed at 95 % coverage.

## What this adds to the paper
A clinically usable system needs three things no published work on this dataset reports: (1) a way to say "unknown brand" – softmax confidence
is not enough, feature-distance rejection is; (2) calibrated probabilities – temperature scaling turns the under-confident outputs into reliable ones;
(3) an abstain option – deferring 5 % of cases removes more than half of the errors. All three come at no extra training cost.

Files on Hugging Face: `protocols/openset/` (openset.csv, all scores), `report_calibration/` (calibration.csv, selective.csv, calibration.png).
Scripts: `openset.py`, `calibration.py`.
