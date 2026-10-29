# Leakage-free benchmark: dental implant brand identification (Bego / Bicon / ITI)

A fixed, patient-level evaluation protocol for the public Roboflow dataset
[`dental-implant-nyw2h/-dental-implant`](https://universe.roboflow.com/dental-implant-nyw2h/-dental-implant) (version 1, CC BY 4.0),
the dataset used by several 2023–2025 studies. Published results on it range from 98 to 99 %, but most splits let copies of the same
radiograph appear in both training and test. This benchmark removes that leakage so results can be compared fairly.

**Data on Hugging Face:** [emad2001/dental-implant-brand-benchmark](https://huggingface.co/datasets/emad2001/dental-implant-brand-benchmark) (crops as PNG + metadata, folds, baselines).

## The data
* 2,064 periapical radiographs (640 × 640) from **1,011 source radiographs**; Roboflow stores 1–3 augmented copies of many radiographs.
* 3,276 implant crops: **Bego 171, Bicon 1,879, ITI 1,226**. Each crop = the annotated box + 15 % margin per side, letterboxed to 224 × 224, grey scale.
* Near-duplicate radiographs (dHash Hamming distance ≤ 12 of 256 bits) were merged into one group: **1,009 source groups**.

## The protocol
* **5 folds, grouped by source radiograph** (`benchmark/folds.json`, rep 0; scikit-learn `StratifiedGroupKFold`, stratified by brand).
  All copies and near-duplicates of a radiograph are in the same fold. Reps 1 and 2 are two further fold splits for repeated CV.
* Fold k: train on `train`, select epochs / hyper-parameters on `val` (primary crops of ~1/6 of the training groups), test on `test`.
* **Test set = one primary crop per implant of each test radiograph** (the first copy), so every one of the **1,590 test implants (76 Bego, 934 Bicon,
  580 ITI)** is tested exactly once over the 5 folds. Augmented copies are only ever used for training.
* Report macro-F1 (primary – Bego is rare), balanced accuracy, Bego recall, accuracy; mean ± SD over folds, and ideally ≥ 3 training seeds:
  training noise alone moves a 5-fold mean macro-F1 by ~0.01–0.02.
* Do not use the annotation boxes at test time for anything except cropping (a box-derived mask or ROI preprocessing leaks the label location).

## Files (`benchmark/`)
| file | content |
|---|---|
| `crops.csv` | one row per crop: `idx`, `brand` (0 Bego, 1 Bicon, 2 ITI), `brand_name`, `source_group`, `primary`, `test_fold`, `original_file` (Roboflow v1 file name), `roboflow_split`, `box_idx`, box `cx, cy, w, h` (normalised, full precision) |
| `folds.json` | the folds: `folds[i]` with `rep`, `fold`, `train`, `val`, `test` (crop indices). Use `rep` 0. |
| `build_crops.py` | rebuilds all crops from the Roboflow v1 export (reproduces the published crops; 4 crops may differ by one grey level in one pixel between OpenCV versions) |
| `evaluate.py` | scores a predictions CSV (`idx, p_Bego, p_Bicon, p_ITI` or `idx, pred`) per fold and pooled; `--compare` gives an exact McNemar test |
| `baselines/` | out-of-fold probabilities of our DenseNet121 and ViT-B/16 (3 seeds each + 3-seed ensembles) in the evaluation format |

```bash
python benchmark/evaluate.py my_predictions.csv --compare benchmark/baselines/vit_b_16_seed0.csv
```

## Results on this benchmark (grouped folds, 1,590 test implants)

| method | input | accuracy | macro-F1 | Bego recall | notes |
|---|---|---|---|---|---|
| **ViT-B/16, 3-seed ensemble** (ours) | crops 224 | **0.991** | **0.972** | **0.961** | probabilities averaged over 3 seeds |
| **ViT-B/16** (ours) | crops 224 | 0.986 | 0.954 | 0.915 | mean of 3 seeds × 5 folds |
| DenseNet121 (ours) | crops 224 | 0.976 | 0.936 | 0.935 | mean of 3 seeds × 5 folds |
| ViT-B/16, deferring the 5 % least confident | crops 224 | 0.996 | 0.986 | 0.986 | 3-seed ensemble, 95 % coverage |
| Tiryaki et al. 2023 pipeline, best single model (VGG-16) | tight crops 224 | 0.963 | 0.927 | 0.812 | SGD, no augmentation, 50 epochs |
| Tiryaki et al. 2023 pipeline, 5-model majority vote | tight crops 224 | 0.966 | 0.916 | 0.760 | |
| Lashaki et al. 2025 pipeline (VGG-16, preprocessing + box mask) | whole image 128 | 0.841 | 0.753 | 0.488 | image level (1,009 test images); uses ground-truth boxes for the mask |

Detection with the same folds (implant found and brand correct; `RESULTS_DETECTION.md`): YOLO11n mAP50 0.887, brand macro-F1 0.905;
RF-DETR Small mAP50 0.900, brand macro-F1 0.917; detector → ViT/DenseNet two-stage macro-F1 0.948–0.956.

**Same pipelines with a leaky split** (for reference, `RESULTS_PROTOCOLS.md`): Tiryaki 98.4 % / 0.960, Lashaki 3-brand 89.8 % / 0.874
(Bicon vs Bego 99.3 %), Tekis YOLO detection F1 0.956 → 0.857 grouped.

## Citation
See `CITATION.cff`. Please also credit the dataset: Roboflow Universe, "Dental implant" (dental-implant-nyw2h), CC BY 4.0.
