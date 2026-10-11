# Comparison with Lashaki et al., BMC Oral Health 2025;25:535 (doi:10.1186/s12903-025-05704-0)

| Aspect | Lashaki et al. 2025 | This work |
|---|---|---|
| Data | Roboflow "dental-implant-nyw2h": 511 periapical images (275 Bicon, 70 Bego, 166 ITI) | Roboflow ToothFairy: 1,011 source radiographs, 3,276 implant crops (1,879 Bicon / 1,226 ITI / 171 Bego) |
| Task | **Pairwise binary** only (Bicon–Bego, ITI–Bego, Bicon–ITI) | **3-class** brand classification (all brands at once) + detection |
| Augmentation / balance | 511 expanded to 5,110 images; not stated that augmentation was restricted to training folds (Discussion instead says "10 images per group" – inconsistent) | Augmentation on-the-fly inside training folds only; class-weighted loss instead of synthetic copies |
| Split | 5-fold CV; no grouping by source image / patient described | Group-stratified 5-fold CV by source radiograph, near-duplicates merged (dHash); one cross-split duplicate found and removed |
| Input size | 128 × 128 | 224 and 384 px crops |
| Preprocessing | Sharpen, median denoise, CLAHE, **implant mask**, histogram equalisation; reported as improving performance | Same family of steps tested against raw with ablation; **no gain**; strong sharpening harmful; resolution matters more |
| Models | CNN, CNN+SVM/DT/RF, frozen VGG16 / ResNet50 / Xception | Custom CNN, fine-tuned VGG16, ResNet50, DenseNet121, EfficientNet-B0, ViT-B/16; YOLO11n and RF-DETR detectors |
| Statistics | Mean of 5 folds; no significance tests, no CIs | Mean ± SD, bootstrap 95% CIs, paired t / Wilcoxon / McNemar with Holm correction |
| Rare class | Bego is in every task but its per-class result is not reported in a 3-class setting | Bego recall reported per model (best 0.950) |
| Explainability | none | Grad-CAM for all six classifiers |
| Best result | 0.9865 accuracy (VGG16, Bicon vs Bego, binary) | 3-class: accuracy 0.982, macro-F1 0.955, balanced accuracy 0.972 (DenseNet121, raw 384 px) |

## New findings relative to the paper
1. **3-class classification at the level of their best binary result.** A single 3-class model reaches 0.98 accuracy and 0.955 macro-F1, comparable to their best *binary* accuracy (0.9865) on an easier task.
2. **Preprocessing does not improve classification when tested properly.** Contrary to the paper's central claim, a sharpen/CLAHE/ROI pipeline is never better than raw crops and is significantly worse for DenseNet121 (McNemar Holm p = 0.005) and ViT-B/16. Ablation pins most of the loss on strong sharpening; a gentler pipeline only matches raw.
3. **Ground-truth-based masking/ROI processing leaks the answer.** When the preprocessing region comes from the annotation (as an "implant-specific mask" does), the processed image encodes the label location. We measured this: YOLO11n mAP50-95 rises from 0.45 to 0.80 with GT-ROI preprocessing while brand accuracy and mAP50 do not improve. Masks derived from annotations cannot be used at inference without a detector/segmenter, and inflate metrics.
4. **Resolution beats enhancement.** 384 px crops gave the best macro-F1 and Bego recall (0.95); HAT super-resolution gave no benefit at 224 or 384 px.
5. **Leakage-aware evaluation.** Group-by-source splitting with near-duplicate merging; augmentation only inside training folds.
6. **Detection on the same folds** (YOLO11n, RF-DETR) with brand accuracy derived from detections, so detectors and classifiers are compared directly.

## Caveats
* Different Roboflow datasets (possibly overlapping sources), so absolute numbers are not directly comparable; the methodological differences are the main contribution.
* Our "raw" images are the HAT images area-downsampled to 640 px (proxy for the original Roboflow export).
* The paper does not state whether augmented copies were kept within training folds; claims about leakage there should be worded as "not reported" / "risk", not as fact.
