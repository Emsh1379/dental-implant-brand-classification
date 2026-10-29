# Synthetic Bego crops from a class-conditional diffusion model (after ThirdMolarSynthesis)

Source method: github.com/s4nyam/ThirdMolarSynthesis, `3m-cdiff` (Jain et al., third-molar ROI synthesis with conditional DDPM/DDIM).
No trained weights are released, so the model was trained from scratch on our crops.

## Generator
* Kept from the repository: class-conditional diffusers U-Net at 128 × 128, DDPM with 1,000 steps, `squaredcos_cap_v2` schedule, ε-prediction with MSE.
* Changed:
  * the repository's `UNet2DConditionModel` uses `AttnDownBlock2D`/`AttnUpBlock2D`, which have no cross-attention, so its class context reaches the network
    only through the mid block (checked: changing the label changes the output by at most 0.035);
  * here the brand is added to the timestep embedding at every level (`UNet2DModel`, `num_class_embeds`), 4 levels (32–256 channels),
    attention only at 16 px (31 s instead of 176 s per epoch on a T4);
  * 1 grey channel, classifier-free guidance (label dropped 10 %, guidance 2.0), batch 32, AdamW 2e-4, fp16, EMA 0.999, horizontal flips, 120 epochs;
  * sampling with DDIM, 100 steps.
* **One generator per fold, trained on that fold's training crops only**, so synthetic images never derive from test implants.
* 400 synthetic Bego per fold (+32 Bicon, +32 ITI for the check), upsampled to 224 px. Script: `dental_cls/cdiff_brand.py`.

## Quality check (`qc_syn.py`)
| fold | synthetic Bego called Bego* | real test Bego called Bego* | synthetic ITI called ITI* | closest-real-crop correlation, synthetic Bego (median / max) | same for real test Bego (median) |
|---|---|---|---|---|---|
| 0 | 0.58 | 0.87 | 0.41 | 0.80 / 0.96 | 0.86 |
| 1 | 0.64 | 1.00 | 0.31 | 0.79 / 0.95 | 0.87 |
| 2 | 0.81 | 0.94 | 0.41 | 0.78 / 0.94 | 0.88 |
| 3 | 0.50 | 1.00 | 0.41 | 0.75 / 0.92 | 0.88 |
| 4 | 0.54 | 0.94 | 0.16 | 0.80 / 0.94 | 0.85 |

\*by the fold's DenseNet121 (raw 384 px) trained on real crops of the same training set only.

* **Brand fidelity is moderate:** only 50–81 % of synthetic Bego look like Bego to a real-data classifier (real Bego: 87–100 %); synthetic ITI is mostly not
  recognised as ITI. The generator separates brands only partly with ~120–130 real Bego per fold.
* **No memorisation:** synthetic crops are less similar to their nearest training crop than real unseen test crops are; the closest pairs are different implants.
* **Visual quality is mixed:** implant-shaped bodies at varied angles with realistic letterbox padding, but frequent noise, blur, speckle artefacts and little thread detail.

## Retraining (synthetic Bego in the training set only; same 1,590 test implants; 3 seeds × 5 folds)

| model | metric | baseline | + synthetic Bego | Δ (per seed) | folds better | paired t |
|---|---|---|---|---|---|---|
| DenseNet121 | macro-F1 | 0.936 | 0.941 | +0.004 (+0.008 / +0.012 / −0.008) | 3 of 5 | p = 0.74 |
| DenseNet121 | **Bego recall** | 0.935 | **0.820** | **−0.115 (−0.110 / −0.116 / −0.117)** | **0 of 5** | **p = 0.035** |
| DenseNet121 | Bego F1 | 0.847 | 0.854 | +0.007 | 3 of 5 | p = 0.84 |
| DenseNet121 | ITI recall | 0.967 | 0.980 | +0.014 | 5 of 5 | p = 0.029 |
| ViT-B/16 | macro-F1 | 0.954 | 0.942 | −0.013 (−0.020 / +0.003 / −0.021) | 1 of 5 | p = 0.28 |
| ViT-B/16 | Bego recall | 0.915 | 0.855 | −0.060 (−0.082 / −0.027 / −0.071) | 1 of 5 | p = 0.24 |
| ViT-B/16 | Bego F1 | 0.884 | 0.855 | −0.029 | 2 of 5 | p = 0.36 |

3-seed ensembles: DenseNet121 macro-F1 0.953 → 0.952, Bego recall 0.934 → 0.855 (Bego→ITI errors 5 → 11, ITI→Bego 12 → 6);
ViT-B/16 0.972 → 0.963, Bego recall 0.961 → 0.895. No ensemble McNemar test is significant.

## Reading
1. **Synthetic Bego did not help and lowered Bego recall** – the opposite of the aim. With 400 extra "Bego" per fold the models find fewer real Bego
   (DenseNet121 −0.115, lower in every seed and fold, p = 0.035; ViT-B/16 −0.060) and fewer ITI are called Bego. The likely reason: the synthetic Bego
   differ from real Bego (noisy, blurred, only partly brand-specific), so the classifiers learn "synthetic look = Bego" and then call fewer real implants Bego.
   Macro-F1 is unchanged for DenseNet121 (+0.004) and slightly lower for ViT-B/16 (−0.013), both not significant.
2. The best model is still ViT-B/16 on real original crops only (0.954 / Bego recall 0.915).
3. A generator that could help would need clearly better samples (more Bego data, pretrained latent diffusion fine-tuned on X-rays, longer training,
   filtering synthetic images by a real-data classifier). With ~120 real Bego per fold, the data that limits the classifier also limits the generator.
4. For the paper: a short negative result – "class-conditional diffusion augmentation (ThirdMolarSynthesis-style) did not improve, and reduced, Bego recall".

Files on Hugging Face: `cdiff/` (5 generators, 5 × 464 synthetic crops, preview sheets), `report_syn_qc/` (QC table, closest-pair sheets),
`results_syn/` (30 jobs), `report_syn/` (comparison, per-run, analysis output). Scripts: `cdiff_brand.py`, `qc_syn.py`, `build_syn.py`, `syn_pipeline.py`.
