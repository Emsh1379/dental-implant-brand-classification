# Extra Bego radiographs (auto-annotated)

Source: Roboflow `lamiaa/implant-recognition` version 3 (CC BY 4.0). Of its 2,796 images, 2,778 are augmented copies of radiographs already in
our dataset (checked by image content). The other **18 images have no annotations** and come from **3 Bego radiographs that are not in our dataset**:
`bego-40` (7 copies), `bego-51` (4), `bego-52` (7). Each shows one implant. Side-by-side checks against the most similar existing radiographs
(`dup_check.jpg`) show they are different radiographs.

## How the boxes were made
1. `auto_annotate.py`: the 5 YOLO11n fold detectors (`det_checkpoints/yolo11n_f*`) run on each image; a box is kept when ≥ 3 of 5 models agree (IoU > 0.5, conf ≥ 0.25).
   The detectors voted Bego for most boxes (some votes went to ITI – the usual Bego/ITI confusion).
2. `refine_boxes.py`: one implant per image, so only the best-supported box is kept; it is snapped to the implant metal
   (threshold halfway between Otsu and the 99.5th percentile inside the box enlarged by 30 %). All 18 snapped; none fell back to the detector box.
3. Checked visually (`contact_sheet.jpg`, `preview/`): every box covers the whole implant. **Label = Bego** for all, from the file name.

These are automatic annotations. Have them confirmed (ideally by the clinician who checks `consistent_errors.csv`) before publishing.

## Files
`images/`, `labels/` (YOLO, class 0 = Bego), `_annotations.coco.json`, `auto_boxes.csv` (all consensus candidates and votes), `preview/`, `contact_sheet.jpg`.

## In the dataset: `crops_plus/` (Hugging Face)
`build_plus.py` appends the 18 crops (idx 3276–3293) to `raw_orig`, `raw_orig384`, `pipeline_orig`, `meta.csv` and `meta_groups.csv`, and writes an
extended `folds.json`. Nothing existing changes: rows 0–3275 and all fold assignments stay as they were, so every earlier result remains valid.
Each new radiograph is its own group, used as test data (primary image only) in the fold with the fewest Bego test groups and as training data in the other four.
Rep 0: `bego-40` and `bego-51` → test in fold 1, `bego-52` → test in fold 0. Unique Bego test implants: 76 → 79.
There is no HAT version of these images, so `crops_plus` only supports the original-image inputs.
Use: `python train_cv.py --crops <crops_plus> --versions raw_orig ...`
