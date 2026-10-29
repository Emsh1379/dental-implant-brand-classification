# Dental implant brand identification – a leakage-free benchmark

Code, results and a reusable evaluation protocol for identifying the implant brand (**Bego, Bicon, ITI**) on periapical radiographs of the public
Roboflow dataset [`dental-implant-nyw2h/-dental-implant`](https://universe.roboflow.com/dental-implant-nyw2h/-dental-implant) (CC BY 4.0).

* **Benchmark** – fixed patient-grouped folds, crop table, crop builder, evaluator and baseline predictions: [`BENCHMARK.md`](BENCHMARK.md), [`benchmark/`](benchmark/)
* **Data** – crops ready to use: [Hugging Face: emad2001/dental-implant-brand-benchmark](https://huggingface.co/datasets/emad2001/dental-implant-brand-benchmark)
* **Main findings** – [`FINAL_SUMMARY.md`](FINAL_SUMMARY.md)

## Key results
* Best model: ViT-B/16 on implant crops, macro-F1 0.954 / accuracy 0.986 (3 seeds × 5 grouped folds); 3-seed ensemble 0.972, Bego recall 0.961.
* Published protocols on the same data reach 98–99 % only with leaky splits; leakage-free they drop by 1.5–6 accuracy points and up to 0.36 Bego recall
  ([`RESULTS_PROTOCOLS.md`](RESULTS_PROTOCOLS.md), [`RESULTS_PAPER_PIPELINE.md`](RESULTS_PAPER_PIPELINE.md)).
* Image enhancement (sharpening / CLAHE pipeline), super-resolution, extra internet / public-dataset images and diffusion-generated images do not help
  ([`RESULTS_SEEDS.md`](RESULTS_SEEDS.md), [`RESULTS_ORIGINAL.md`](RESULTS_ORIGINAL.md), [`RESULTS_EXTRA_BEGO.md`](RESULTS_EXTRA_BEGO.md),
  [`RESULTS_EXTERNAL.md`](RESULTS_EXTERNAL.md), [`RESULTS_SYNTHETIC.md`](RESULTS_SYNTHETIC.md)).
* Clinical reliability: softmax confidence cannot reject unknown implant brands, feature-distance rejection can (AUROC 0.95–0.99); temperature scaling
  makes the probabilities reliable (ECE 0.25 → 0.01); deferring 5 % of cases removes 57 % of errors ([`RESULTS_OPENSET_CALIBRATION.md`](RESULTS_OPENSET_CALIBRATION.md)).

## Repository layout
* `benchmark/` – the benchmark (see above)
* `dental_cls/` – all experiment code (classification, detection, protocols, open-set, calibration, synthesis)
* `backup/` – result files and reports of every experiment
* `RESULTS_*.md` – one write-up per experiment; `HANDOFF.md` – how everything fits together and how to re-run it

## Licence
Code: MIT (`LICENSE`). Data: derived from the Roboflow dataset above, CC BY 4.0 – credit the original dataset when using it.
