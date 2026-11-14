# Part A: Landmark Detection (code guide)

HRNet-W32 heatmap regression for the four OFD/BPD end-points, with a direct coordinate-regression baseline.
The full write-up (motivation, data checks, method, results, figures, limitations) is in the [top-level README](../README.md#part-a-landmark-detection).

Headline: median landmark error **1.65 mm**, **70.7 %** of landmarks within 2 mm, median length error **1.0 % (OFD) / 1.5 % (BPD)**, head circumference ICC 0.90 (47 test images).

## Folder layout

```
part_a_landmark_detection/
├── scripts/                 # the three entry points
│   ├── train.py             #   train the heatmap model, or the coordinate baseline (--model_type regression --head flatten)
│   ├── evaluate.py          #   landmark and OFD/BPD errors on a split -> runs/<run>/eval_<split>/
│   └── test.py              #   evaluate a trained model on any image folder + annotation CSV, in millimetres
├── src/                     # library code
│   ├── model.py  heatmap.py losses.py          # HRNet heatmap net, coordinate baseline, heatmap targets/decoding, losses
│   ├── dataset.py augmentation.py preprocessing.py splitting.py
│   ├── inference.py metrics.py postprocess.py train_utils.py config.py constants.py io_utils.py
│   └── make_report.py report.py figures.py mm_metrics.py stats.py derived.py load.py   # millimetre report layer
├── tools/                   # supporting utilities, run in this order for a full study
│   ├── check_dataset.py     #   dataset quantity/quality check -> reports/
│   ├── prepare_data.py      #   remove implausible annotations, patient-grouped split -> data_processed/
│   ├── compare_models.py    #   paired comparison of two evaluated models
│   ├── predict.py           #   predictions on new images (no ground truth needed)
│   ├── export_model.py      #   export a checkpoint as a full saved model
│   ├── visualize_heatmaps.py#   show each landmark's predicted heatmap on an image
│   └── check_model.py       #   CPU self-test (shapes, heatmap round trip, sub-pixel refinement, metrics)
├── data/                    # images/ and annotations/ (not tracked; see data/README.md)
├── weights/                 # trained weights (not tracked)
├── run_pipeline.sh          # the full sequence of experiments behind the report
└── requirements.txt
```

Created when you run the code (not tracked): `reports/`, `data_processed/`, `runs/`, `report/`, `predictions/`, `results/`.

## Running

Run everything from inside this folder.

```bash
cd part_a_landmark_detection
pip install -r requirements.txt
# copy the data into data/ (see data/README.md), then the whole study:
bash run_pipeline.sh
```

Or step by step:

```bash
python tools/check_dataset.py --img_dir data/images --csv data/annotations/role_challenge_dataset_ground_truth.csv
python tools/prepare_data.py  --img_dir data/images --csv data/annotations/role_challenge_dataset_ground_truth.csv
python scripts/train.py    --run_name hrnet_w32_base_run1 --epochs 150 --batch_size 4 --device cuda --amp --num_workers 4
python scripts/evaluate.py --checkpoint runs/hrnet_w32_base_run1/best.pt --split test

# baseline: only the head and the loss differ
python scripts/train.py    --inherit runs/hrnet_w32_base_run1/config.json --run_name coord_baseline --model_type regression --head flatten
python scripts/evaluate.py --checkpoint runs/coord_baseline/best.pt --split test

python tools/compare_models.py runs/hrnet_w32_base_run1/eval_test runs/coord_baseline/eval_test --name_a heatmap --name_b coord
python src/make_report.py --runs runs/hrnet_w32_base_run1/eval_test runs/coord_baseline/eval_test \
    --names heatmap coord --splits_dir data_processed --pixel_spacing_mm 0.1 --out report/comparison
```

Evaluate a trained model on your own test set (all results in millimetres):

```bash
python scripts/test.py --weights weights/best.pt --images <image folder> --annotations <annotations.csv> \
    --pixel_spacing_mm 0.1 --result_dir results/my_test
```

## Metrics implemented

Mean radial error per image (median, IQR, 90th percentile, bootstrap CIs), per-landmark error, success detection rate at 2 / 2.5 / 3 / 4 mm, SDR-AUC, endpoint-order-invariant NME, OFD/BPD/HC absolute error, bias and 95 % limits of agreement (Bland-Altman), ICC(2,1), share of lengths within 5 %, and failure analysis (images above 3 × median, share of error in the worst 10 %). Head circumference is HC = 1.62 × (BPD + OFD).

## Code layout note

Step scripts that previously sat in `src/` with flat imports now live in `tools/` and import from the `src` package; `Trainer.py` became `scripts/train.py` and `Tester.py` became `scripts/test.py`. Only import lines and the tester's default `weights/` and `results/` locations changed.
