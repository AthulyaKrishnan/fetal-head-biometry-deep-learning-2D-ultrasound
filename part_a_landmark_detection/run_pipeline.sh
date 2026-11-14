#!/usr/bin/env bash
# Part A: landmark detection. Run from this folder (part_a_landmark_detection/).
# Order and flags follow the experiments in the report: proposed heatmap model, then the coordinate baseline.
set -euo pipefail

IMG_DIR=data/images
CSV=data/annotations/role_challenge_dataset_ground_truth.csv

# 1. Dataset check -> reports/
python tools/check_dataset.py --img_dir "$IMG_DIR" --csv "$CSV"

# 2. Remove implausible annotations, patient-grouped split -> data_processed/
#    Open data_processed/overlays and check that the points sit on the right structures.
python tools/prepare_data.py --img_dir "$IMG_DIR" --csv "$CSV"

# 3. Proposed model: HRNet-W32 heatmap regression -> runs/hrnet_w32_base_run1/
python scripts/train.py --run_name hrnet_w32_base_run1 --epochs 150 --batch_size 4 \
    --device cuda --amp --num_workers 4

# 4. Evaluate on the test split -> runs/hrnet_w32_base_run1/eval_test/
python scripts/evaluate.py --checkpoint runs/hrnet_w32_base_run1/best.pt --split test

# Hypothesis 2: direct coordinate regression, same backbone / data / schedule (only head and loss differ)
python scripts/train.py --inherit runs/hrnet_w32_base_run1/config.json \
    --run_name coord_baseline --model_type regression --head flatten
python scripts/evaluate.py --checkpoint runs/coord_baseline/best.pt --split test

# Paired comparison on the same test images
python tools/compare_models.py runs/hrnet_w32_base_run1/eval_test runs/coord_baseline/eval_test \
    --name_a heatmap --name_b coord

# Report in millimetres (the report uses 0.1 mm per pixel) -> report/comparison/
python src/make_report.py --runs runs/hrnet_w32_base_run1/eval_test runs/coord_baseline/eval_test \
    --names heatmap coord --splits_dir data_processed --pixel_spacing_mm 0.1 --out report/comparison

# Optional
# python tools/predict.py --checkpoint runs/hrnet_w32_base_run1/best.pt --input data/new_images --out_dir predictions
# python tools/export_model.py --checkpoint runs/hrnet_w32_base_run1/best.pt --output weights/hrnet_landmarks_full_saved_model.pth
# python tools/visualize_heatmaps.py --checkpoint runs/hrnet_w32_base_run1/best.pt --image <image.png> --out heatmaps.png
