#!/usr/bin/env bash
# Part B: segmentation + ellipse fitting. Run from this folder (part_b_segmentation/).
# Three models, as in the report: baseline U-Net, + attention, + attention + Hausdorff-type loss.
set -euo pipefail

IMG_DIR=data/images
GT_DIR=data/annotations
GT_SUFFIX=_Annotation

# 1. Dataset check -> reports/
python tools/check_dataset.py --img_dir "$IMG_DIR" --gt_dir "$GT_DIR" --gt_suffix "$GT_SUFFIX"

# 2. Outline -> filled mask, patient-wise 80/10/10 split -> data_processed/
#    Look at data_processed/overlays/ before training: a wrong mask gives no error message.
python tools/prepare_data.py --img_dir "$IMG_DIR" --gt_dir "$GT_DIR" --gt_suffix "$GT_SUFFIX"

# 3. Train (checkpoint, log and curves in runs/<run_name>/)
# Baseline U-Net (selected model)
python scripts/train.py --data_dir data_processed --run_name unet_baseline \
    --epochs 100 --amp --num_workers 4 --no_attention
# + self- and cross-attention (attention is on by default)
python scripts/train.py --data_dir data_processed --run_name unet_run \
    --epochs 100 --amp --num_workers 4
# + attention + Hausdorff-type boundary loss (weight 0.5, linear warm-up over 30 epochs)
python scripts/train.py --data_dir data_processed --run_name unet_attention_hd05 \
    --epochs 100 --amp --num_workers 4 --hd_weight 0.5 --hd_warmup_epochs 30

# 4. Full evaluation on the test split -> eval_full/<name>/
python scripts/evaluate.py --checkpoint runs/unet_baseline/best.pt --name unet_baseline --split test
python scripts/evaluate.py --checkpoint runs/unet_run/best.pt --name attention_unet --split test
python scripts/evaluate.py --checkpoint runs/unet_attention_hd05/best.pt --name attention_hd05 --split test

# 5. Comparison figures -> report_figures/
python tools/make_report_figures.py \
    --evals U-Net=eval_full/unet_baseline U-Net_plus_Attention_Layers=eval_full/attention_unet \
            U-Net_attention_hdloss=eval_full/attention_hd05 \
    --runs  U-Net=runs/unet_baseline U-Net_plus_Attention_Layers=runs/unet_run \
            U-Net_attention_hdloss=runs/unet_attention_hd05 \
    --out report_figures

# Optional
# python tools/predict.py --checkpoint runs/unet_baseline/best.pt --input some_folder --out_dir predictions
# python tools/export_model.py --checkpoint runs/unet_baseline/best.pt --output weights/unet_full_saved_model.pth
# python tools/visualize_attention.py --checkpoint runs/unet_run/best.pt --image <image.png> --query_xy 0.5 0.5 --out attention.png
