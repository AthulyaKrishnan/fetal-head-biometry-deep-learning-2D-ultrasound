# Part B: Segmentation and Ellipse Fitting (code guide)

U-Net skull segmentation, ellipse fit for the OFD/BPD end-points and head circumference, plus self-/cross-attention and Hausdorff-type loss variants.
The full write-up (motivation, data checks, method, results, figures, limitations) is in the [top-level README](../README.md#part-b-segmentation-and-ellipse-fitting).

Headline: mean Dice **0.969**, median OFD / BPD end-point error **12.1 / 9.5 px**; attention and HD loss gave no significant gain, so the baseline U-Net was kept (56 test images).

## Folder layout

```
part_b_segmentation/
├── scripts/                 # the three entry points
│   ├── train.py             #   train the U-Net; --no_attention = baseline, --hd_weight adds the Hausdorff-type loss
│   ├── evaluate.py          #   full evaluation of one model on a split -> eval_full/<name>/
│   └── test.py              #   evaluate a model on any image folder + ground-truth folder (mm if a pixel-size CSV is given)
├── src/                     # library code
│   ├── model.py attention.py                   # U-Net, self-/cross-attention blocks
│   ├── losses.py                               # BCE, Dice, BCE+Dice, distance-weighted Hausdorff-type loss, weight warm-up
│   ├── dataset.py augmentation.py preprocessing.py   # outline -> mask, letterbox, z-score
│   ├── postprocess.py                          # mask clean-up, ellipse fit, OFD/BPD end-points, HC, point matching
│   ├── metrics.py full_metrics.py              # overlap, boundary and ranking metrics
│   └── inference.py train_utils.py io_utils.py
├── tools/                   # supporting utilities
│   ├── check_dataset.py     #   ground-truth and pairing check -> reports/
│   ├── prepare_data.py      #   outline -> filled mask, patient-wise 80/10/10 split, overlays -> data_processed/
│   ├── quick_evaluate.py    #   lighter evaluation (Dice, HD95, biometry)
│   ├── make_report_figures.py #  figures and tables comparing models -> report_figures/
│   ├── predict.py           #   masks and ellipse measurements for new images
│   ├── export_model.py      #   export a checkpoint as a full saved model
│   ├── visualize_attention.py #  attention maps for a query point
│   └── check_model.py       #   self-test of the attention U-Net
├── data/                    # images/ and annotations/ (not tracked; see data/README.md)
├── weights/                 # trained weights (not tracked)
├── run_pipeline.sh          # the three experiments behind the report
└── requirements.txt
```

Created when you run the code (not tracked): `reports/`, `data_processed/`, `runs/`, `eval_full/`, `report_figures/`, `predictions/`, `results/`.

## Running

Run everything from inside this folder.

```bash
cd part_b_segmentation
pip install -r requirements.txt
# copy the data into data/ (see data/README.md), then the whole study:
bash run_pipeline.sh
```

Or step by step:

```bash
python tools/check_dataset.py --img_dir data/images --gt_dir data/annotations --gt_suffix _Annotation
python tools/prepare_data.py  --img_dir data/images --gt_dir data/annotations --gt_suffix _Annotation   # then look at data_processed/overlays/

python scripts/train.py --data_dir data_processed --run_name unet_baseline      --epochs 100 --amp --num_workers 4 --no_attention
python scripts/train.py --data_dir data_processed --run_name unet_run           --epochs 100 --amp --num_workers 4
python scripts/train.py --data_dir data_processed --run_name unet_attention_hd05 --epochs 100 --amp --num_workers 4 --hd_weight 0.5 --hd_warmup_epochs 30

python scripts/evaluate.py --checkpoint runs/unet_baseline/best.pt --name unet_baseline --split test
```

Evaluate a trained model on your own test set:

```bash
python scripts/test.py --weights weights/best.pt --images <image folder> --gt <ground-truth folder> \
    --gt_suffix _Annotation --result_dir results/my_run --make_figures
```

## Code layout note

Step scripts that previously sat in `src/` with flat imports now live in `tools/` and import from the `src` package; `Trainer.py` became `scripts/train.py` and `Tester.py` became `scripts/test.py`; the full evaluation is `scripts/evaluate.py` and the lighter one is `tools/quick_evaluate.py`. Only import lines and the tester's root-folder lookup changed.
