# Fetal Head Biometry from 2D Ultrasound

Measuring the fetal head (OFD, BPD, head circumference) in axial ultrasound with two deep-learning approaches, compared honestly.

| | **Part A: Landmark detection** | **Part B: Segmentation + ellipse fit** |
|---|---|---|
| Idea | Predict the 4 OFD/BPD end-points directly | Segment the skull, fit an ellipse, read the end-points from its axes |
| Model | HRNet-W32 (ImageNet-pretrained), Gaussian heatmaps | U-Net (7.76 M params); attention and HD-loss variants tested |
| Baseline | Direct coordinate regression, same backbone | Plain U-Net vs attention vs attention + HD loss |
| Data used | 470 of 622 annotations (24.4 % removed by a geometry check) | 620 of 622 images |
| Split (patient-grouped) | 376 / 47 / 47 images | 506 / 58 / 56 images |

**Stack:** Python, PyTorch, timm, OpenCV, albumentations, pandas, SciPy

---

## Results at a glance

| | Part A (47 test images, mm) | Part B (56 test images, px) |
|---|---|---|
| Landmark / segmentation | Median error **1.65 mm**, 70.7 % within 2 mm | Mean Dice **0.969** (median 0.978) |
| Length / end-point | Median length error **1.0 % OFD, 1.5 % BPD** | Median end-point error **12.1 px OFD, 9.5 px BPD** |
| Head circumference | ICC 0.90 | Median abs. error 9.9 px |
| Selected model | Heatmap regression | Baseline U-Net |

> The two parts use different splits and units, so the numbers are not directly comparable.

---

## Part A: Landmark detection

**Pipeline:** check annotations → patient-grouped split → CLAHE + resize 384×384 → HRNet-W32 heatmaps (96×96) → argmax + sub-pixel refinement

**Key choices**
- Removed implausible annotations (OFD/BPD >30° from perpendicular, midpoints >150 px apart, diameter <100 px, point outside image)
- Split by patient so no patient is in two sets
- Heatmaps keep spatial layout; HRNet keeps high resolution throughout
- Baseline changes only the head and loss (everything else identical)

**Heatmap vs coordinate regression**

| Metric | Heatmap | Coordinate |
|---|---|---|
| Median landmark error (mm) | **1.65** | 2.48 |
| Landmarks within 2 mm | **70.7 %** | 46.8 % |
| BPD length MAE (mm) | **1.97** | 2.77 |
| OFD ICC / BPD ICC / HC ICC | 0.59 / 0.82 / 0.90 | **0.91 / 0.91 / 0.97** |

<p align="center"><img src="images/part_a/cumulative_success_curve.png" width="60%" alt="Cumulative success curve"></p>

**Takeaways**
- Heatmaps place landmarks more accurately
- The coordinate model agrees better on OFD, BPD and HC
- 4 of 47 images carry about 43 % of the total error

---

## Part B: Segmentation and ellipse fitting

**Pipeline:** outline → filled mask → letterbox 256×256 + z-score → U-Net → clean mask → `fitEllipse` → OFD (major axis), BPD (minor axis), HC

**Key choices**
- 1-pixel ellipse outlines filled into masks; open outlines excluded and masks inspected visually
- Letterboxing keeps the ellipse undistorted
- Attention (+39 % params) and HD loss tested one change at a time

**Model comparison (56 test images)**

| | Baseline U-Net | + Attention | + Attention + HD loss |
|---|---|---|---|
| Parameters | **7.76 M** | 10.79 M | 10.79 M |
| Dice (mean) | 0.969 | 0.969 | 0.969 |
| HD95, median (px) | **9.64** | 10.12 | 10.53 |
| OFD / BPD end-point error, median (px) | 12.1 / **9.5** | **11.8** / 9.7 | 12.4 / 10.4 |
| HC abs. error, median (px) | **9.9** | 13.4 | 11.6 |

No difference survived Holm correction, so the simpler baseline was kept.

<p align="center"><img src="images/part_b/best_and_worst_fits.png" width="75%" alt="Best and worst fits"></p>
<p align="center"><em>Best and worst test images: green = ground truth, red = prediction, yellow = OFD points, blue = BPD points.</em></p>

**Takeaways**
- Overlap is saturated (Dice 0.97) but biometry is not: end-point errors are 10 to 12 px
- The same image (`007_HC`) is the worst case for all three models
- Dice 0.97 hides about ±11 % per-image area error

---

## What the project shows

- A headline average hides the tail, so medians, spreads and failure shares are reported next to means
- The optimised metric is not the clinical measurement: better landmarks did not mean better length agreement, and high Dice did not mean accurate end-points
- Changes are tested one at a time, and extensions are dropped when they do not earn their cost

## Limitations

- Part A results apply to the 470 kept annotations; the removal rule may reflect a different annotation convention
- Small test sets (47 and 56 images, one split each)
- Part A assumes 0.1 mm/px and may not have fully converged (early stopping at epoch 70)
- Part B end-point matching takes the better of two pairings, which slightly flatters the error

---


# Repository guide

```
.
├── README.md                      # this write-up
├── part_a_landmark_detection/     # code for Part A
│   ├── scripts/                   #   train.py, evaluate.py, test.py
│   ├── src/                       #   model, dataset, augmentation, heatmap, losses, metrics, splitting, inference, report layer
│   ├── tools/                     #   check_dataset, prepare_data, predict, export_model, compare_models, visualize_heatmaps, check_model
│   ├── data/  weights/            #   not tracked
│   ├── run_pipeline.sh  requirements.txt  README.md
├── part_b_segmentation/           # code for Part B (same structure)
│   ├── scripts/                   #   train.py, evaluate.py (full evaluation), test.py
│   ├── src/                       #   U-Net, attention, losses (incl. HD loss), dataset, preprocessing, postprocess (ellipse fit), metrics
│   ├── tools/                     #   check_dataset, prepare_data, predict, export_model, quick_evaluate, make_report_figures, visualize_attention, check_model
│   ├── data/  weights/            #   not tracked
│   ├── run_pipeline.sh  requirements.txt  README.md
├── docs/
│   ├── images/                    # figures used in this README
│   └── reports/                   # the original PDF reports
├── requirements.txt               # installs both parts
```

Each part is run from inside its own folder; see [Part A](part_a_landmark_detection/README.md) and [Part B](part_b_segmentation/README.md) for the exact commands.

```bash
git clone https://github.com/<your-username>/fetal-head-biometry.git
cd fetal-head-biometry
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd part_a_landmark_detection   # or part_b_segmentation
bash run_pipeline.sh           # after copying the data into data/ (see data/README.md)
```

# References

- Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
- Zeng, Y., Tsui, P.-H., Wu, W., Zhou, Z., Wu, S. *Fetal Ultrasound Image Segmentation for Automatic Head Circumference Biometry Using Deeply Supervised Attention-Gated V-Net.* Journal of Digital Imaging 34:134 to 148, 2021.
- Karimi, D., Salcudean, S. E. *Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks.* IEEE Transactions on Medical Imaging, 2019.
- Payer, C., Štern, D., Bischof, H., Urschler, M. *Regressing Heatmaps for Multiple Landmark Localization using CNNs.* MICCAI 2016.
- Chen, R., Ma, Y., Chen, N., Lee, D., Wang, W. *Cephalometric Landmark Detection by Attentive Feature Pyramid Fusion and Regression-Voting.* MICCAI 2019.
- Noothout, J. M. H., de Vos, B. D., Wolterink, J. M., et al. *Deep Learning-Based Regression and Classification for Automatic Landmark Localization in Medical Images.* IEEE Transactions on Medical Imaging, 2020.
- Di Vece, C., Mao, Z., Avisdris, N., et al. *A multicentre benchmark dataset for comprehensive landmark-based fetal ultrasound biometry.* Scientific Reports. Dataset: https://github.com/surgical-vision/Multicentre-Fetal-Biometry
- ISUOG Clinical Standards Committee. *ISUOG Practice Guidelines (updated): performance of the routine mid-trimester fetal ultrasound scan.*
