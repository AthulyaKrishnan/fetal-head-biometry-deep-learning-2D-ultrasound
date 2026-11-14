# Part A data

```
data/
├── images/        # 622 axial fetal head images: NNN_HC.png (NNN_2HC.png = second image of the same patient)
└── annotations/   # role_challenge_dataset_ground_truth.csv  (image_name + ofd_1_x ... bpd_2_y, original pixels)
```

| Folder | Created by | Contents |
|---|---|---|
| `reports/` | `tools/check_dataset.py` | `dataset_check.json`, `dataset_problems.csv`, `dataset_overview.png` |
| `data_processed/` | `tools/prepare_data.py` | cleaned images/annotations, `overlays/` for visual checking, patient-grouped train/val/test CSVs |
| `runs/<run_name>/` | `scripts/train.py`, `scripts/evaluate.py` | `best.pt`, `last.pt`, `log.csv`, `curves.png`, `config.json`, `eval_<split>/` |
| `report/` | `src/make_report.py` | tables, figures and written comparison report |
| `predictions/` | `tools/predict.py` | overlays and `predictions.json` for new images |
| `results/` | `scripts/test.py` | summary, per-image tables, figures for an external test set |
