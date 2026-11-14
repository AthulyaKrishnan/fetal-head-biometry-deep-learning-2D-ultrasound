#!/usr/bin/env python
"""Self-test of the landmark pipeline. Run from the repository root:   python tools/check_model.py

Verifies (1) network output shape and the heatmap resolution, (2) heatmap generation and decoding
round trip, (3) the sub-pixel refinement, (4) metric values on known errors, (5) dataset items
(shapes, heatmap peak, flip augmentation), (6) that a saved checkpoint reloads to identical outputs.
Uses a small backbone (hrnet_w18_small, no download) so it runs on a CPU in seconds.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
import torch

from src.config import default_config
from src.dataset import FetalLandmarkDataset
from src.augmentation import build_train_augmentation
from src.heatmap import decode_heatmaps, generate_heatmaps
from src.inference import load_model
from src.metrics import compute_metrics
from src.model import HRNetLandmarkNet, build_model
from src.splitting import size_strata, split_quality, stratified_group_split
from src.train_utils import save_checkpoint


def small_cfg():
    cfg = default_config()
    cfg["data"]["input_size"] = 256
    cfg["model"].update(backbone="hrnet_w18_small", pretrained=False, head_channels=64)
    return cfg


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # 1. output shape
    net = HRNetLandmarkNet(backbone="hrnet_w18_small", pretrained=False, num_keypoints=4,
                           input_size=256, stride=4, head_channels=64).eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 256, 256))
    assert out.shape == (2, 4, 64, 64), out.shape
    print("1  output shape OK:", tuple(out.shape), "(batch, landmarks, 64, 64 heatmap)")

    # 2. heatmap round trip
    kps = np.array([[100.0, 60.0], [200.0, 40.0], [50.0, 150.0], [120.0, 180.0]], dtype=np.float32)
    hm = generate_heatmaps(kps, 96, 4, sigma=2.0)
    pred = decode_heatmaps(hm[None], 4)[0]
    err = np.abs(pred - kps).max()
    assert err <= 2.5, err
    print(f"2  heatmap generate -> decode recovers the landmarks (max error {err:.2f} px at stride 4)")

    # 3. sub-pixel refinement moves the estimate toward the higher neighbour
    line = np.zeros((1, 1, 1, 5), np.float32)
    line[..., 2], line[..., 3] = 1.0, 0.9
    x_refined = decode_heatmaps(line, 1)[0, 0, 0]
    assert x_refined > 2.0, x_refined
    print(f"3  quarter-pixel refinement: peak at cell 2 with neighbour 0.9 -> x = {x_refined:.2f}")

    # 4. metrics on known errors
    gt = np.array([[[0, 0], [10, 0], [0, 10], [10, 10]]], dtype=np.float32)
    m, _ = compute_metrics(gt + np.array([3.0, 4.0], np.float32), gt, pixel_spacing_mm=2.0)
    assert abs(m["mre_px"] - 5.0) < 1e-5 and abs(m["mre_mm"] - 10.0) < 1e-5 and m["ofd_abs_err_px"] < 1e-5
    print("4  metrics: 5 px offset -> MRE 5 px = 10 mm at 2 mm/px, OFD/BPD lengths unchanged")

    # 5. dataset items
    tmp = tempfile.mkdtemp()
    img_path = os.path.join(tmp, "000_HC.png")
    cv2.imwrite(img_path, np.full((120, 200), 128, np.uint8))
    rec = dict(image_path=img_path, image_name="000_HC.png",
               gt=np.array([[10, 10], [150, 10], [40, 60], [120, 60]], np.float32))
    cfg = small_cfg()
    flip_only = dict(cfg["augment"], enabled=True, hflip_p=1.0, affine_p=0.0,
                     brightness_contrast_p=0.0, gamma_p=0.0, blur_noise_p=0.0)
    ds = FetalLandmarkDataset([rec], input_size=256, stride=4, sigma=2.0,
                              transform=build_train_augmentation(flip_only))
    item = ds[0]
    assert tuple(item["image"].shape) == (3, 256, 256)
    assert tuple(item["heatmap"].shape) == (4, 64, 64)
    peak_x = int(item["heatmap"][0].argmax() % 64)
    assert peak_x > 32, peak_x            # ofd_1 at x=10 of 200 flips to the right half
    print("5  dataset: image (3,256,256), heatmaps (4,64,64); horizontal flip moves ofd_1 to the right half")

    # 6. checkpoint round trip
    model = build_model(cfg).eval()
    ckpt = os.path.join(tmp, "best.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(ckpt, model, opt, None, epoch=1, best_score=0.0, config=cfg)
    reloaded, _ = load_model(ckpt, torch.device("cpu"))
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x), atol=1e-6)
    print("6  checkpoint saves and reloads with identical outputs (config stored inside)")
    # 7. patient-grouped stratified split: every patient in one split, fractions close to targets
    prng = np.random.RandomState(1)
    n_img = {f"{i:03d}": int(prng.choice([1, 2])) for i in range(200)}
    size = {p: float(prng.uniform(120, 260)) for p in n_img}
    strata = size_strata(size, n_bins=3)
    fracs = {"train": 0.8, "val": 0.1, "test": 0.1}
    assign = stratified_group_split(n_img, strata, fracs, seed=42)
    assert set(assign) == set(n_img) and set(assign.values()) == {"train", "val", "test"}
    q = split_quality(assign, n_img, strata, fracs)
    assert q["max_fraction_deviation"] < 0.02, q
    print(f"7  patient-grouped stratified split: every patient in one split; max deviation from targets "
          f"{q['max_fraction_deviation']:.4f} (train/val/test = {q['train']['images']}/{q['val']['images']}/{q['test']['images']} images)")
    # 8. baseline: coordinate-regression network, its target and its normalised coordinates
    from src.model import CoordRegressionNet
    from src.inference import get_target
    reg = CoordRegressionNet(backbone="hrnet_w18_small", pretrained=False, num_keypoints=4,
                             input_size=256, head="flatten").eval()
    with torch.no_grad():
        yc = reg(torch.zeros(2, 3, 256, 256))
    assert yc.shape == (2, 4, 2) and reg.output_type == "coords", yc.shape
    plain = dict(image_path=img_path, image_name="000_HC.png", gt=rec["gt"])
    ds_plain = FetalLandmarkDataset([plain], input_size=256, stride=4, sigma=2.0, transform=None)
    it = ds_plain[0]
    expected = rec["gt"] / np.array([200.0, 120.0], np.float32)      # coordinates as fractions of the input
    assert np.allclose(it["coords_norm"].numpy(), expected, atol=1e-5), (it["coords_norm"], expected)
    assert get_target(reg, it) is it["coords_norm"]
    assert get_target(model, it) is it["heatmap"]
    print("8  baseline: coordinate head outputs (2, 4, 2); its target is the normalised coordinates (checked "
          "against the ground truth); the heatmap model still uses heatmaps")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
