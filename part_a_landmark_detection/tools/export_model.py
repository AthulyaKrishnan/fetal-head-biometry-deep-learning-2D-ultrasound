#!/usr/bin/env python
"""STEP 6 - Export a trained checkpoint as a FULL saved model (torch.save(model)).

    python tools/export_model.py --checkpoint runs/hrnet_w32_base/best.pt --output "Model Weights/hrnet_landmarks_full_saved_model.pth"
Load later with:   model = torch.load(path, weights_only=False)   # needs the `src` package importable
Preprocessing settings are stored on the model as `model.preprocess_cfg`.
"""
import argparse
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.inference import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    model, cfg = load_model(a.checkpoint, torch.device("cpu"))
    model.preprocess_cfg = cfg["preprocess"]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, a.output)

    try:                                                    # verify it reloads and matches
        reloaded = torch.load(a.output, map_location="cpu", weights_only=False)
    except TypeError:                                       # older torch without weights_only
        reloaded = torch.load(a.output, map_location="cpu")
    S = cfg["data"]["input_size"]
    x = torch.randn(1, 3, S, S)                             # the network takes 3 channels
    assert torch.allclose(model.eval()(x), reloaded.eval()(x), atol=1e-5), "reload mismatch"
    print(f"saved + verified: {a.output}  ({Path(a.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
