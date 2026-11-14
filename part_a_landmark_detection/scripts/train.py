#!/usr/bin/env python
"""STEP 3 - Train the HRNet landmark detector (heatmap regression).

Example:
    python scripts/train.py --data_dir data_processed --run_name hrnet_w32_base --epochs 150 --batch_size 16 --amp
Each run lives in runs/<run_name>/ : best.pt, last.pt, log.csv, curves.png, config.json
The checkpoint stores the model, the optimiser state and the full configuration, so evaluation
and prediction need no extra flags.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.config import default_config
from src.dataset import build_datasets, make_loader
from src.inference import get_device, get_target, run_inference, torch_load
from src.io_utils import save_json, seed_everything
from src.losses import CoordSmoothL1Loss, HeatmapMSELoss
from src.metrics import compute_metrics
from src.model import build_model
from src.train_utils import build_scheduler, plot_curves, save_checkpoint, selection_metric


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inherit", default=None,
                    help="config.json of an earlier run: copy all its settings, then apply the flags given here "
                         "(use this to make a baseline differ from the proposed model in one setting only)")
    ap.add_argument("--data_dir", default="data_processed")
    ap.add_argument("--run_name", default="hrnet_run")
    ap.add_argument("--out_dir", default="runs")
    # data and preprocessing
    ap.add_argument("--input_size", type=int, default=384)
    ap.add_argument("--heatmap_stride", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma in heatmap pixels")
    ap.add_argument("--pixel_spacing_mm", type=float, default=None, help="mm per pixel; enables mm metrics")
    ap.add_argument("--no_clahe", action="store_true", help="disable CLAHE contrast enhancement")
    ap.add_argument("--no_aug", action="store_true", help="disable training augmentation")
    ap.add_argument("--max_samples", type=int, default=None, help="debug: use only N samples per split")
    # model
    ap.add_argument("--backbone", default="hrnet_w32", help="timm HRNet name, e.g. hrnet_w18, hrnet_w48")
    ap.add_argument("--no_pretrained", action="store_true", help="train the backbone from random weights")
    ap.add_argument("--head_channels", type=int, default=256)
    ap.add_argument("--model_type", default="heatmap", choices=["heatmap", "regression"],
                    help="heatmap = proposed HRNet heatmap model; regression = direct coordinate baseline")
    ap.add_argument("--head", default="flatten", choices=["flatten", "gap"],
                    help="regression baseline head: flatten keeps spatial layout, gap pools it away")
    # optimisation
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--pos_weight", type=float, default=0.0, help=">0 up-weights heatmap peaks in the loss")
    ap.add_argument("--patience", type=int, default=30, help="early-stopping patience (epochs); 0 = off")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    # misc
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="resume from runs/<run_name>/last.pt")
    # --inherit: take the defaults from an earlier run's saved arguments; explicit flags still win
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--inherit", default=None)
    known, _ = pre.parse_known_args()
    if known.inherit:
        saved = json.loads(Path(known.inherit).read_text())["args"]
        valid = {act.dest for act in ap._actions}
        ap.set_defaults(**{k: v for k, v in saved.items() if k in valid and k not in ("run_name", "resume")})
        print(f"inheriting settings from {known.inherit}")
    return ap.parse_args()


def build_config(args):
    cfg = default_config()
    cfg["seed"] = args.seed
    cfg["data"].update(input_size=args.input_size, heatmap_stride=args.heatmap_stride,
                       sigma=args.sigma, pixel_spacing_mm=args.pixel_spacing_mm)
    cfg["preprocess"]["use_clahe"] = not args.no_clahe
    cfg["augment"]["enabled"] = not args.no_aug
    cfg["model"].update(backbone=args.backbone, pretrained=not args.no_pretrained,
                        head_channels=args.head_channels, type=args.model_type, head=args.head)
    cfg["train"].update(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
                        lr=args.lr, weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
                        grad_clip=args.grad_clip, amp=args.amp, pos_weight=args.pos_weight,
                        patience=args.patience)
    cfg["args"] = vars(args)
    return cfg


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, grad_clip):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        target = get_target(model, batch).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(img)
        loss = criterion(out.float(), target)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        n += 1
    return total / max(1, n)


def main():
    args = parse()
    seed_everything(args.seed)
    device = get_device(args.device)
    amp = args.amp and device.type == "cuda"
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    save_json(cfg, run_dir / "config.json")
    stride, epochs, t = cfg["data"]["heatmap_stride"], args.epochs, cfg["train"]

    print(f"device: {device} | loading data ...")
    datasets = build_datasets(args.data_dir, cfg, args.max_samples)
    train_loader = make_loader(datasets["train"], t["batch_size"], True, t["num_workers"], drop_last=True)
    val_loader = make_loader(datasets["val"], t["batch_size"], False, t["num_workers"])
    print(f"train {len(datasets['train'])} | val {len(datasets['val'])}")

    model = build_model(cfg).to(device)
    print(f"model: {cfg['model']['type']}" + (f" (head={cfg['model']['head']})" if cfg['model']['type'] == 'regression' else "")
          + f" | parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    if getattr(model, "output_type", "heatmap") == "coords":
        criterion = CoordSmoothL1Loss()          # baseline: loss on normalised coordinates
    else:
        criterion = HeatmapMSELoss(t["pos_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = build_scheduler(optimizer, epochs, t["warmup_epochs"])
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    start_epoch, best, bad_epochs, history = 1, float("inf"), 0, []
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch_load(run_dir / "last.pt", device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("scheduler_state"):
            scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch, best, bad_epochs = ck["epoch"] + 1, ck["best_score"], ck.get("bad_epochs", 0)
        if (run_dir / "log.csv").exists():
            history = pd.read_csv(run_dir / "log.csv").to_dict("records")
        print(f"resumed from epoch {ck['epoch']} (best score {best:.4f})")

    try:
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp, t["grad_clip"])
            val = run_inference(model, val_loader, device, stride, criterion, amp)
            val_metrics, _ = compute_metrics(val["pred"], val["gt"], cfg["data"]["pixel_spacing_mm"])
            score = selection_metric(val_metrics)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            improved = score < best
            if improved:
                best, bad_epochs = score, 0
            else:
                bad_epochs += 1
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, best, cfg, bad_epochs)
            if improved:
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, best, cfg, bad_epochs)

            history.append(dict(epoch=epoch, lr=f"{lr:.3e}", train_loss=round(tr_loss, 6),
                                val_loss=round(val["loss"], 6), val_mre_px=round(val_metrics["mre_px"], 4),
                                val_mre_mm=round(val_metrics["mre_mm"], 4) if "mre_mm" in val_metrics else None,
                                seconds=round(time.time() - t0, 1)))
            pd.DataFrame(history).to_csv(run_dir / "log.csv", index=False)
            print(f"epoch {epoch:3d}/{epochs} | train loss {tr_loss:.5f} | val loss {val['loss']:.5f} "
                  f"| val MRE {val_metrics['mre_px']:.2f}px"
                  + (f" / {val_metrics['mre_mm']:.3f}mm" if "mre_mm" in val_metrics else "")
                  + f" | lr {lr:.1e} {'*best*' if improved else ''}")
            if epoch % 5 == 0:
                plot_curves(run_dir / "log.csv", run_dir / "curves.png")
            if t["patience"] and bad_epochs >= t["patience"]:
                print(f"early stopping: no improvement for {t['patience']} epochs")
                break
    except KeyboardInterrupt:
        print("interrupted - last.pt is saved; rerun with --resume to continue")
    finally:
        if (run_dir / "log.csv").exists():
            plot_curves(run_dir / "log.csv", run_dir / "curves.png")
    print(f"done. best validation score {best:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
