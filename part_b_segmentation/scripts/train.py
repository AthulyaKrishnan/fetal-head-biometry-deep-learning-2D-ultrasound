#!/usr/bin/env python
"""STEP 3 - Train the U-Net with self- and cross-attention (see --self_attn_levels / --cross_attn_levels).

Example:
    python scripts/train.py --data_dir data_processed --run_name unet_base32 --epochs 100
Each run lives in runs/<run_name>/ : best.pt, last.pt, log.csv, curves.png, config.json
The checkpoint stores the model + preprocessing config, so evaluation needs no extra flags.
"""
import argparse
import time
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.dataset import FetalHeadDataset, make_loader
from src.inference import get_device, torch_load
from src.io_utils import save_json, seed_everything
from src.losses import HausdorffDistanceLoss, get_loss, hd_weight_at
from src.metrics import batch_dice_iou
from src.model import UNet
from src.train_utils import AverageMeter, CSVLogger, plot_curves, save_checkpoint


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data_processed")
    ap.add_argument("--run_name", default="unet_run")
    ap.add_argument("--out_dir", default="runs")
    # preprocessing
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--clahe", action="store_true", help="apply CLAHE contrast enhancement")
    ap.add_argument("--norm", default="zscore", choices=["zscore", "minmax"])
    ap.add_argument("--no_aug", action="store_true")
    # model
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--bilinear", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    # attention (the ONLY difference to the baseline U-Net repository)
    ap.add_argument("--self_attn_levels", type=int, nargs="*", default=None,
                    help="encoder levels (1..depth, depth = bottleneck) followed by self-attention. "
                         "Default: bottleneck and the level above. Give no values to disable.")
    ap.add_argument("--cross_attn_levels", type=int, nargs="*", default=None,
                    help="skip-connection levels (0..depth-1) using decoder->encoder cross-attention. "
                         "Default: the two deepest. Give no values to disable.")
    ap.add_argument("--no_attention", action="store_true", help="plain U-Net (identical to the baseline repo)")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--attn_dropout", type=float, default=0.0)
    ap.add_argument("--mlp_ratio", type=float, default=2.0, help="hidden size of the self-attention feed-forward layer")
    ap.add_argument("--max_kv_tokens", type=int, default=1024,
                    help="keys/values are spatially reduced so each attention layer uses at most this many")
    # optimisation
    ap.add_argument("--loss", default="bce_dice", choices=["dice", "bce", "bce_dice"])
    # Hausdorff-distance loss, added to the base loss with a warm-up:  total = base + w(epoch) * hd
    ap.add_argument("--hd_weight", type=float, default=0.0,
                    help="maximum weight of the Hausdorff loss (0.5 suggested); 0 = off")
    ap.add_argument("--hd_warmup_epochs", type=int, default=30,
                    help="weight rises linearly from 0 to hd_weight over these epochs")
    ap.add_argument("--hd_alpha", type=float, default=2.0, help="exponent of the distance weights")
    ap.add_argument("--hd_max_dist", type=float, default=32.0, help="distances are clipped at this many pixels")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau", "none"])
    ap.add_argument("--patience", type=int, default=20, help="early-stopping patience (epochs); 0 = off")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    # misc
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="resume from runs/<run_name>/last.pt")
    ap.add_argument("--max_samples", type=int, default=None, help="debug: use only N samples per split")
    return ap.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False,
              hd_loss=None, hd_weight=0.0):
    train = optimizer is not None
    model.train(train)
    loss_m, dice_m, iou_m = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, y = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
                if hd_loss is not None and hd_weight > 0:
                    loss = loss + hd_weight * hd_loss(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            d, i = batch_dice_iou(logits.float(), y)
            n = x.size(0)
            loss_m.update(loss.item(), n); dice_m.update(d.mean().item(), n); iou_m.update(i.mean().item(), n)
    return loss_m.avg, dice_m.avg, iou_m.avg


def main():
    args = parse()
    seed_everything(args.seed)
    device = get_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data = Path(args.data_dir)
    print(f"device: {device} | loading data ...")
    ds_kw = dict(image_size=args.image_size, clahe=args.clahe, norm=args.norm, max_samples=args.max_samples)
    train_ds = FetalHeadDataset(data / "train.csv", train=True, augment=not args.no_aug, **ds_kw)
    val_ds = FetalHeadDataset(data / "val.csv", train=False, **ds_kw)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    print(f"train {len(train_ds)} | val {len(val_ds)}")

    if args.no_attention:
        self_lv, cross_lv = [], []
    else:
        self_lv = [args.depth, args.depth - 1] if args.self_attn_levels is None else args.self_attn_levels
        cross_lv = [args.depth - 1, args.depth - 2] if args.cross_attn_levels is None else args.cross_attn_levels
    model_cfg = dict(in_channels=1, n_classes=1, base_channels=args.base_channels, depth=args.depth,
                     bilinear=args.bilinear, dropout=args.dropout,
                     self_attn_levels=list(self_lv), cross_attn_levels=list(cross_lv),
                     num_heads=args.num_heads, attn_dropout=args.attn_dropout, mlp_ratio=args.mlp_ratio,
                     max_kv_tokens=args.max_kv_tokens, image_size=args.image_size)
    print(f"self-attention levels: {self_lv or 'none'} | cross-attention levels: {cross_lv or 'none'}")
    config = dict(model=model_cfg,
                  preprocess=dict(image_size=args.image_size, clahe=args.clahe, norm=args.norm),
                  args=vars(args))
    save_json(config, run_dir / "config.json")

    model = UNet(**model_cfg).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    criterion = get_loss(args.loss)
    hd_loss = HausdorffDistanceLoss(args.hd_alpha, args.hd_max_dist) if args.hd_weight > 0 else None
    if hd_loss is not None:
        print(f"Hausdorff loss: max weight {args.hd_weight} | warm-up {args.hd_warmup_epochs} epochs | "
              f"alpha {args.hd_alpha} | max_dist {args.hd_max_dist}px")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    else:
        scheduler = None
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_dice, bad_epochs = 1, -1.0, 0
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch_load(run_dir / "last.pt", device)
        model.load_state_dict(ck["model_state"]); optimizer.load_state_dict(ck["optimizer_state"])
        if scheduler and ck.get("scheduler_state"):
            scheduler.load_state_dict(ck["scheduler_state"])
            if args.scheduler == "cosine" and scheduler.T_max != args.epochs:
                print(f"note: cosine schedule length changed {scheduler.T_max} -> {args.epochs} epochs")
                scheduler.T_max = args.epochs          # otherwise the saved length would override --epochs
        start_epoch, best_dice, bad_epochs = ck["epoch"] + 1, ck["best_dice"], ck.get("bad_epochs", 0)
        print(f"resumed from epoch {ck['epoch']} (best val dice {best_dice:.4f})")

    logger = CSVLogger(run_dir / "log.csv", resume=args.resume)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            lam = hd_weight_at(epoch, args.hd_weight, args.hd_warmup_epochs) if hd_loss is not None else 0.0
            tr_loss, tr_dice, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp,
                                            hd_loss=hd_loss, hd_weight=lam)
            va_loss, va_dice, va_iou = run_epoch(model, val_loader, criterion, device, use_amp=use_amp,
                                                 hd_loss=hd_loss, hd_weight=lam)
            lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step(va_dice) if args.scheduler == "plateau" else scheduler.step()

            improved = va_dice > best_dice
            if improved:
                best_dice, bad_epochs = va_dice, 0
            else:
                bad_epochs += 1
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, best_dice, config, bad_epochs)
            if improved:
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, best_dice, config, bad_epochs)

            logger.log(dict(epoch=epoch, train_loss=f"{tr_loss:.5f}", train_dice=f"{tr_dice:.5f}",
                            val_loss=f"{va_loss:.5f}", val_dice=f"{va_dice:.5f}", val_iou=f"{va_iou:.5f}",
                            lr=f"{lr:.2e}", hd_weight=f"{lam:.4f}", seconds=f"{time.time() - t0:.1f}"))
            print(f"epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
                  f"val loss {va_loss:.4f} dice {va_dice:.4f} iou {va_iou:.4f} | lr {lr:.1e} | "
                  f"hd w {lam:.3f} | {'*best*' if improved else ''}")
            if epoch % 5 == 0:
                plot_curves(run_dir / "log.csv", run_dir / "curves.png")
            if args.patience and bad_epochs >= args.patience:
                print(f"early stopping: no val-dice improvement for {args.patience} epochs")
                break
    except KeyboardInterrupt:
        print("interrupted - last.pt is saved; rerun with --resume to continue")
    finally:
        plot_curves(run_dir / "log.csv", run_dir / "curves.png")
    print(f"done. best val dice {best_dice:.4f} -> {run_dir/'best.pt'}")


if __name__ == "__main__":
    main()
