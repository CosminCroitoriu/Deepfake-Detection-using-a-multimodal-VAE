#!/usr/bin/env python3
"""
Train the Masked Concatenation Multimodal VAE (Step 5).

Same training protocol as Step 3 (Concat VAE) but with the model class swapped
for one that randomly zeroes modality channels during training. Loss formula is
the unweighted MSE+KL+L1 from Step 3 — no gating, no α weights.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..concat_vae.transform import ConcatTransform
from .dataset import RealTrainDataset
from .model import MaskedConcatVAE, vae_loss

SCRIPT_DIR = Path(__file__).resolve().parent


def run_epoch(model, loader, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    total_n = 0
    with torch.set_grad_enabled(training):
        for input_5ch, target_5ch in loader:
            input_5ch = input_5ch.to(device)
            target_5ch = target_5ch.to(device)
            # NOTE: model.forward() does the masking internally when training.
            # The target stays unmasked — that's what makes this a robustness
            # task: predict the full target from a partial input.
            recon, mu, logvar = model(input_5ch)
            loss = vae_loss(recon, target_5ch, mu, logvar)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * input_5ch.size(0)
            total_n += input_5ch.size(0)
    return total_loss / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--ckpt_dir", default=str(SCRIPT_DIR / "../../checkpoints/masked_vae"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--mask_prob", type=float, default=0.3,
                        help="Per-sample, per-modality masking probability (paper default 0.3)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    transform = ConcatTransform()
    train_ds = RealTrainDataset(args.data_dir, split="train", transform=transform, seed=args.seed)
    val_ds = RealTrainDataset(args.data_dir, split="val", transform=transform, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = MaskedConcatVAE(
        latent_dim=args.latent_dim, in_channels=5, out_channels=5,
        mask_prob=args.mask_prob,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MaskedConcatVAE  |  parameters: {n_params/1e6:.1f} M  |  mask_prob: {args.mask_prob}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val = float("inf")
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}")

    print(f"Training on {len(train_ds)} real images, validating on {len(val_ds)}")
    print(f"Device: {device}  |  Batch: {args.batch}  |  LR: {args.lr}  |  WD: {args.weight_decay}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, training=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, training=False)
        scheduler.step()

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        print(f"Epoch {epoch+1:3d}/{args.epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "history": history,
            "args": vars(args),
        }
        torch.save(state, ckpt_dir / "latest.pth")

        if val_loss < best_val:
            best_val = val_loss
            state["best_val"] = best_val
            torch.save(state, ckpt_dir / "best.pth")
            print(f"  -> new best val loss: {best_val:.5f}")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
