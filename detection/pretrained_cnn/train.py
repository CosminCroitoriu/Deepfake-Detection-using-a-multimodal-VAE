#!/usr/bin/env python3
"""
Train a pretrained CNN binary classifier (real vs. fake).

Supported architectures (--arch):
  resnet50        — 23.5 M params
  efficientnet_b0 —  4.0 M params

Both are fine-tuned end-to-end from ImageNet pretrained weights.
Checkpoints are saved to checkpoints/{arch}/.

Pretrained models converge much faster than the from-scratch ViT:
20 epochs is typically sufficient (vs. 30 for the ViT).
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ClassifierTrainDataset, train_transform, eval_transform
from .model import PretrainedCNNClassifier

SCRIPT_DIR = Path(__file__).resolve().parent


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    correct = 0
    total_n = 0
    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.float().to(device)
            logits = model(imgs)                          # [B]
            loss   = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = (logits.detach() >= 0.0).long()
            correct += (preds == labels.long()).sum().item()
            total_n += imgs.size(0)
    return total_loss / max(total_n, 1), correct / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arch",         default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--data_dir",     default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--ckpt_dir",     default=None,
                        help="Checkpoint dir (default: checkpoints/{arch})")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch",        type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--resume",       default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir or (SCRIPT_DIR / f"../../checkpoints/{args.arch}"))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ClassifierTrainDataset(
        args.data_dir, split="train", seed=args.seed, transform=train_transform
    )
    val_ds = ClassifierTrainDataset(
        args.data_dir, split="val",   seed=args.seed, transform=eval_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = PretrainedCNNClassifier(arch=args.arch, pretrained=True).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch}  |  parameters: {n_params/1e6:.2f} M  (pretrained)")
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"Device: {device}  |  Batch: {args.batch}  |  LR: {args.lr}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 0
    best_val_acc = 0.0
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history      = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, False)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
        })
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        state = {
            "epoch": epoch,
            "arch":  args.arch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
            "args": vars(args),
        }
        torch.save(state, ckpt_dir / "latest.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state["best_val_acc"] = best_val_acc
            torch.save(state, ckpt_dir / "best.pth")
            print(f"  -> new best val accuracy: {best_val_acc:.4f}")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
