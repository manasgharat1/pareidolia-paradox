"""
train.py
========
Trains the Depth vs Rise classifier and tracks BALANCED ACCURACY
(the competition scoring metric).

Enhanced according to analysis_plan.txt:
  - Supports ResNet34 with deep hierarchical classifier
  - Differential learning rates (backbone vs classifier)
  - Linear warmup + Cosine Annealing scheduler
  - Label smoothing cross-entropy loss with class reweighting
  - Optimal threshold calibration for Balanced Accuracy
  - Fast AMP FP16 training on CUDA
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

# Ensure src is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import LunarDataset, CLASS_NAMES
from model import get_model


def compute_class_weights(df, num_classes=2):
    counts = df["label"].value_counts()
    total = counts.sum()
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss, preds, labels_all = 0.0, [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds.extend(outputs.argmax(1).detach().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    bacc = balanced_accuracy_score(labels_all, preds)
    return avg_loss, bacc


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, all_probs, labels_all = 0.0, [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.array(all_probs)
    labels_all = np.array(labels_all)

    # Standard metrics at default 0.5 threshold
    preds_default = (all_probs >= 0.5).astype(int)
    default_bacc = balanced_accuracy_score(labels_all, preds_default)
    acc = accuracy_score(labels_all, preds_default)
    cm = confusion_matrix(labels_all, preds_default)

    # Calibrate decision threshold to maximize Balanced Accuracy
    best_th = 0.5
    best_bacc = default_bacc
    for th in np.linspace(0.30, 0.70, 41):
        p = (all_probs >= th).astype(int)
        score = balanced_accuracy_score(labels_all, p)
        if score > best_bacc:
            best_bacc = score
            best_th = float(th)

    return avg_loss, default_bacc, best_bacc, best_th, acc, cm


def main(args):
    # Robust path resolution
    base_dir = os.path.dirname(SCRIPT_DIR)
    train_csv = os.path.abspath(args.train_csv if os.path.isabs(args.train_csv) else os.path.join(base_dir, args.train_csv))
    img_dir = os.path.abspath(args.img_dir if os.path.isabs(args.img_dir) else os.path.join(base_dir, args.img_dir))
    checkpoint_path = os.path.abspath(args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(base_dir, args.checkpoint))
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    # Data split handling: check if pre-split CSVs exist in SCRIPT_DIR
    train_split_path = os.path.join(SCRIPT_DIR, "train_split.csv")
    val_split_path = os.path.join(SCRIPT_DIR, "val_split.csv")

    if os.path.exists(train_split_path) and os.path.exists(val_split_path):
        print(f"Using existing split from {train_split_path} and {val_split_path}")
        train_df = pd.read_csv(train_split_path)
        val_df = pd.read_csv(val_split_path)
        full_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        print(f"Loading and splitting dataset from {train_csv}...")
        full_df = pd.read_csv(train_csv)
        train_df, val_df = train_test_split(
            full_df, test_size=0.2, stratify=full_df["label"], random_state=42
        )
        train_df.to_csv(train_split_path, index=False)
        val_df.to_csv(val_split_path, index=False)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    print("Label distribution in train:\n", train_df["label"].value_counts())

    train_ds = LunarDataset(train_split_path, img_dir, image_size=args.image_size, train=True)
    val_ds = LunarDataset(val_split_path, img_dir, image_size=args.image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = get_model(args.model, num_classes=2).to(device)
    print(f"Model architecture: {args.model}")

    class_weights = compute_class_weights(train_df).to(device)
    print(f"Class weights (Depth, Rise): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # Optimizer with differential learning rates for backbone vs classifier
    if hasattr(model, "backbone") and hasattr(model, "classifier"):
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr * 0.3},
            {"params": model.classifier.parameters(), "lr": args.lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup + Cosine Annealing schedule
    warmup_epochs = min(3, max(1, args.epochs // 10))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    best_val_bacc = 0.0
    best_opt_th = 0.5
    patience_counter = 0

    print(f"\n--- Starting Training for {args.epochs} epochs ---")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_bacc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_loss, default_bacc, best_bacc, best_th, val_acc, cm = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"Epoch {epoch+1:02d}/{args.epochs} ({elapsed:.1f}s, lr={current_lr:.2e}) | "
              f"train_loss={train_loss:.4f} train_bacc={train_bacc:.4f} | "
              f"val_loss={val_loss:.4f} val_bacc={default_bacc:.4f} (opt_bacc={best_bacc:.4f} @ th={best_th:.2f})")

        # Track the best calibrated Balanced Accuracy
        target_bacc = max(default_bacc, best_bacc)
        if target_bacc > best_val_bacc:
            best_val_bacc = target_bacc
            best_opt_th = best_th if best_bacc > default_bacc else 0.5
            patience_counter = 0
            save_payload = {
                "model_state_dict": model.state_dict(),
                "best_val_bacc": best_val_bacc,
                "best_threshold": best_opt_th,
                "model_name": args.model,
                "epoch": epoch + 1,
            }
            torch.save(save_payload, checkpoint_path)
            print(f"  -> [BEST] Checkpoint saved! val_balanced_acc={best_val_bacc:.4f} @ threshold={best_opt_th:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\n=======================================================")
    print(f"Training Complete! Best validation Balanced Accuracy: {best_val_bacc:.4f} (optimal threshold: {best_opt_th:.2f})")
    print(f"Saved checkpoint: {checkpoint_path}")
    print("Final evaluation confusion matrix (rows=true, cols=pred):", CLASS_NAMES)
    print(cm)
    print("=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="data/train/train_metadata.csv")
    parser.add_argument("--img_dir", type=str, default="data/train/images")
    parser.add_argument("--model", type=str, default="resnet34", choices=["simple", "resnet18", "resnet34", "resnet"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_resnet34.pt")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    args = parser.parse_args()
    main(args)