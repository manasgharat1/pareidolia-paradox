"""
train_v2.py  — Advanced Training Pipeline for The Pareidolia Paradox
=====================================================================
Implements:
  1. K-Fold Stratified Cross-Validation (uses ALL 7,854 training images)
  2. Pseudo-labeling on high-confidence test set images
  3. Focal Loss (concentrates on hard, ambiguous terrain samples)
  4. Warmup + Cosine Annealing LR schedule
  5. Differential Learning Rates (backbone vs classifier)
  6. AMP FP16 for fast GPU training
  7. Calibrated decision threshold per fold
  8. Saves each fold's best checkpoint → used in final ensemble

Run from the project root directory:
    .\\venv\\Scripts\\python.exe src/train_v2.py --model convnext --folds 5 --epochs 30
    .\\venv\\Scripts\\python.exe src/train_v2.py --model efficientnet --folds 5 --epochs 30
    .\\venv\\Scripts\\python.exe src/train_v2.py --model resnet34 --folds 5 --epochs 30
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from PIL import Image
import torchvision.transforms as T

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import CLASS_NAMES, normalize_azimuth
from model import get_model


# ──────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


# ──────────────────────────────────────────────────────
# Dataset with augmentations
# ──────────────────────────────────────────────────────
def build_transforms(image_size, train=True):
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.7),
            T.RandomAffine(degrees=0, translate=(0.07, 0.07), scale=(0.93, 1.07)),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
            T.RandomErasing(p=0.35, scale=(0.02, 0.18), ratio=(0.5, 2.0), value=0),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])


class LunarDatasetV2(Dataset):
    def __init__(self, df, img_dir, transform, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Support per-row img_dir column (for mixing train + pseudo-labeled test images)
        img_dir = row['img_dir'] if 'img_dir' in row.index else self.img_dir
        img = Image.open(os.path.join(img_dir, row['image_id'])).convert('L')
        img = normalize_azimuth(img, row['sun_azimuth_angle'])
        tensor = self.transform(img)
        if self.has_labels:
            return tensor, torch.tensor(int(row['label']), dtype=torch.long)
        return tensor, row['image_id']


# ──────────────────────────────────────────────────────
# Training / Evaluation
# ──────────────────────────────────────────────────────
def compute_class_weights(df, num_classes=2, device='cpu'):
    counts = df['label'].value_counts()
    total = counts.sum()
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        preds_all.extend(out.argmax(1).detach().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), balanced_accuracy_score(labels_all, preds_all)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, probs_all, labels_all = 0.0, [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        probs_all.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    probs = np.array(probs_all)
    targets = np.array(labels_all)
    preds05 = (probs >= 0.5).astype(int)
    bacc_05 = balanced_accuracy_score(targets, preds05)
    acc = accuracy_score(targets, preds05)
    cm = confusion_matrix(targets, preds05)
    # threshold sweep
    best_th, best_bacc = 0.5, bacc_05
    for th in np.linspace(0.30, 0.70, 41):
        p = (probs >= th).astype(int)
        b = balanced_accuracy_score(targets, p)
        if b > best_bacc:
            best_bacc, best_th = b, float(th)
    return total_loss / len(loader.dataset), bacc_05, best_bacc, best_th, acc, cm


@torch.no_grad()
def infer_probs(model, loader, device, use_amp):
    model.eval()
    probs_all, ids_all = [], []
    for imgs, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
        probs_all.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
        if isinstance(ids[0], str):
            ids_all.extend(ids)
        else:
            ids_all.extend(ids.numpy().tolist())
    return np.array(probs_all), ids_all


# ──────────────────────────────────────────────────────
# Main: K-Fold + optional pseudo-labeling
# ──────────────────────────────────────────────────────
def main(args):
    train_csv = os.path.join(BASE_DIR, args.train_csv)
    train_img_dir = os.path.join(BASE_DIR, args.train_img_dir)
    test_csv = os.path.join(BASE_DIR, args.test_csv)
    test_img_dir = os.path.join(BASE_DIR, args.test_img_dir)
    ckpt_dir = os.path.join(BASE_DIR, args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    use_amp = device.type == 'cuda' and not args.no_amp
    print(f'AMP: {"enabled" if use_amp else "disabled"}')

    full_df = pd.read_csv(train_csv)
    print(f'Training data: {len(full_df)} images, label dist:\n{full_df["label"].value_counts().to_dict()}')

    # ── Optional pseudo-labeling ──
    pseudo_df_aug = None
    if args.pseudo_csv and os.path.exists(args.pseudo_csv):
        print(f'\n=== PHASE 1: Loading precomputed pseudo-labels from {args.pseudo_csv} ===')
        pseudo_df = pd.read_csv(args.pseudo_csv)
        if 'img_dir' not in pseudo_df.columns:
            pseudo_df['img_dir'] = test_img_dir
        pseudo_df_aug = pseudo_df[['image_id', 'sun_azimuth_angle', 'label', 'img_dir']].copy()
        full_df['img_dir'] = train_img_dir
        full_df_augmented = pd.concat([full_df, pseudo_df_aug], ignore_index=True)
        print(f'  Loaded {len(pseudo_df_aug)} pseudo-labeled test images: '
              f'{(pseudo_df_aug["label"]==0).sum()} Depth, {(pseudo_df_aug["label"]==1).sum()} Rise')
        print(f'  Augmented training set: {len(full_df_augmented)} images')
    elif args.pseudo_label:
        print('\n=== PHASE 1: Generating pseudo-labels for test set ===')
        # Use the best available checkpoint in ckpt_dir
        candidate_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')), reverse=True)
        ckpt_path = candidate_ckpts[0] if candidate_ckpts else None
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f'  Pseudo-label checkpoint not found, skipping.')
        else:
            c = torch.load(ckpt_path, map_location=device)
            pl_model_name = c.get('model_name', 'resnet34')
            print(f'  Using checkpoint {os.path.basename(ckpt_path)} ({pl_model_name})')
            pl_model = get_model(pl_model_name).to(device)
            pl_model.load_state_dict(c['model_state_dict'] if 'model_state_dict' in c else c)
            pl_model.eval()
            test_meta = pd.read_csv(test_csv)
            test_tf = build_transforms(args.image_size, train=False)
            test_ds = LunarDatasetV2(test_meta, test_img_dir, test_tf, has_labels=False)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            pl_probs, pl_ids = infer_probs(pl_model, test_loader, device, use_amp)
            del pl_model
            torch.cuda.empty_cache()
            # Keep only high-confidence predictions
            conf_mask = (pl_probs >= args.pseudo_threshold) | (pl_probs <= (1 - args.pseudo_threshold))
            pl_labels = (pl_probs >= 0.5).astype(int)
            pseudo_df = pd.DataFrame({
                'image_id': pl_ids,
                'sun_azimuth_angle': test_meta['sun_azimuth_angle'].values,
                'label': pl_labels,
            })[conf_mask]
            pseudo_df['img_dir'] = test_img_dir
            print(f'  Pseudo-labeled {conf_mask.sum()} / {len(pl_probs)} test images '
                  f'(threshold={args.pseudo_threshold}): '
                  f'{(pl_labels[conf_mask]==0).sum()} Depth, {(pl_labels[conf_mask]==1).sum()} Rise')
            # Mark original train images
            full_df['img_dir'] = train_img_dir
            pseudo_df_aug = pseudo_df.copy()
            full_df_augmented = pd.concat([full_df, pseudo_df_aug], ignore_index=True)
            print(f'  Augmented training set: {len(full_df_augmented)} images '
                  f'(+{len(pseudo_df_aug)} pseudo-labeled)')
    else:
        full_df['img_dir'] = train_img_dir
        full_df_augmented = full_df

    # ── K-Fold Cross-Validation ──
    print(f'\n=== PHASE 2: {args.folds}-Fold Stratified Cross-Validation ({args.model}) ===')
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    # Only fold on original training data (label-verified)
    original_df = full_df.copy()
    X = original_df.index.values
    y = original_df['label'].values

    fold_best_baccs = []
    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        if args.single_fold is not None and fold != args.single_fold:
            continue
        print(f'\n--- Fold {fold}/{args.folds} ----------------------------')
        train_df_fold = original_df.iloc[train_idx].copy()
        val_df_fold = original_df.iloc[val_idx].copy()
        train_df_fold['img_dir'] = train_img_dir
        val_df_fold['img_dir'] = train_img_dir

        # If pseudo-labeling, append pseudo data to each fold's train
        if pseudo_df_aug is not None and len(pseudo_df_aug) > 0:
            train_df_fold = pd.concat([train_df_fold, pseudo_df_aug], ignore_index=True)
            print(f'  [Pseudo-Augmentation] Added {len(pseudo_df_aug)} test samples to Fold {fold} training set.')

        cw = compute_class_weights(train_df_fold, device=device)
        print(f'  Train: {len(train_df_fold)}, Val: {len(val_df_fold)}, class weights: {cw.tolist()}')

        train_tf = build_transforms(args.image_size, train=True)
        val_tf   = build_transforms(args.image_size, train=False)

        train_ds = LunarDatasetV2(train_df_fold, train_img_dir, train_tf)
        val_ds   = LunarDatasetV2(val_df_fold,   train_img_dir, val_tf)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=True)

        model = get_model(args.model, num_classes=2).to(device)
        criterion = FocalLoss(gamma=args.focal_gamma, weight=cw, label_smoothing=args.label_smoothing)
        scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

        # Differential LRs
        if hasattr(model, 'backbone') and hasattr(model, 'classifier'):
            param_groups = [
                {'params': model.backbone.parameters(), 'lr': args.lr * 0.25},
                {'params': model.classifier.parameters(), 'lr': args.lr},
            ]
        elif hasattr(model, 'features') and hasattr(model, 'classifier'):
            feat_params = list(model.features.parameters())
            if hasattr(model, 'norm'):
                feat_params += list(model.norm.parameters())
            param_groups = [
                {'params': feat_params, 'lr': args.lr * 0.25},
                {'params': model.classifier.parameters(), 'lr': args.lr},
            ]
        else:
            param_groups = [{'params': model.parameters(), 'lr': args.lr}]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        warmup_ep = max(2, args.epochs // 10)
        sched = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                               total_iters=warmup_ep),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=max(1, args.epochs - warmup_ep),
                                                         eta_min=1e-6),
        ], milestones=[warmup_ep])

        suffix = '_pseudo' if (args.pseudo_csv or args.pseudo_label) else ''
        ckpt_path = os.path.join(ckpt_dir, f'best_{args.model}{suffix}_fold{fold}.pt')
        best_bacc = 0.0
        best_th = 0.5
        patience = 0

        for epoch in range(args.epochs):
            t0 = time.time()
            tr_loss, tr_bacc = train_one_epoch(model, train_loader, optimizer, criterion,
                                               device, scaler, use_amp)
            vl_loss, vl_bacc, vl_opt_bacc, vl_th, vl_acc, cm = evaluate(
                model, val_loader, criterion, device, use_amp)
            sched.step()
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[-1]['lr']
            print(f'  Ep {epoch+1:02d}/{args.epochs} ({elapsed:.0f}s, lr={lr_now:.2e}) | '
                  f'tr_loss={tr_loss:.4f} tr_bacc={tr_bacc:.4f} | '
                  f'vl_bacc={vl_bacc:.4f} (opt={vl_opt_bacc:.4f}@th={vl_th:.2f})', flush=True)
            target = max(vl_bacc, vl_opt_bacc)
            if target > best_bacc:
                best_bacc = target
                best_th = vl_th if vl_opt_bacc > vl_bacc else 0.5
                patience = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_val_bacc': best_bacc,
                    'best_threshold': best_th,
                    'model_name': args.model,
                    'fold': fold,
                    'epoch': epoch + 1,
                    'image_size': args.image_size,
                }, ckpt_path)
                print(f'  -> [BEST fold {fold}] bacc={best_bacc:.4f} th={best_th:.2f} saved.', flush=True)
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print(f'  Early stop at epoch {epoch+1}')
                    break

        fold_best_baccs.append(best_bacc)
        fold_thresholds.append(best_th)
        print(f'  Fold {fold} best bacc = {best_bacc:.4f} @ th={best_th:.2f}')
        del model
        torch.cuda.empty_cache()

    print(f'\n{"="*60}')
    print(f'K-FOLD COMPLETE  ({args.model}, {args.folds} folds)')
    for i, (b, t) in enumerate(zip(fold_best_baccs, fold_thresholds), 1):
        print(f'  Fold {i}: best_bacc={b:.4f} @ th={t:.2f}')
    print(f'  Mean bacc = {np.mean(fold_best_baccs):.4f} ± {np.std(fold_best_baccs):.4f}')
    print(f'  Checkpoints saved to: {ckpt_dir}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv',      type=str,   default='data/train/train_metadata.csv')
    parser.add_argument('--train_img_dir',  type=str,   default='data/train/images')
    parser.add_argument('--test_csv',       type=str,   default='data/test/test_metadata.csv')
    parser.add_argument('--test_img_dir',   type=str,   default='data/test/images')
    parser.add_argument('--model',          type=str,   default='resnet34',
                        choices=['simple','resnet18','resnet34','densenet121','densenet','convnext','efficientnet','efficientnet_v2_s','effnet_v2'])
    parser.add_argument('--folds',          type=int,   default=5)
    parser.add_argument('--single_fold',     type=int,   default=None,
                        help='Train only a single fold (1-indexed)')
    parser.add_argument('--epochs',         type=int,   default=30)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--image_size',     type=int,   default=256)
    parser.add_argument('--lr',             type=float, default=5e-4)
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--label_smoothing',type=float, default=0.05)
    parser.add_argument('--focal_gamma',    type=float, default=2.0)
    parser.add_argument('--early_stop_patience', type=int, default=8)
    parser.add_argument('--ckpt_dir',       type=str,   default='checkpoints')
    parser.add_argument('--no_amp',         action='store_true')
    parser.add_argument('--pseudo_label',   action='store_true',
                        help='Use test-set pseudo-labels to augment training data')
    parser.add_argument('--pseudo_threshold', type=float, default=0.90,
                        help='Minimum confidence to accept pseudo-label')
    parser.add_argument('--pseudo_csv',     type=str,   default=None,
                        help='Path to precomputed high-confidence pseudo-label CSV')
    args = parser.parse_args()
    main(args)
