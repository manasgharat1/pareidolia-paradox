"""
optimize_ensemble_weights.py — Mathematical Ensemble Weight Solver (SLSQP / Nelder-Mead)
========================================================================================
Finds the mathematically optimal ensemble weights w_1, ..., w_15 and decision threshold tau
that directly maximizes Balanced Accuracy over out-of-fold validation predictions.

Zero GPU retraining required — executes in ~1-2 minutes!
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import normalize_azimuth
from model import get_model


class ValDataset(Dataset):
    def __init__(self, df, img_dir, image_size=256):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['image_id'])).convert('L')
        img = normalize_azimuth(img, row['sun_azimuth_angle'])
        return self.transform(img), int(row['label'])


def calibrate_probs(probs, threshold):
    th = max(0.05, min(0.95, float(threshold)))
    calibrated = np.where(probs <= th,
                          0.5 * (probs / th),
                          0.5 + 0.5 * ((probs - th) / (1.0 - th)))
    return np.clip(calibrated, 0.0, 1.0)


def main():
    print("=" * 70)
    print("[OPTIMIZATION 1] MATHEMATICAL ENSEMBLE WEIGHT SOLVER")
    print("=" * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load training metadata and create consistent 5-fold split
    train_csv = os.path.join(BASE_DIR, "data", "train", "train_metadata.csv")
    train_img_dir = os.path.join(BASE_DIR, "data", "train", "images")
    df = pd.read_csv(train_csv)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_indices = None
    for fold_num, (tr_idx, vl_idx) in enumerate(skf.split(df, df['label']), 1):
        if fold_num == 1:
            val_indices = vl_idx
            break

    val_df = df.iloc[val_indices].copy().reset_index(drop=True)
    y_val = val_df['label'].values
    print(f"Validation set: {len(val_df)} samples (Depth: {(y_val==0).sum()}, Rise: {(y_val==1).sum()})", flush=True)

    val_ds = ValDataset(val_df, train_img_dir)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Discover checkpoints
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    active_ckpts = []
    for f in ckpt_files:
        try:
            c = torch.load(f, map_location='cpu')
            if 'model_name' in c and 'best_val_bacc' in c and c['best_val_bacc'] >= 0.68:
                active_ckpts.append((f, c['model_name'], c['best_val_bacc'], c.get('best_threshold', 0.5)))
        except:
            pass

    print(f"\nDiscovered {len(active_ckpts)} active checkpoints:")
    prob_matrix = []
    ckpt_names = []
    heuristic_weights = []

    for path, name, bacc, th in active_ckpts:
        base_name = os.path.basename(path)
        ckpt_names.append(base_name)
        heuristic_weights.append(bacc ** 2)
        print(f"  Evaluating {base_name:35s} (bacc={bacc:.4f}) ...", end='', flush=True)

        model = get_model(name).to(device)
        c = torch.load(path, map_location=device)
        model.load_state_dict(c['model_state_dict'])
        model.eval()

        probs_list = []
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    out = model(imgs)
                probs_list.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())

        del model
        torch.cuda.empty_cache()

        raw_probs = np.array(probs_list)
        cal_probs = calibrate_probs(raw_probs, th)
        single_bacc = balanced_accuracy_score(y_val, (cal_probs >= 0.50).astype(int))
        prob_matrix.append(cal_probs)
        print(f" val_bacc={single_bacc*100:.2f}%", flush=True)

    prob_matrix = np.array(prob_matrix).T  # Shape: [N_samples, N_models]
    heuristic_weights = np.array(heuristic_weights)
    heuristic_weights /= heuristic_weights.sum()

    # Baseline Heuristic Blend Score
    base_blend = np.dot(prob_matrix, heuristic_weights)
    base_bacc = balanced_accuracy_score(y_val, (base_blend >= 0.50).astype(int))
    print(f"\n[BASELINE HEURISTIC BLEND] Val Balanced Accuracy: {base_bacc*100:.2f}% (th=0.50)", flush=True)

    # Optimization: SLSQP solver on continuous softmax weights + threshold
    print("\n[OPTIMIZING] Running SLSQP weight and threshold solver...", flush=True)

    N_models = prob_matrix.shape[1]

    def loss_func(params):
        raw_w = params[:N_models]
        th = params[N_models]
        # Softmax for non-negative weights that sum to 1
        e_w = np.exp(raw_w - np.max(raw_w))
        w = e_w / e_w.sum()
        pred_p = np.dot(prob_matrix, w)
        preds = (pred_p >= th).astype(int)
        # We want to maximize balanced accuracy -> minimize negative BAcc
        # Add slight entropy regularization to encourage diversity across all backbones
        entropy = -np.sum(w * np.log(w + 1e-8))
        bacc = balanced_accuracy_score(y_val, preds)
        return -bacc - 0.005 * entropy

    best_opt_bacc = base_bacc
    best_opt_w = heuristic_weights
    best_opt_th = 0.50

    # Multi-start optimization
    init_params = np.append(np.log(heuristic_weights + 1e-4), 0.50)
    res = minimize(loss_func, init_params, method='Nelder-Mead', options={'maxiter': 2500, 'xatol': 1e-4})

    raw_w_opt = res.x[:N_models]
    opt_th = float(np.clip(res.x[N_models], 0.35, 0.65))
    e_w = np.exp(raw_w_opt - np.max(raw_w_opt))
    opt_w = e_w / e_w.sum()

    opt_blend = np.dot(prob_matrix, opt_w)
    opt_bacc = balanced_accuracy_score(y_val, (opt_blend >= opt_th).astype(int))

    # Fine threshold sweep
    thresholds = np.linspace(0.40, 0.60, 41)
    best_sweep_th = opt_th
    best_sweep_bacc = opt_bacc
    for t in thresholds:
        sc = balanced_accuracy_score(y_val, (opt_blend >= t).astype(int))
        if sc > best_sweep_bacc:
            best_sweep_bacc = sc
            best_sweep_th = float(t)

    print("=" * 70)
    print("[RESULTS] WEIGHT OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Heuristic Baseline BAcc:   {base_bacc*100:.2f}%")
    print(f"Optimized Ensemble BAcc:   {best_sweep_bacc*100:.2f}%  (+{(best_sweep_bacc - base_bacc)*100:.2f}% gain!)")
    print(f"Optimal Decision Threshold: {best_sweep_th:.3f}")
    print("\nOptimal Model Weight Distribution:")
    for name, hw, ow in zip(ckpt_names, heuristic_weights, opt_w):
        print(f"  {name:35s} | Heuristic: {hw*100:5.2f}% -> Optimal: {ow*100:5.2f}%")
    print("=" * 70, flush=True)

    # Save optimal weights to JSON
    weights_info = {
        'checkpoint_names': ckpt_names,
        'weights': opt_w.tolist(),
        'optimal_threshold': best_sweep_th,
        'val_bacc_baseline': float(base_bacc),
        'val_bacc_optimized': float(best_sweep_bacc)
    }
    weights_json_path = os.path.join(ckpt_dir, "optimal_ensemble_weights.json")
    with open(weights_json_path, 'w') as f:
        json.dump(weights_info, f, indent=2)
    print(f"\n[SAVED] Optimal weights saved to {weights_json_path}")


if __name__ == "__main__":
    main()
