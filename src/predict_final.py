"""
predict_final.py — Final Ensemble Submission Generator
=======================================================
Combines ALL trained checkpoints (ResNet-34, ConvNeXt folds,
EfficientNet folds, ResNet-34 folds) with 4-pass TTA and
calibrated decision thresholding for maximum Balanced Accuracy.

Usage:
    python src/predict_final.py --output submission.csv

The script auto-discovers all checkpoint files in the checkpoints/
directory and weights them by their saved val_bacc score.
"""

import argparse
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import normalize_azimuth
from model import get_model


class TestDataset(Dataset):
    def __init__(self, csv_path, img_dir, image_size=256, aug_mode=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        tf = []
        if aug_mode == 'scale_up':
            tf.append(T.Resize((int(image_size * 1.05), int(image_size * 1.05))))
            tf.append(T.CenterCrop((image_size, image_size)))
        elif aug_mode == 'scale_down':
            tf.append(T.Resize((int(image_size * 0.95), int(image_size * 0.95))))
            tf.append(T.Pad(int(image_size * 0.025) + 1, padding_mode='reflect'))
            tf.append(T.CenterCrop((image_size, image_size)))
        else:
            tf.append(T.Resize((image_size, image_size)))

        if aug_mode == 'bright_up':
            tf.append(T.ColorJitter(brightness=(1.12, 1.12)))
        elif aug_mode == 'bright_down':
            tf.append(T.ColorJitter(brightness=(0.88, 0.88)))
        elif aug_mode == 'contrast_up':
            tf.append(T.ColorJitter(contrast=(1.15, 1.15)))
        elif aug_mode == 'contrast_down':
            tf.append(T.ColorJitter(contrast=(0.88, 0.88)))

        tf += [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
        self.transform = T.Compose(tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['image_id'])).convert('L')
        img = normalize_azimuth(img, row['sun_azimuth_angle'])
        return self.transform(img), row['image_id']


def calibrate_probs(probs, threshold):
    """
    Piecewise-linear probability calibration:
    Maps each model's native optimal threshold `threshold` to the canonical 0.50.
    This eliminates decision-boundary distortion when ensembling models with different thresholds.
    """
    th = max(0.05, min(0.95, float(threshold)))
    calibrated = np.where(probs <= th,
                          0.5 * (probs / th),
                          0.5 + 0.5 * ((probs - th) / (1.0 - th)))
    return np.clip(calibrated, 0.0, 1.0)


@torch.no_grad()
def get_tta_probs(model, csv_path, img_dir, image_size, device, batch_size=32):
    tta_modes = [None, 'bright_up', 'bright_down', 'contrast_up', 'contrast_down', 'scale_up', 'scale_down']
    all_probs, filenames = None, None
    for mode in tta_modes:
        ds = TestDataset(csv_path, img_dir, image_size=image_size, aug_mode=mode)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        probs_list, fname_list = [], []
        for imgs, fnames in loader:
            imgs = imgs.to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                out = model(imgs)
            probs_list.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            fname_list.extend(fnames)
        probs_arr = np.array(probs_list)
        if all_probs is None:
            all_probs, filenames = probs_arr, fname_list
        else:
            all_probs += probs_arr
    return all_probs / len(tta_modes), filenames


def discover_checkpoints(ckpt_dir):
    """Auto-discover all .pt checkpoint files and their metadata."""
    checkpoints = []
    for pt_file in glob.glob(os.path.join(ckpt_dir, '*.pt')):
        if '.gitkeep' in pt_file:
            continue
        try:
            c = torch.load(pt_file, map_location='cpu')
            if isinstance(c, dict) and 'model_state_dict' in c:
                model_name = c.get('model_name', None)
                bacc = c.get('best_val_bacc', 0.5)
                th = c.get('best_threshold', 0.5)
                epoch = c.get('epoch', 0)
                fold = c.get('fold', 0)
                if model_name:
                    checkpoints.append({
                        'path': pt_file,
                        'model_name': model_name,
                        'bacc': bacc,
                        'threshold': th,
                        'epoch': epoch,
                        'fold': fold,
                        'filename': os.path.basename(pt_file),
                    })
        except Exception as e:
            print(f'  Skipping {os.path.basename(pt_file)}: {e}')
    return checkpoints


def main(args):
    test_csv = os.path.join(BASE_DIR, args.test_csv)
    img_dir  = os.path.join(BASE_DIR, args.test_img_dir)
    ckpt_dir = os.path.join(BASE_DIR, args.ckpt_dir)
    out_path = os.path.join(BASE_DIR, args.output)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Discover checkpoints ──
    checkpoints = discover_checkpoints(ckpt_dir)
    if not checkpoints:
        print('No valid checkpoints found! Run train.py or train_v2.py first.')
        return

    print(f'\nDiscovered {len(checkpoints)} checkpoints:')
    for c in sorted(checkpoints, key=lambda x: x['bacc'], reverse=True):
        print(f"  {c['filename']:40s}  model={c['model_name']:<14s}  "
              f"val_bacc={c['bacc']:.4f}  fold={c['fold']}  epoch={c['epoch']}")

    # ── Filter: only keep checkpoints above minimum threshold ──
    valid = [c for c in checkpoints if c['bacc'] >= args.min_bacc]
    if not valid:
        valid = sorted(checkpoints, key=lambda x: x['bacc'], reverse=True)[:3]
    print(f'\nUsing {len(valid)} checkpoints (min_bacc={args.min_bacc})')

    # ── Check for mathematically optimal ensemble weights ──
    optimal_weights_map = {}
    optimal_th = None
    opt_json_path = os.path.join(ckpt_dir, "optimal_ensemble_weights.json")
    if os.path.exists(opt_json_path):
        try:
            with open(opt_json_path, 'r') as f:
                opt_info = json.load(f)
            for c_name, w_val in zip(opt_info['checkpoint_names'], opt_info['weights']):
                optimal_weights_map[c_name] = w_val
            optimal_th = opt_info.get('optimal_threshold', None)
            print(f"  [OPT] Loaded mathematically optimal weights from {os.path.basename(opt_json_path)}")
        except Exception as e:
            print(f"  [OPT] Could not load optimal weights JSON: {e}")

    # ── Ensemble inference ──
    all_weighted_probs = None
    total_weight = 0.0
    filenames = None

    for c in valid:
        print(f"\n  Inferring: {c['filename']} (bacc={c['bacc']:.4f}) ...", end='', flush=True)
        model = get_model(c['model_name'], num_classes=2).to(device)
        ckpt = torch.load(c['path'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        probs, fnames = get_tta_probs(model, test_csv, img_dir,
                                      args.image_size, device, args.batch_size)
        probs_calibrated = calibrate_probs(probs, c['threshold'])
        
        # Use mathematically optimal weight if available, else bacc^2
        if c['filename'] in optimal_weights_map:
            weight = optimal_weights_map[c['filename']]
        else:
            weight = c['bacc'] ** 2
            
        if all_weighted_probs is None:
            all_weighted_probs = probs_calibrated * weight
            filenames = fnames
        else:
            all_weighted_probs += probs_calibrated * weight
        total_weight += weight
        del model
        torch.cuda.empty_cache()
        print(f' done (th={c["threshold"]:.2f} -> calibrated to 0.50, weight={weight:.4f})')

    ensemble_probs = all_weighted_probs / total_weight

    # Decision threshold:
    if args.threshold is not None:
        final_th = args.threshold
    elif optimal_th is not None:
        final_th = optimal_th
    else:
        final_th = 0.500
    print(f'\nFinal decision threshold: {final_th:.3f}')

    preds = (ensemble_probs >= final_th).astype(int)
    submission = pd.DataFrame({'image_id': filenames, 'label': preds})

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    submission.to_csv(out_path, index=False)

    # Also save a copy to root submission.csv for backward compatibility
    root_sub = os.path.join(BASE_DIR, 'submission.csv')
    if os.path.abspath(out_path) != os.path.abspath(root_sub):
        submission.to_csv(root_sub, index=False)

    print(f'\nSaved {len(submission)} predictions to {out_path} and {root_sub}')
    print('Class distribution:')
    print(submission['label'].value_counts().to_dict())
    print('\nFirst 10 rows:')
    print(submission.head(10))

    # Sanity checks
    assert len(submission) == 2000, f'Expected 2000 rows, got {len(submission)}'
    assert list(submission.columns) == ['image_id', 'label'], 'Wrong columns'
    assert submission['label'].isnull().sum() == 0, 'NaN labels found'
    assert set(submission['label'].unique()).issubset({0, 1}), 'Labels must be 0 or 1'
    print('\n[OK] All sanity checks passed. Submission is ready!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv',      type=str,   default='data/test/test_metadata.csv')
    parser.add_argument('--test_img_dir',  type=str,   default='data/test/images')
    parser.add_argument('--ckpt_dir',      type=str,   default='checkpoints')
    parser.add_argument('--image_size',    type=int,   default=256)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--output',        type=str,   default='submission/submission.csv')
    parser.add_argument('--min_bacc',      type=float, default=0.69,
                        help='Minimum val_bacc to include a checkpoint in ensemble')
    parser.add_argument('--threshold',     type=float, default=None,
                        help='Override decision threshold (default: auto from checkpoints)')
    args = parser.parse_args()
    main(args)
