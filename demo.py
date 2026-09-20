"""
demo.py — Interactive Prediction & Visualization Demo
=====================================================
Allows evaluators and judges to test single lunar patches with the ensemble
and inspect probability breakdowns, sun angle normalization, and predicted terrain.

Usage:
    python demo.py
    python demo.py --image data/test/images/eval_00001.png --azimuth 158.9
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.dataset import normalize_azimuth
from src.model import get_model


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  THE PAREIDOLIA PARADOX - INTERACTIVE DEMONSTRATOR")
    print("=" * 65)
    print(f"Device: {device}")

    # Determine image path and azimuth
    img_path = args.image
    azimuth = args.azimuth

    if img_path is None:
        # Pick the first test image by default
        test_meta = os.path.join(SCRIPT_DIR, "data", "test", "test_metadata.csv")
        if os.path.exists(test_meta):
            df = pd.read_csv(test_meta)
            sample = df.iloc[0]
            img_path = os.path.join(SCRIPT_DIR, "data", "test", "images", sample["image_id"])
            azimuth = float(sample["sun_azimuth_angle"])
            img_name = sample["image_id"]
        else:
            print("[ERROR] No image specified and test metadata not found!")
            return
    else:
        img_name = os.path.basename(img_path)

    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        return

    print(f"\nEvaluating Image:     {img_name}")
    print(f"Solar Azimuth Angle:  {azimuth:.2f} deg")

    # Load and normalize image
    img = Image.open(img_path).convert("L")
    norm_img = normalize_azimuth(img, azimuth)

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(norm_img).unsqueeze(0).to(device)

    # Load top representative models
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    models_to_test = [
        ("best_efficientnet_fold1.pt", "efficientnet"),
        ("best_resnet34_fold3.pt", "resnet34"),
        ("best_efficientnet_v2_s_pseudo_fold1.pt", "efficientnet_v2_s"),
        ("best_convnext_fold1.pt", "convnext"),
        ("best_densenet121_fold1.pt", "densenet121")
    ]

    print("\n--- Multi-Model Prediction Breakdown ---")
    probs = []
    for fname, mname in models_to_test:
        fpath = os.path.join(ckpt_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            model = get_model(mname, num_classes=2).to(device)
            ckpt = torch.load(fpath, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            with torch.no_grad():
                out = model(tensor)
                p = torch.softmax(out, dim=1)[0, 1].item()
                probs.append(p)
                decision = "Rise (Mound)" if p >= 0.50 else "Depth (Crater)"
                print(f"  {fname:<40s} | P(Rise): {p*100:5.1f}% | -> {decision}")
        except Exception as e:
            pass

    if probs:
        ensemble_prob = float(np.mean(probs))
        final_class = 1 if ensemble_prob >= 0.527 else 0
        final_label = "RISE / MOUND (Positive Relief)" if final_class == 1 else "DEPTH / CRATER (Negative Relief)"
        confidence = ensemble_prob if final_class == 1 else (1.0 - ensemble_prob)

        print("\n" + "=" * 65)
        print(f"FINAL PREDICTION:  Class {final_class} -> {final_label}")
        print(f"Ensemble Confidence: {confidence*100:.2f}%")
        print(f"Ensemble P(Rise):    {ensemble_prob*100:.2f}% (Threshold = 52.7%)")
        print("=" * 65 + "\n")
    else:
        print("[WARNING] No checkpoints could be loaded. Please ensure weights are in checkpoints/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo prediction on single lunar imagery")
    parser.add_argument("--image",   type=str,   default=None, help="Path to lunar PNG image")
    parser.add_argument("--azimuth", type=float, default=180.0, help="Sun azimuth angle in degrees")
    args = parser.parse_args()
    run_demo(args)
