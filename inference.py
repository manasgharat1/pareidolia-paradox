"""
inference.py — Main Inference & Prediction Script for The Pareidolia Paradox
=============================================================================
Competition: The Pareidolia Paradox (IEEE SIES GST)
Metric: Balanced Accuracy

Usage:
    python inference.py
    python inference.py --output submission.csv
"""

import sys
import os
import argparse

# Add src to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict_final import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ensemble predictions for The Pareidolia Paradox")
    parser.add_argument("--test_csv",      type=str,   default="data/test/test_metadata.csv")
    parser.add_argument("--test_img_dir",  type=str,   default="data/test/images")
    parser.add_argument("--ckpt_dir",      type=str,   default="checkpoints")
    parser.add_argument("--image_size",    type=int,   default=256)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--output",        type=str,   default="submission/submission.csv")
    parser.add_argument("--min_bacc",      type=float, default=0.68,
                        help="Minimum val_bacc to include a checkpoint in ensemble")
    parser.add_argument("--threshold",     type=float, default=None,
                        help="Override decision threshold (default: 0.50 canonical calibrated)")
    args = parser.parse_args()
    main(args)
