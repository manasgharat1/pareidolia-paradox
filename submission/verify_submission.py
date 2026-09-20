"""
verify_submission.py
====================
Standalone verification script to validate competition submission files.
Ensures 100% compliance with The Pareidolia Paradox competition requirements.

Usage:
    python verify_submission.py [optional_path_to_submission.csv]
"""

import os
import sys
import pandas as pd

def verify(sub_path="submission.csv", test_meta_path="../data/test/test_metadata.csv"):
    if not os.path.exists(sub_path):
        # Try local folder
        alt_path = os.path.join(os.path.dirname(__file__), sub_path)
        if os.path.exists(alt_path):
            sub_path = alt_path
        else:
            print(f"[FAIL] Submission file not found: {sub_path}")
            return False

    print(f"--> Validating submission: {sub_path}")
    df = pd.read_csv(sub_path)

    # 1. Check Row Count
    expected_rows = 2000
    if len(df) != expected_rows:
        print(f"[FAIL] Row count mismatch: expected {expected_rows}, got {len(df)}")
        return False
    print(f"[PASS] Row count: {len(df)} (exactly {expected_rows})")

    # 2. Check Column Names
    expected_cols = ["image_id", "label"]
    if list(df.columns) != expected_cols:
        print(f"[FAIL] Column names mismatch: expected {expected_cols}, got {list(df.columns)}")
        return False
    print(f"[PASS] Columns: {list(df.columns)}")

    # 3. Check for NaNs / Nulls
    null_counts = df.isnull().sum().to_dict()
    if any(count > 0 for count in null_counts.values()):
        print(f"[FAIL] Missing / NaN values detected: {null_counts}")
        return False
    print("[PASS] Missing values: 0")

    # 4. Check Label Values
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        print(f"[FAIL] Invalid label values found: {unique_labels}. Expected only {0, 1}")
        return False
    print(f"[PASS] Label values valid: {unique_labels}")

    # 5. Check Class Distribution
    counts = df["label"].value_counts().to_dict()
    print(f"[INFO] Class Distribution: Depth (0): {counts.get(0, 0)} ({counts.get(0, 0)/len(df)*100:.1f}%), Rise (1): {counts.get(1, 0)} ({counts.get(1, 0)/len(df)*100:.1f}%)")

    # 6. Check Matching Test IDs if metadata available
    if not os.path.exists(test_meta_path):
        alt_meta = os.path.join(os.path.dirname(__file__), test_meta_path)
        if os.path.exists(alt_meta):
            test_meta_path = alt_meta

    if os.path.exists(test_meta_path):
        meta = pd.read_csv(test_meta_path)
        meta_ids = set(meta["image_id"])
        sub_ids = set(df["image_id"])
        if meta_ids != sub_ids:
            print(f"[FAIL] Image IDs do not exactly match test_metadata.csv!")
            diff = meta_ids.symmetric_difference(sub_ids)
            print(f"       Difference count: {len(diff)}")
            return False
        print(f"[PASS] Image IDs: 100% exact match with official test metadata ({len(meta_ids)} IDs)")

    print("\n==========================================")
    print(" ALL CHECKS PASSED -- 100% COMPETITION READY!")
    print("==========================================")
    return True

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "submission.csv")
    success = verify(path)
    sys.exit(0 if success else 1)
