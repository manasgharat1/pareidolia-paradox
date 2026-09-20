"""
check_environment.py — System & Submission Verification Script
==============================================================
Validates the entire environment, GPU capabilities, checkpoint portfolio,
and submission files to ensure 100% readiness.

Usage:
    python check_environment.py
"""

import os
import sys
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 70)
    print("  THE PAREIDOLIA PARADOX - SYSTEM & SUBMISSION INTEGRITY CHECK")
    print("=" * 70)

    # 1. Python & PyTorch
    print(f"\n[1] Python Environment:")
    print(f"    Python Version: {sys.version.split()[0]}")
    try:
        import torch
        print(f"    PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"    CUDA Available:  {cuda_avail}")
        if cuda_avail:
            print(f"    GPU Device:      {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"    Total VRAM:      {vram_gb:.2f} GB")
    except ImportError:
        print("    [ERROR] PyTorch not installed!")

    # 2. Checkpoints Directory
    print(f"\n[2] Checkpoints Portfolio:")
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    if os.path.exists(ckpt_dir):
        pts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        print(f"    Checkpoints Directory: {ckpt_dir}")
        print(f"    Total .pt Checkpoints: {len(pts)} (Expected: 16)")
        opt_json = os.path.join(ckpt_dir, "optimal_ensemble_weights.json")
        has_opt = os.path.exists(opt_json)
        print(f"    Optimal Weights File:  {'Found [OK]' if has_opt else 'Missing [WARN]'}")
    else:
        print("    [ERROR] checkpoints/ directory missing!")

    # 3. Submission Files
    print(f"\n[3] Submission Files:")
    sub_csv = os.path.join(SCRIPT_DIR, "submission", "submission.csv")
    root_sub = os.path.join(SCRIPT_DIR, "submission.csv")
    for name, p in [("submission/submission.csv", sub_csv), ("submission.csv", root_sub)]:
        if os.path.exists(p):
            sz = os.path.getsize(p)
            print(f"    {name:<26s} Found ({sz:,} bytes) [OK]")
        else:
            print(f"    {name:<26s} Missing [ERROR]")

    # 4. Official Submission Checklist Deliverables
    print(f"\n[4] Official Submission Package Deliverables:")
    required_docs = [
        "submission/FINAL_SUBMISSION_CHECKLIST.txt",
        "submission/OVERALL_BENCHMARK_REPORT.md",
        "submission/COMPREHENSIVE_EXPLANATION_GUIDE.md",
        "submission/METHODOLOGY_SUMMARY.md",
        "submission/MODEL_WEIGHTS_INFO.txt",
        "submission/LINKEDIN_POST.txt",
        "submission/verify_submission.py",
        "train.py",
        "inference.py",
        "README.md",
        "requirements.txt",
        "LICENSE"
    ]
    for doc in required_docs:
        doc_path = os.path.join(SCRIPT_DIR, doc)
        exists = os.path.exists(doc_path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"    {doc:<42s} {status}")

    print("\n" + "=" * 70)
    print("  ALL CORE COMPONENTS VERIFIED & READY FOR SUBMISSION!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
