<div align="center">

# 🌕 The Pareidolia Paradox — Monocular Lunar Topography Classification

[![Competition](https://img.shields.io/badge/IEEE%20SIES%20GST-The%20Pareidolia%20Paradox-blue?style=for-the-badge&logo=ieee)](https://ieee.org)
[![Balanced Accuracy](https://img.shields.io/badge/Balanced%20Accuracy-86.20%25-brightgreen?style=for-the-badge&logo=target)](submission/submission.csv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA%2012.x-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![GitHub Stars](https://img.shields.io/badge/⭐%20Leave%20a%20Star-If%20Helpful!-gold?style=for-the-badge)](#-support--acknowledgements)

**Official Submission & Research Codebase for "The Pareidolia Paradox" (IEEE SIES GST)**  
*Binary Classification of Monocular Lunar Surface Imagery under Dynamic Solar Illumination*

[**Key Innovations**](#-key-engineering-innovations) • [**Benchmark Progression**](#-benchmark-progression) • [**Model Portfolio**](#-16-model-checkpoints-portfolio) • [**Interactive Demo**](#-interactive-demo-cli) • [**Quickstart**](#-quickstart-guide)

</div>

---

## 📌 Executive Summary

On the lunar surface, the absence of atmospheric diffusion causes extreme monocular shading pareidolia: optical inversions where craters (depths) appear as mounds (rises) and vice versa depending on the solar azimuth angle. 

This repository presents an **86.20% Balanced Accuracy** deep learning pipeline that mathematically resolves monocular pareidolia across **2,000 official evaluation images**. Moving beyond standard unpadded CNNs (**58.20% baseline**), this solution introduces:
1. **The 55-Pixel Reflection-Padding Theorem:** Mathematically eliminates 100% of black triangular rotation shortcuts.
2. **3-Channel Differential Physics Tensors:** Extracts real-time Sobel shadow-fall and rim-curvature gradients on GPU.
3. **5 Distinct Architectural Families (16 Checkpoints):** ResNet-34, EfficientNet-B0, EfficientNet-V2-S, ConvNeXt-Tiny, and DenseNet-121.
4. **Target-Domain Semi-Supervised Pseudo-Labeling:** Adapts feature representations to the test set's $158.9^\circ$ solar azimuth shift.
5. **SLSQP Non-Linear Weight Solver:** Directly maximizes Balanced Accuracy over out-of-fold probability matrices with optimal boundary calibration ($\tau = 0.527$).

<div align="center">
  <img src="assets/pareidolia_illusion.png" width="650" alt="The Pareidolia Paradox Illusion" />
  <p><em>Figure 1: The Monocular Pareidolia Illusion on Lunar Terrains (Depth vs Rise Inversion)</em></p>
</div>

```text
 58.20%  ──►  74.08%  ──►  78.43%  ──►  80.26%  ──►  81.20%  ──►  84.80%  ──►  85.10%  ──►  86.20%
 Baseline    55px Pad     Focal Loss   EffNet-B0    Full Phase 1   Phase 2 PL     Opt 1       Opt 2 (FINAL)
```

---

## 🔬 Key Engineering Innovations

### 1. The 55-Pixel Reflection Padding Theorem
* **The Spurious Artifact:** Standard image rotation (`img.rotate(-angle)`) leaves black triangular wedges in image corners occupying up to 30% of the canvas. Because Depth ($283.8^\circ$) and Rise ($181.4^\circ$) training images have distinct azimuth distributions, standard CNNs memorized corner wedge positions rather than lunar topography, causing catastrophic test generalization failure.
* **The Mathematical Solution:** For a $256 \times 256$ image, the corner radius is:
  $$r_{\text{corner}} = \sqrt{128^2 + 128^2} = 128\sqrt{2} \approx 181.02\text{ px}$$
  Expanding the image by a **55-pixel reflection pad** (`padding_mode='reflect'`) produces a $366 \times 366$ canvas:
  $$r_{\text{canvas}} = \frac{366}{2} = 183.00\text{ px} > 181.02\text{ px}$$
  Center-cropping $(256, 256)$ after rotation **completely eliminates 100% of black corner pixels across all 360° angles**, providing an immediate **+15.88% jump** (58.20% $\rightarrow$ 74.08%).

<div align="center">
  <img src="assets/azimuth_rotation.jpg" width="650" alt="Azimuth Rotation & Reflection Padding" />
  <p><em>Figure 2: Deterministic Azimuth Rotation with 55px Reflection Padding Eliminating Corner Wedge Shortcuts</em></p>
</div>

### 2. 3-Channel Differential Physics Tensor
Instead of feeding raw 1-channel grayscale into multi-channel architectures, we compute a 3-channel differential tensor on GPU:
* **Channel 0 ($I$):** Normalized grayscale intensity $I(x,y) \in [-1, 1]$.
* **Channel 1 ($\frac{\partial I}{\partial y}$):** Vertical Sobel gradient capturing shadow-fall slope along the normalized solar vector (top-to-bottom illumination).
* **Channel 2 ($\frac{\partial I}{\partial x}$):** Horizontal Sobel gradient capturing bilateral rim curvature and symmetry.

<div align="center">
  <img src="assets/three_channel_input.png" width="850" alt="3-Channel Physics Tensor Representation" />
  <p><em>Figure 3: Multi-Channel Physics Tensor Formulation (Intensity, Vertical Slope, Horizontal Slope)</em></p>
</div>

### 3. Inverse-Class-Weighted Focal Loss ($\gamma = 2.0$)
Addresses the 64% Rise to 36% Depth class imbalance by dynamically down-weighting easy background terrain and concentrating gradients on ambiguous, low-contrast crater boundaries:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

### 4. Target-Domain Semi-Supervised Pseudo-Labeling
The test set exhibits an azimuth distribution shift (mean $158.9^\circ$). We extracted 666 consensus high-confidence test patches ($p \le 0.41$ or $p \ge 0.68$) and folded them into training, driving ResNet-34 training accuracy to **91.20%**.

### 5. SLSQP Mathematical Ensemble Weight Solver
Rather than uniform averaging ($1/N$) or heuristic $BAcc^2$ weighting, we formulated ensemble blending as a constrained non-linear optimization problem solved via Sequential Least Squares Programming (SLSQP), optimizing continuous softmax weights and calibrating the decision boundary to $\tau = 0.527$.

---

## 📊 Benchmark Progression

| Stage | Pipeline / Milestone | Key Engineering Technique | Balanced Accuracy | Net Gain |
| :---: | :--- | :--- | :---: | :---: |
| **0** | **Initial Baseline** | Raw Grayscale ResNet-18 (Unpadded Black Corners) | 58.20% | Baseline |
| **1** | **Geometric Fix** | ResNet-34 + 55px Reflection Padding Expansion | 74.08% | **+15.88%** |
| **2** | **Loss & Head** | Class-Weighted Focal Loss ($\gamma=2.0$) + Deep Hierarchical Head | 78.43% | **+4.35%** |
| **3** | **K-Fold CV** | Full 5-Fold Stratified Cross-Validation ResNet-34 Blend | 79.40% | **+0.97%** |
| **4** | **Physics Tensor** | EfficientNet-B0 (Fold 1) + 3-Ch Sobel + Squeeze-and-Excitation | **80.26%** ⭐ | **+0.86%** |
| **5** | **Phase 1** | Complete 5-Fold EfficientNet-B0 CV Stratified Ensemble | 81.20% | **+0.94%** |
| **6** | **Diverse Backbones** | ConvNeXt-Tiny (7×7 Depthwise Conv) + DenseNet-121 (Dense Reuse) | 82.50% | **+1.30%** |
| **7** | **Phase 2 (Pseudo)** | 15-Model Ensemble + 666 High-Confidence Pseudo-Labels | 84.80% | **+2.30%** |
| **8** | **Optimization 1** | 15 Models + Non-Linear SLSQP Weight Solver ($\tau=0.528$) | 85.10% | **+0.30%** |
| **9** | **Optimization 2** | **16-Model Ensemble (incl. EfficientNet-V2-S Fused-MBConv) + 7-Pass TTA** | **86.20%** 🏆 | **+1.10%** |

---

## 📁 16-Model Checkpoints Portfolio (~731 MB)

All 16 checkpoints are preserved in `checkpoints/` and combined using `checkpoints/optimal_ensemble_weights.json`:

```text
 1. best_resnet34_fold2.pt            (87.0 MB) | Val: 73.54% | Weight: 6.54%
 2. best_resnet34_fold3.pt            (87.0 MB) | Val: 73.67% | Weight: 6.36%
 3. best_resnet34_fold5.pt            (87.0 MB) | Val: 73.12% | Weight: 6.35%
 4. best_efficientnet_fold3.pt        (19.5 MB) | Val: 72.79% | Weight: 6.35%
 5. best_efficientnet_fold1.pt        (19.5 MB) | Val: 69.66% (Peak: 80.26%) | Weight: 6.33%
 6. best_efficientnet_pseudo_fold2.pt (19.5 MB) | Val: 73.36% | Weight: 6.32%
 7. best_efficientnet_fold2.pt        (19.5 MB) | Val: 72.73% | Weight: 6.30%
 8. best_efficientnet_fold4.pt        (19.5 MB) | Val: 70.75% | Weight: 6.28%
 9. best_efficientnet_fold5.pt        (19.5 MB) | Val: 72.70% | Weight: 6.27%
10. best_resnet34_fold1.pt            (87.0 MB) | Val: 70.79% | Weight: 6.23%
11. best_efficientnet_v2_s_pseudo_fold1.pt (84.8 MB) | Val: 69.89% | Weight: 6.13% [NEW]
12. best_resnet34_fold4.pt            (87.0 MB) | Val: 72.12% | Weight: 6.13%
13. best_densenet121_fold1.pt         (31.1 MB) | Val: 69.40% | Weight: 6.11%
14. best_efficientnet_pseudo_fold1.pt (19.5 MB) | Val: 69.13% | Weight: 6.11%
15. best_convnext_fold1.pt           (113.5 MB) | Val: 71.20% | Weight: 6.10%
16. best_resnet34_pseudo_fold1.pt     (87.0 MB) | Val: 71.27% | Weight: 6.09%
```

---

## 🎮 Interactive Demo CLI

Test any individual lunar patch and inspect multi-model consensus predictions with a single command:

```bash
# Evaluate sample test image with interactive breakdown
python demo.py

# Evaluate custom image with specified sun azimuth angle
python demo.py --image data/test/images/eval_00001.png --azimuth 233.43
```

**Example Output:**
```text
=================================================================
  THE PAREIDOLIA PARADOX - INTERACTIVE DEMONSTRATOR
=================================================================
Device: cuda
Evaluating Image:     eval_00001.png
Solar Azimuth Angle:  233.43 deg

--- Multi-Model Prediction Breakdown ---
  best_efficientnet_fold1.pt               | P(Rise):  72.8% | -> Rise (Mound)
  best_resnet34_fold3.pt                   | P(Rise):  57.2% | -> Rise (Mound)
  best_efficientnet_v2_s_pseudo_fold1.pt   | P(Rise):  50.9% | -> Rise (Mound)
  best_convnext_fold1.pt                   | P(Rise):  57.2% | -> Rise (Mound)
  best_densenet121_fold1.pt                | P(Rise):  64.3% | -> Rise (Mound)

=================================================================
FINAL PREDICTION:  Class 1 -> RISE / MOUND (Positive Relief)
Ensemble Confidence: 60.50%
Ensemble P(Rise):    60.50% (Threshold = 52.7%)
=================================================================
```

---

## 🔍 System & Submission Integrity Check

Validate environment, GPU acceleration, all 16 checkpoints, and submission deliverables with one command:

```bash
python check_environment.py
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/YOUR_USERNAME/pareidolia-paradox.git
cd pareidolia-paradox

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Pretrained Weights
Download the 16 checkpoint `.pt` files from the Google Drive link specified in `checkpoints/README.md` and place them directly in `checkpoints/`.

### 3. Run Inference & Generate Submission
```bash
python inference.py --output submission/submission.csv
```

### 4. Verify Submission File Compliance
```bash
python submission/verify_submission.py submission/submission.csv
```

---

## 📁 Repository Directory Structure

```text
pareidolia_local/
├── README.md                          # Main publication documentation
├── LICENSE                            # MIT Open Source License
├── requirements.txt                   # Dependency specification
├── train.py                           # Root training entry point
├── inference.py                       # Root inference entry point
├── demo.py                            # Interactive single-image demonstration CLI
├── check_environment.py               # Complete environment & submission validator
├── checkpoints/
│   ├── README.md                      # Download links and weights placement guide
│   ├── optimal_ensemble_weights.json  # SLSQP solved continuous weights & threshold
│   └── *.pt                           # 16 trained model weights (~731 MB, via Drive)
├── submission/
│   ├── submission.csv                 # Official 2,000 predictions (verified)
│   ├── verify_submission.py           # Automated 100% compliance verifier
│   └── METHODOLOGY_SUMMARY.md         # Solar azimuth handling explanation
└── src/
    ├── dataset.py                     # Reflection padding & azimuth rotation engine
    ├── model.py                       # 5 neural architectures & GPU Sobel tensors
    ├── train_v2.py                    # Master K-Fold stratified training loop
    ├── optimize_ensemble_weights.py   # SLSQP mathematical weight solver
    └── predict_final.py               # 7-pass TTA inference engine
```

---

## 📄 Citation & Academic Reference

If you find this repository, the reflection-padding theorem, or the differential physics tensor methodology useful in your research or competitions, please cite:

```bibtex
@misc{gharat2026pareidolia,
  author = {Manas Pran Gharat and Krushna Arvind Katkar and Harish Kiran Desai},
  title = {The Pareidolia Paradox: Resolving Monocular Lunar Shading Inversion via Reflection-Padded Rotation and Multi-Backbone Ensemble},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/manasgharat1/pareidolia-paradox}},
  note = {IEEE SIES GST Machine Learning Competition}
}
```

---

## ⭐ Support & Acknowledgements

If you appreciate the depth of this research, the mathematical proofs, or the open-source pipeline, **please give this repository a ⭐ on GitHub!**

Organized with thanks to **IEEE SIES GST** for hosting *The Pareidolia Paradox* challenge bridging space exploration, physics, and computer vision.
