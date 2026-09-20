# Model Checkpoints Directory

This directory stores the trained PyTorch model checkpoints (`.pt` files) and optimal ensemble weights used for inference.

## 📥 Model Weights Download Link

Due to GitHub file size limits (>100 MB per file, ~784 MB total for all 16 checkpoints), the model weights are hosted externally:

- **Google Drive Download Link**: `https://drive.google.com/drive/folders/1_XutUlxi8foYycdoOToVvXfzszHt2z7P?usp=sharing`


---

### 🛠️ Instructions:
1. Download `model_weights.zip` from the Google Drive link above.
2. Extract the `.zip` archive.
3. Place all 16 `.pt` checkpoint files directly into this `checkpoints/` folder.
4. Run inference to reproduce the predictions:
   ```bash
   python inference.py
   ```

---

### 📦 Files in the Checkpoints Portfolio:

| # | Checkpoint File | Architecture | Single-Model BAcc | Optimal Ensemble Weight |
| :-: | :--- | :--- | :-: | :-: |
| 1 | `best_resnet34_fold2.pt` | ResNet-34 (Fold 2) | 78.18% | **6.54%** |
| 2 | `best_resnet34_fold3.pt` | ResNet-34 (Fold 3) | 78.43% | **6.36%** |
| 3 | `best_resnet34_fold5.pt` | ResNet-34 (Fold 5) | 76.29% | **6.35%** |
| 4 | `best_efficientnet_fold3.pt` | EfficientNet-B0 (Fold 3) | 72.79% | **6.35%** |
| 5 | `best_efficientnet_fold1.pt` | EfficientNet-B0 (Fold 1) | **80.26%** (Peak) | **6.33%** |
| 6 | `best_efficientnet_pseudo_fold2.pt` | EfficientNet-B0 (Fold 2 + PL) | 73.36% | **6.32%** |
| 7 | `best_efficientnet_fold2.pt` | EfficientNet-B0 (Fold 2) | 72.73% | **6.30%** |
| 8 | `best_efficientnet_fold4.pt` | EfficientNet-B0 (Fold 4) | 70.75% | **6.28%** |
| 9 | `best_efficientnet_fold5.pt` | EfficientNet-B0 (Fold 5) | 72.70% | **6.27%** |
| 10 | `best_resnet34_fold1.pt` | ResNet-34 (Fold 1) | 77.57% | **6.23%** |
| 11 | `best_efficientnet_v2_s_pseudo_fold1.pt` | EfficientNet-V2-S (Fold 1 + PL) | 69.89% | **6.13%** |
| 12 | `best_resnet34_fold4.pt` | ResNet-34 (Fold 4) | 74.08% | **6.13%** |
| 13 | `best_densenet121_fold1.pt` | DenseNet-121 (Fold 1) | 73.88% | **6.11%** |
| 14 | `best_efficientnet_pseudo_fold1.pt` | EfficientNet-B0 (Fold 1 + PL) | 69.13% | **6.11%** |
| 15 | `best_convnext_fold1.pt` | ConvNeXt-Tiny (Fold 1) | 71.75% | **6.10%** |
| 16 | `best_resnet34_pseudo_fold1.pt` | ResNet-34 (Fold 1 + PL) | 71.27% | **6.09%** |
| 17 | `optimal_ensemble_weights.json` | SLSQP Non-Linear Weights | **86.20% (Ensemble)** | Calibrated $\tau = 0.527$ |
