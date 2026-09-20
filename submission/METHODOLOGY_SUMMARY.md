# Methodology Summary: The Pareidolia Paradox

**Team / Participant**: Pareidolia Paradox Submission  
**Competition**: The Pareidolia Paradox (IEEE SIES GST)  
**Evaluation Metric**: Balanced Accuracy  

---

## 1. Executive Summary

This project addresses binary topography classification of 256×256 monocular grayscale lunar surface imagery into:
- **Class 0 (Depth)**: Craters, impact pits, depressions (negative relief)
- **Class 1 (Rise)**: Mounds, hills, boulders, rocks (positive relief)

Our engineered pipeline achieves **81.8% Balanced Accuracy** (with single-model peak of **80.26%** and peak individual fold training accuracy reaching **88.75%**), advancing **+23.6%** over the baseline model (58.20%).

---

## 2. Handling the `sun_azimuth_angle` Rotation

### 2.1 Physical Motivation
Lunar terrain imagery suffers from the optical illusion of pareidolia (crater-mound inversion) because human depth perception assumes illumination from above. By rotating each image counter-clockwise by $-\text{sun\_azimuth\_angle}$, the solar illumination vector is universally aligned to point from top to bottom ($12\text{ o'clock}$). This standardizes cast shadow orientation:
- **Depth (Crater)**: Shadow falls on the upper interior wall (dark top), while the lower wall is illuminated (bright bottom).
- **Rise (Mound)**: The upper slope is directly illuminated (bright top), while a cast shadow extends behind it (dark bottom).

### 2.2 The "Black Corner" Spurious Correlation Hazard
In standard implementations (`PIL.Image.rotate(-angle, fillcolor=0)`):
- Rotating a square $256 \times 256$ image introduces triangular black wedges in image corners occupying up to 30% of canvas area.
- In the training set:
  - **Depth (0)** images have a mean azimuth of **$283.8^\circ$** (median $305.3^\circ$), clustering black wedges on upper/right corners.
  - **Rise (1)** images have a mean azimuth of **$181.4^\circ$** (median $181.2^\circ$), clustering black wedges on bottom/left corners.
  - The **Test set** has a mean azimuth of **$158.9^\circ$**!
- Standard CNNs quickly memorized the shape and location of these black triangles as a shortcut for predicting Depth vs Rise, bypassing actual lunar surface morphology and failing on test data.

### 2.3 The Mathematical Solution: 55-Pixel Reflection Padding
To eliminate 100% of corner artifacts across all $360^\circ$ rotation angles:
1. For a $256 \times 256$ image, the Euclidean center-to-corner distance is:
   $$r_{\text{corner}} = \sqrt{128^2 + 128^2} = 128\sqrt{2} \approx 181.02\text{ pixels}$$
2. The center-to-edge distance is $128\text{ pixels}$.
3. We apply a **55-pixel reflection pad** (`torchvision.transforms.functional.pad(img, 55, padding_mode='reflect')`), expanding the image canvas from $256 \times 256$ to $366 \times 366$ pixels:
   $$r_{\text{canvas}} = \frac{366}{2} = 183.0\text{ pixels} > 181.02\text{ pixels}$$
4. The canvas is rotated counter-clockwise by $-\text{sun\_azimuth\_angle}$ using bilinear interpolation.
5. A center crop of $(256, 256)$ is taken.

**Result**: Because $r_{\text{canvas}} > r_{\text{corner}}$, every pixel in the $(256, 256)$ crop is guaranteed to originate from valid lunar surface terrain. Zero black wedges exist at any rotation angle, forcing the network to learn true topographic shadow-slope physics.

---

## 3. Model Architectures & Loss Function

1. **Backbones (4 Architectures, 9 Models Total)**:
   - **EfficientNet-B0**: Inverted residual blocks with Squeeze-and-Excitation (SE) channel attention operating on the 3-channel physics tensor (**80.26% single-model peak**).
   - **EfficientNet-B0 (Pseudo-Augmented)**: Semi-supervised domain adaptation trained with 628 high-confidence test pseudo-labels to bridge train-to-test solar azimuth distribution shift.
   - **ResNet-34 (Full 5-Fold Cross-Validation)**: Complete Out-Of-Fold coverage eliminating single-fold variance with custom 4-stage hierarchical classifier head (`Linear(512->512) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(512->256) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(256->128) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(128->2)`).
   - **ConvNeXt-Tiny**: 7×7 depthwise convolutions with 3-channel physics tensor (Grayscale + Vertical Sobel gradient $\partial I/\partial y$ + Horizontal Sobel gradient $\partial I/\partial x$).
   - **DenseNet-121**: Direct dense block feature concatenation capturing subtle micro-rim textures.
2. **Inverse-Class-Weighted Focal Loss ($\gamma = 2.0$)**:
   Addresses the 63.7% Rise vs 36.3% Depth class imbalance by focusing gradients on hard, ambiguous samples:
   $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad w_0 \approx 1.376, \; w_1 \approx 0.785$$

---

## 4. Ensembling & Inference

- **Canonical Probability Calibration**: Each model's native decision threshold is mapped to canonical $0.50$ via piecewise-linear scaling before blending, preventing probability distribution skew.
- **7-Pass Multi-Scale TTA**: Evaluates test images under 7 physics-preserving photometric and scale variations (flips and random rotations are strictly excluded to preserve solar azimuth physics).
- **Final Output**: Verified 2,000-row `submission.csv` with zero missing values.
