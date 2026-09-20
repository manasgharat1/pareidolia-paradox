"""
model.py  — ALL model architectures for the Pareidolia Paradox pipeline
========================================================================

Available architectures:
  "simple"      – SimpleCNN (no download, fast smoke-test)
  "resnet18"    – ResNet-18, ImageNet pretrained, 1-ch grayscale
  "resnet34"    – ResNet-34, ImageNet pretrained, deep classifier head (TRAINED √)
  "convnext"    – ConvNeXt-Tiny, ImageNet pretrained, 3-ch physics input
  "efficientnet"– EfficientNet-B0, ImageNet pretrained, 3-ch physics input

3-Channel Physics Input (convnext / efficientnet)
--------------------------------------------------
Instead of naive grayscale, these models receive a 3-channel tensor:
  Ch0: normalized grayscale   I(x,y)
  Ch1: vertical Sobel         dI/dy   (shadow slope – key for depth vs rise)
  Ch2: horizontal Sobel       dI/dx   (rim curvature)
The build_physics_tensor() function converts 1-ch → 3-ch tensors on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ──────────────────────────────────────────────────────
# Sobel edge-detection filters (used by ConvNeXt / EfficientNet pipeline)
# ──────────────────────────────────────────────────────
_SOBEL_Y = torch.tensor([[-1., -2., -1.],
                          [ 0.,  0.,  0.],
                          [ 1.,  2.,  1.]], requires_grad=False).view(1, 1, 3, 3) / 4.0

_SOBEL_X = torch.tensor([[-1.,  0.,  1.],
                          [-2.,  0.,  2.],
                          [-1.,  0.,  1.]], requires_grad=False).view(1, 1, 3, 3) / 4.0


def build_physics_tensor(gray_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert 1-channel grayscale tensor [B, 1, H, W] →
    3-channel physics tensor [B, 3, H, W]:
       Ch0 = original grayscale (already normalised to ~[-1, 1])
       Ch1 = vertical Sobel gradient   (shadow slope)
       Ch2 = horizontal Sobel gradient (rim curvature)
    """
    dev = gray_tensor.device
    ky = _SOBEL_Y.to(dev)
    kx = _SOBEL_X.to(dev)
    gy = F.conv2d(gray_tensor, ky, padding=1)
    gx = F.conv2d(gray_tensor, kx, padding=1)
    return torch.cat([gray_tensor, gy, gx], dim=1)


# ──────────────────────────────────────────────────────
# SimpleCNN  (fallback, no pretrained weights needed)
# ──────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, img):
        return self.classifier(self.features(img).flatten(1))


# ──────────────────────────────────────────────────────
# ResNetClassifier  (ResNet-18 / ResNet-34)
# ──────────────────────────────────────────────────────
class ResNetClassifier(nn.Module):
    """ResNet-18 or ResNet-34 backbone, 1-ch grayscale, ImageNet pretrained."""

    def __init__(self, num_classes=2, depth=18, pretrained=True):
        super().__init__()
        if depth == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif depth == 34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported depth: {depth}. Use 18 or 34.")

        # Adapt first conv for 1-channel input
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        feature_dim = backbone.fc.in_features  # 512 for both
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if depth == 18:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )
        else:  # 34 — deeper hierarchical head per analysis plan
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )

    def forward(self, img):
        return self.classifier(self.backbone(img))


# ──────────────────────────────────────────────────────
# ConvNeXtClassifier  (3-channel physics input)
# ──────────────────────────────────────────────────────
class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Tiny backbone accepting a **3-channel physics tensor**
    (grayscale + Sobel-Y + Sobel-X) built by build_physics_tensor().
    First conv is re-initialised from the pretrained RGB weights.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # Re-use features + avgpool; replace the full classifier
        feature_dim = 768  # ConvNeXt-Tiny feature dim after avgpool
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.norm     = backbone.classifier[0]  # LayerNorm2d

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, img):
        # img: [B, 1, H, W] → build 3-ch physics tensor internally
        x = build_physics_tensor(img)          # [B, 3, H, W]
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = x.flatten(1)                       # [B, 768]
        return self.classifier(x)


# ──────────────────────────────────────────────────────
# EfficientNetClassifier  (3-channel physics input)
# ──────────────────────────────────────────────────────
class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 backbone accepting a 3-channel physics tensor.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        feature_dim = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, img):
        x = build_physics_tensor(img)   # [B, 3, H, W]
        x = self.backbone(x)            # [B, 1280]
        return self.classifier(x)


# ──────────────────────────────────────────────────────
# DenseNetClassifier (DenseNet-121, 1-ch grayscale)
# ──────────────────────────────────────────────────────
class DenseNetClassifier(nn.Module):
    """DenseNet-121 backbone, 1-ch grayscale, ImageNet pretrained."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Adapt first conv for 1-channel input
        old_conv = backbone.features.conv0
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.features.conv0 = new_conv

        feature_dim = backbone.classifier.in_features  # 1024
        self.features = backbone.features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, img):
        features = self.features(img)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        return self.classifier(out)


# ──────────────────────────────────────────────────────
# EfficientNetV2Classifier (EfficientNet-V2-S, 3-ch physics)
# ──────────────────────────────────────────────────────
class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNet-V2-S backbone accepting a 3-channel physics tensor
    (grayscale + Sobel-Y + Sobel-X) with Fused-MBConv layers.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)

        feature_dim = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, img):
        x = build_physics_tensor(img)   # [B, 3, H, W]
        x = self.backbone(x)            # [B, 1280]
        return self.classifier(x)


# ──────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────
def get_model(name="resnet34", num_classes=2):
    """
    name options:
        "simple"            → SimpleCNN (no download)
        "resnet18"          → ResNet-18 pretrained (1-ch)
        "resnet34"          → ResNet-34 pretrained (1-ch, deep head)   [main model]
        "densenet121"       → DenseNet-121 pretrained (1-ch, deep head) [main model]
        "convnext"          → ConvNeXt-Tiny pretrained (3-ch physics)
        "efficientnet"      → EfficientNet-B0 pretrained (3-ch physics)
        "efficientnet_v2_s" → EfficientNet-V2-S pretrained (3-ch physics)
        "resnet"            → alias for resnet18 (backward compat)
        "densenet"          → alias for densenet121
        "effnet_v2"         → alias for efficientnet_v2_s
    """
    if name == "simple":
        return SimpleCNN(num_classes=num_classes)
    elif name in ("resnet", "resnet18"):
        return ResNetClassifier(num_classes=num_classes, depth=18)
    elif name == "resnet34":
        return ResNetClassifier(num_classes=num_classes, depth=34)
    elif name in ("densenet", "densenet121"):
        return DenseNetClassifier(num_classes=num_classes)
    elif name == "convnext":
        return ConvNeXtClassifier(num_classes=num_classes)
    elif name == "efficientnet":
        return EfficientNetClassifier(num_classes=num_classes)
    elif name in ("efficientnet_v2_s", "effnet_v2"):
        return EfficientNetV2Classifier(num_classes=num_classes)
    raise ValueError(f"Unknown model name: '{name}'. Use simple/resnet18/resnet34/densenet121/convnext/efficientnet/efficientnet_v2_s.")