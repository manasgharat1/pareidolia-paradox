"""
dataset.py
==========
Loads Pareidolia Paradox competition data.

Official spec:
  - Class 0 = Depth  (craters, holes, depressions)
  - Class 1 = Rise   (mounds, hills, rocks, boulders)
  - CSV columns: image_id, sun_azimuth_angle, label (label absent in test CSV)
  - Images are 256x256 grayscale .png files

HOW AZIMUTH IS HANDLED (per the official problem statement):
The problem statement gives an EXACT, deterministic preprocessing rule:

    "Rotate each image counter-clockwise by -sun_azimuth_angle before
    training/inference. This normalizes lighting across all images and
    prevents shadow-flipping errors."

This means we do NOT feed azimuth to the model as a learned feature.
Instead, we physically rotate every image so that, after rotation, the
sun direction is the SAME reference direction for every single image in
the dataset. Once that's done, a crater lit from "normalized north" always
casts its shadow the same way as every other crater in the dataset -- so
the model can learn shape (crater vs mound) directly from pixels, without
needing to disentangle lighting direction at all. This is why the model
below takes ONLY the image as input -- no azimuth branch needed anymore.

IMPORTANT: because the rotation IS the azimuth-correction step, we must
NOT use random rotation as a data augmentation technique -- that would
undo the careful normalization we just did. Augmentation below is
restricted to things that don't touch rotation (brightness/contrast/
zoom/translation/blur).
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

CLASS_NAMES = ["Depth", "Rise"]  # index 0, index 1 -- matches official spec exactly


def normalize_azimuth(img: Image.Image, sun_azimuth_angle: float) -> Image.Image:
    """
    Rotates a PIL image counter-clockwise by -sun_azimuth_angle degrees,
    exactly as specified in the problem statement, to normalize lighting
    direction across the whole dataset.

    Uses reflection padding (55 pixels) prior to rotation, followed by
    center-cropping back to the original (256, 256) size. This mathematically
    eliminates 100% of black corner artifacts across all 360 degrees:
    center-to-corner distance is 128*sqrt(2) ~= 181.02 <= 128+55 = 183.
    """
    angle = -float(sun_azimuth_angle)
    padded = TF.pad(img, 55, padding_mode='reflect')
    rotated = padded.rotate(angle, resample=Image.BILINEAR)
    return TF.center_crop(rotated, (256, 256))


class LunarDataset(Dataset):
    def __init__(self, csv_path, img_dir, image_size=256, train=True, has_labels=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.train = train
        self.has_labels = has_labels

        if "sun_azimuth_angle" not in self.df.columns:
            raise ValueError(
                "Expected a 'sun_azimuth_angle' column per the problem statement, "
                f"but got columns: {list(self.df.columns)}"
            )

        # Augmentation intentionally excludes rotation and flips -- the
        # azimuth-normalization rotation IS our rotation step, and further
        # rotating would re-introduce the shadow-direction ambiguity we
        # just removed. Safe augmentations only affect brightness/contrast/
        # scale/small translation/blur, none of which touch orientation.
        if train:
            self.post_transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25)], p=0.7),
                T.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.94, 1.06)),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
                T.RandomErasing(p=0.35, scale=(0.02, 0.18), ratio=(0.5, 2.0), value=0),
            ])
        else:
            self.post_transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = Image.open(img_path).convert("L")

        # Step 1: azimuth-normalizing rotation (per problem statement) -- always applied, train or eval
        img = normalize_azimuth(img, row["sun_azimuth_angle"])

        # Step 2: standard resize/augment/tensor conversion
        img_tensor = self.post_transform(img)

        if self.has_labels:
            label = torch.tensor(int(row["label"]), dtype=torch.long)
            return img_tensor, label
        else:
            return img_tensor, row["image_id"]
