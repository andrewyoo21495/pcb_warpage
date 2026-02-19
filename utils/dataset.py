#!/usr/bin/env python3
"""PCB Warpage dataset for CVAE training.

Directory layout expected:
  data/
    design/
      design_A.png
      design_B.png
      design_C.png
      design_D.png
    elevation/
      design_A/   <- many elevation images for design A
      design_B/
      design_C/
      design_D/

Each dataset sample is a (design_image, elevation_image, hand_features) tuple.

Leave-one-out split:
  val_fold ∈ {0, 1, 2, 3}  → design index held out as test/validation set.
  split='train'  → all designs except val_fold
  split='val'    → only val_fold design
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from utils.handcrafted_features import extract_handcrafted_features

DESIGN_NAMES = ['design_A', 'design_B', 'design_C', 'design_D']


class PCBWarpageDataset(Dataset):
    """Dataset of (design, elevation) image pairs for CVAE training.

    Args:
        dataset_dir      : Root data directory containing design/ and elevation/
        config           : Config dict from load_config()
        split            : 'train' or 'val'
        val_fold         : Index into DESIGN_NAMES for the held-out design (0-3)
    """

    def __init__(self, dataset_dir: str, config: dict, split: str = 'train', val_fold: int = 0):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.image_size  = int(config.get('image_size', 256))
        self.split       = split
        self.val_fold    = val_fold

        use_design_aug    = config.get('use_design_aug', True)
        use_elevation_aug = config.get('use_elevation_aug', True)
        self.augment          = (split == 'train')
        self.use_design_aug   = use_design_aug and self.augment
        self.use_elevation_aug = use_elevation_aug and self.augment

        self.samples = self._build_sample_list()
        print(f"PCBWarpageDataset [{split}]: {len(self.samples)} samples "
              f"(val_fold={val_fold}, held-out={DESIGN_NAMES[val_fold]})")

    # ------------------------------------------------------------------
    def _build_sample_list(self):
        """Return list of (design_path, elevation_path) tuples."""
        samples = []
        elevation_root = self.dataset_dir / 'elevation'
        design_root    = self.dataset_dir / 'design'

        for idx, name in enumerate(DESIGN_NAMES):
            is_val_design = (idx == self.val_fold)
            if self.split == 'train' and is_val_design:
                continue
            if self.split == 'val' and not is_val_design:
                continue

            design_path   = design_root / f"{name}.png"
            elev_dir      = elevation_root / name

            if not design_path.exists():
                raise FileNotFoundError(f"Design image not found: {design_path}")
            if not elev_dir.exists():
                raise FileNotFoundError(f"Elevation directory not found: {elev_dir}")

            elev_paths = sorted(elev_dir.glob('*.png'))
            if len(elev_paths) == 0:
                raise ValueError(f"No elevation images found in {elev_dir}")

            for elev_path in elev_paths:
                samples.append((str(design_path), str(elev_path)))

        return samples

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        design_path, elev_path = self.samples[idx]

        # Load grayscale PIL images
        design    = Image.open(design_path).convert('L')
        elevation = Image.open(elev_path).convert('L')

        # Resize to target size
        size = (self.image_size, self.image_size)
        design    = design.resize(size, Image.LANCZOS)
        elevation = elevation.resize(size, Image.LANCZOS)

        # --- Shared spatial augmentation (same transform applied to both) ---
        if self.augment:
            if random.random() > 0.5:
                design    = TF.hflip(design)
                elevation = TF.hflip(elevation)
            if random.random() > 0.5:
                design    = TF.vflip(design)
                elevation = TF.vflip(elevation)

        # --- Design-specific augmentation (brightness / contrast jitter) ---
        if self.use_design_aug:
            brightness_factor = random.uniform(0.7, 1.3)
            contrast_factor   = random.uniform(0.7, 1.3)
            design = TF.adjust_brightness(design, brightness_factor)
            design = TF.adjust_contrast(design, contrast_factor)

        # Convert to float tensors in [0, 1]  →  shape (1, H, W)
        design_tensor    = TF.to_tensor(design)    # white=1.0, black=0.0
        elevation_tensor = TF.to_tensor(elevation) # smooth values in [0, 1]

        # Compute handcrafted features from (possibly augmented) design tensor
        hand_features = extract_handcrafted_features(design_tensor)

        return design_tensor, elevation_tensor, hand_features


def create_dataloaders(config: dict):
    """Create train and validation DataLoaders from config.

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    dataset_dir  = config.get('dataset_dir', './data')
    val_fold     = int(config.get('val_fold', 0))
    batch_size   = int(config.get('batch_size', 32))
    num_workers  = int(config.get('num_workers', 4))

    train_dataset = PCBWarpageDataset(dataset_dir, config, split='train', val_fold=val_fold)
    val_dataset   = PCBWarpageDataset(dataset_dir, config, split='val',   val_fold=val_fold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
