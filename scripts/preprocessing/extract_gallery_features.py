#!/usr/bin/env python
"""
Extract CLIP features for gallery (candidate) images.

This script pre-computes CLIP embeddings for all gallery images to enable
efficient retrieval. The embeddings are saved as a numpy array.

Usage:
    # For CIRR
    python scripts/preprocessing/extract_gallery_features.py \
        --dataset cirr \
        --data_root data/cirr \
        --split test \
        --output data/cirr/gallery_embeddings_test.npy \
        --batch_size 64

    # For CIRCO
    python scripts/preprocessing/extract_gallery_features.py \
        --dataset circo \
        --data_root data/circo \
        --output data/circo/gallery_embeddings.npy \
        --batch_size 64
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from decir.models.clip_encoder import CLIPEncoder, CLIPConfig


class ImageDataset(Dataset):
    """Simple dataset for loading images."""

    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return str(self.image_paths[idx])


def load_gallery_image_paths(dataset: str, data_root: Path, split: str = None) -> List[Path]:
    """
    Load gallery image paths for different datasets.

    Args:
        dataset: Dataset name ('cirr', 'circo', 'fashioniq', 'genecis')
        data_root: Root directory of the dataset
        split: Split name (for datasets that have splits)

    Returns:
        List of image paths
    """
    image_paths = []

    if dataset == 'cirr':
        # CIRR gallery images
        if split == 'test':
            img_dir = data_root / 'test1'
        elif split == 'val':
            img_dir = data_root / 'dev'
        else:
            raise ValueError(f"Unknown split for CIRR: {split}")

        if img_dir.exists():
            image_paths = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
        else:
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

    elif dataset == 'circo':
        # CIRCO uses COCO images
        img_dir = data_root / 'COCO2017_unlabeled' / 'unlabeled2017'
        if img_dir.exists():
            image_paths = sorted(img_dir.glob('*.jpg'))
        else:
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

    elif dataset == 'fashioniq':
        # FashionIQ has separate image directories
        img_dir = data_root / 'images'
        if img_dir.exists():
            image_paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        else:
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

    elif dataset == 'genecis':
        # GeneCIS gallery
        img_dir = data_root / 'images'
        if img_dir.exists():
            image_paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        else:
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Found {len(image_paths)} gallery images in {img_dir}")
    return image_paths


def extract_features(
    image_paths: List[Path],
    clip_encoder: CLIPEncoder,
    batch_size: int = 64,
    num_workers: int = 4
) -> np.ndarray:
    """
    Extract CLIP features for a list of images.

    Args:
        image_paths: List of paths to images
        clip_encoder: CLIP encoder instance
        batch_size: Batch size for encoding
        num_workers: Number of data loading workers

    Returns:
        Array of shape (N, embedding_dim) containing CLIP features
    """
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    all_features = []

    print(f"Extracting features for {len(image_paths)} images...")

    with torch.no_grad():
        for batch_paths in tqdm(dataloader, desc="Encoding"):
            # Load images
            images = []
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    # Use a black image as placeholder
                    images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

            # Encode batch
            features = clip_encoder.encode_images(images)
            all_features.append(features.cpu().numpy())

    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)

    return all_features


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP features for gallery images"
    )

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cirr', 'circo', 'fashioniq', 'genecis'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--split', type=str, default=None,
                        help='Split name (for datasets with splits, e.g., test/val for CIRR)')

    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the features (.npy file)')

    # CLIP model
    parser.add_argument('--clip_model', type=str,
                        default='laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
                        help='CLIP model name')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    # Processing
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Convert paths to Path objects
    data_root = Path(args.data_root)
    output_path = Path(args.output)

    print("="*60)
    print("Gallery Feature Extraction")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {data_root}")
    print(f"Split: {args.split}")
    print(f"Output: {output_path}")
    print(f"CLIP model: {args.clip_model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load CLIP encoder
    print("\n[1/3] Loading CLIP encoder...")
    clip_config = CLIPConfig(
        model_name=args.clip_model,
        device=args.device
    )
    clip_encoder = CLIPEncoder(clip_config)
    print(f"CLIP encoder loaded successfully")

    # Load gallery image paths
    print("\n[2/3] Loading gallery image paths...")
    image_paths = load_gallery_image_paths(
        dataset=args.dataset,
        data_root=data_root,
        split=args.split
    )

    # Extract features
    print("\n[3/3] Extracting features...")
    features = extract_features(
        image_paths=image_paths,
        clip_encoder=clip_encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Save features
    print(f"\nSaving features to {output_path}...")
    np.save(output_path, features)

    print("\n" + "="*60)
    print("✓ Done!")
    print(f"Extracted features for {len(features)} images")
    print(f"Feature shape: {features.shape}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"Saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
