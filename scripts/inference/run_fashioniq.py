#!/usr/bin/env python
"""
Run DeCIR inference on FashionIQ dataset.

FashionIQ has three categories (dress, shirt, toptee) and each query
has TWO modification captions. This script uses FLUX.1-dev for inpainting.

Usage:
    # Run for a single category
    python scripts/inference/run_fashioniq.py \
        --data_root data/fashioniq \
        --category dress \
        --split test \
        --output_dir outputs/fashioniq_dress \
        --alpha 0.7 --beta 0.15 --gamma 0.15

    # Run for all categories
    for category in dress shirt toptee; do
        python scripts/inference/run_fashioniq.py \
            --data_root data/fashioniq \
            --category $category \
            --output_dir outputs/fashioniq_$category
    done
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from decir.models.clip_encoder import CLIPEncoder, CLIPConfig
from decir.models.qwen_client import Qwen3VLClient
from decir.models.flux_client import FluxInpaintClient, FluxConfig, MockFluxClient
from decir.core.pipeline import DeCIRPipeline


def load_fashioniq_annotations(data_root: Path, category: str, split: str) -> List[Dict]:
    """Load FashionIQ annotations for a specific category."""
    anno_file = data_root / 'captions' / f'cap.{category}.{split}.json'

    if not anno_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations for {category}/{split}")
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Run DeCIR inference on FashionIQ dataset")

    # Dataset arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of FashionIQ dataset')
    parser.add_argument('--category', type=str, required=True,
                        choices=['dress', 'shirt', 'toptee'],
                        help='Fashion category')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    # Model paths
    parser.add_argument('--qwen_model', type=str,
                        default='Qwen/Qwen2-VL-7B-Instruct',
                        help='Qwen model path')
    parser.add_argument('--clip_model', type=str,
                        default='laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
                        help='CLIP model name')
    parser.add_argument('--flux_model', type=str,
                        default='black-forest-labs/FLUX.1-dev',
                        help='FLUX model name (FLUX.1-dev recommended)')

    # Pipeline parameters (FashionIQ-optimized defaults)
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for reference image (FashionIQ: 0.7)')
    parser.add_argument('--beta', type=float, default=0.15,
                        help='Weight for text (FashionIQ: 0.15)')
    parser.add_argument('--gamma', type=float, default=0.15,
                        help='Weight for visual delta (FashionIQ: 0.15)')
    parser.add_argument('--delta_mode', type=str, default='diff',
                        choices=['diff', 'patch', 'none'],
                        help='Delta computation mode')

    # Processing
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Number of top results')

    # Options
    parser.add_argument('--mock_flux', action='store_true',
                        help='Use mock FLUX for testing')
    parser.add_argument('--combine_captions', type=str, default='concat',
                        choices=['concat', 'separate', 'average'],
                        help='How to handle dual captions')

    args = parser.parse_args()

    # Setup
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DeCIR Inference on FashionIQ")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Category: {args.category}")
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")
    print(f"Fusion weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"Caption handling: {args.combine_captions}")
    print("="*60)

    # Load annotations
    print("\n[1/5] Loading FashionIQ annotations...")
    annotations = load_fashioniq_annotations(data_root, args.category, args.split)

    # Load gallery images
    print("\n[2/5] Loading gallery images...")
    image_dir = data_root / 'images'
    gallery_images = sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.png'))
    print(f"Found {len(gallery_images)} gallery images")

    # Load models
    print("\n[3/5] Loading models...")
    qwen_client = Qwen3VLClient(model_path=args.qwen_model, device=args.device)
    clip_encoder = CLIPEncoder(CLIPConfig(model_name=args.clip_model, device=args.device))

    if args.mock_flux:
        flux_client = MockFluxClient()
    else:
        flux_client = FluxInpaintClient(FluxConfig(model_name=args.flux_model, device=args.device))

    # Create pipeline
    pipeline = DeCIRPipeline(
        qwen_client=qwen_client,
        clip_encoder=clip_encoder,
        inpaint_client=flux_client,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta_mode=args.delta_mode
    )

    # Encode gallery
    print("\n[4/5] Encoding gallery images...")
    from PIL import Image
    gallery_images_pil = [Image.open(p).convert('RGB') for p in tqdm(gallery_images, desc="Loading")]
    gallery_embeddings = clip_encoder.encode_images(gallery_images_pil).cpu().numpy()
    print(f"Gallery embeddings shape: {gallery_embeddings.shape}")

    # Run inference
    print("\n[5/5] Running inference...")

    all_predictions = {}

    for idx, item in enumerate(tqdm(annotations, desc="Processing")):
        candidate = item.get('candidate')
        target = item.get('target')
        captions = item.get('captions', [])

        if len(captions) != 2:
            print(f"Warning: Expected 2 captions, got {len(captions)} for query {idx}")
            continue

        # Get reference image path
        ref_img_path = image_dir / f"{candidate}.jpg"
        if not ref_img_path.exists():
            ref_img_path = image_dir / f"{candidate}.png"

        if not ref_img_path.exists():
            print(f"Warning: Reference image not found: {candidate}")
            continue

        try:
            # Handle dual captions
            if args.combine_captions == 'concat':
                # Concatenate captions with separator
                combined_caption = f"{captions[0]}. {captions[1]}"
                query_embedding = pipeline(
                    reference_image=str(ref_img_path),
                    modification_text=combined_caption
                )

            elif args.combine_captions == 'separate':
                # Process separately and average
                emb1 = pipeline(str(ref_img_path), captions[0])
                emb2 = pipeline(str(ref_img_path), captions[1])
                query_embedding = (emb1 + emb2) / 2

            else:  # average (default)
                combined_caption = f"{captions[0]} {captions[1]}"
                query_embedding = pipeline(
                    reference_image=str(ref_img_path),
                    modification_text=combined_caption
                )

            # Compute similarities
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(gallery_norm, query_norm)

            # Rank
            ranked_indices = np.argsort(similarities)[::-1][:args.top_k]

            # Convert indices to image IDs
            ranked_image_ids = [gallery_images[i].stem for i in ranked_indices]

            all_predictions[str(idx)] = ranked_image_ids

        except Exception as e:
            print(f"\nError processing query {idx}: {e}")
            all_predictions[str(idx)] = [gallery_images[i].stem for i in range(min(args.top_k, len(gallery_images)))]

    # Save results
    print("\n" + "="*60)
    print("Saving results...")

    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"  - Predictions: {predictions_path}")

    print("\n✓ Done!")
    print(f"Processed {len(all_predictions)} queries")
    print(f"Results saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  python scripts/evaluation/evaluate_fashioniq.py \\")
    print(f"      --submission-file {predictions_path} \\")
    print(f"      --dataset-path {data_root}/captions/cap.{args.category}.{args.split}.json")
    print("="*60)


if __name__ == '__main__':
    main()
