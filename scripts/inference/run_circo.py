#!/usr/bin/env python
"""
Run DeCIR inference on CIRCO dataset.

CIRCO is an open-domain composed image retrieval benchmark that uses
COCO images as the gallery. This script uses FLUX.1-dev for inpainting.

Usage:
    python scripts/inference/run_circo.py \
        --data_root data/circo \
        --gallery_embeddings data/circo/gallery_embeddings.npy \
        --output_dir outputs/circo \
        --alpha 0.5 --beta 0.3 --gamma 0.2
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from tqdm import tqdm

from decir.models.clip_encoder import CLIPEncoder, CLIPConfig
from decir.models.qwen_client import Qwen3VLClient
from decir.models.flux_client import FluxInpaintClient, FluxConfig, MockFluxClient
from decir.core.pipeline import DeCIRPipeline


def load_circo_annotations(data_root: Path, split: str = 'val') -> List[Dict]:
    """Load CIRCO annotations."""
    anno_file = data_root / 'annotations' / f'{split}.json'

    if not anno_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations from {anno_file}")
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Run DeCIR inference on CIRCO dataset")

    # Dataset arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of CIRCO dataset')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Dataset split')

    # Gallery embeddings
    parser.add_argument('--gallery_embeddings', type=str, required=True,
                        help='Path to pre-computed gallery embeddings (.npy)')

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

    # Pipeline parameters (CIRCO-optimized defaults)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for reference image (CIRCO: 0.5)')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Weight for text (CIRCO: 0.3)')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Weight for visual delta (CIRCO: 0.2)')
    parser.add_argument('--delta_mode', type=str, default='diff',
                        choices=['diff', 'patch', 'none'],
                        help='Delta computation mode')

    # Processing
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Number of top results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    # Options
    parser.add_argument('--mock_flux', action='store_true',
                        help='Use mock FLUX for testing')

    args = parser.parse_args()

    # Setup
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DeCIR Inference on CIRCO")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Split: {args.split}")
    print(f"Gallery embeddings: {args.gallery_embeddings}")
    print(f"Output dir: {output_dir}")
    print(f"Fusion weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print("="*60)

    # Load annotations
    print("\n[1/4] Loading CIRCO annotations...")
    annotations = load_circo_annotations(data_root, args.split)

    # Load gallery embeddings
    print("\n[2/4] Loading gallery embeddings...")
    gallery_embeddings = np.load(args.gallery_embeddings)
    print(f"Loaded {len(gallery_embeddings)} gallery embeddings")

    # Load models
    print("\n[3/4] Loading models...")
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

    # Run inference
    print("\n[4/4] Running inference...")

    all_predictions = {}

    for item in tqdm(annotations, desc="Processing"):
        query_id = str(item.get('id', item.get('pairid')))
        ref_image_id = item.get('reference_img_id')
        caption = item.get('caption')

        # Get reference image path
        ref_img_path = data_root / 'COCO2017_unlabeled' / 'unlabeled2017' / f"{ref_image_id:012d}.jpg"

        if not ref_img_path.exists():
            print(f"Warning: Image not found: {ref_img_path}")
            continue

        try:
            # Run pipeline
            query_embedding = pipeline(
                reference_image=str(ref_img_path),
                modification_text=caption
            )

            # Compute similarities
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(gallery_norm, query_norm)

            # Rank
            ranked_indices = np.argsort(similarities)[::-1][:args.top_k].tolist()

            all_predictions[query_id] = ranked_indices

        except Exception as e:
            print(f"\nError processing {query_id}: {e}")
            all_predictions[query_id] = list(range(args.top_k))

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
    print("="*60)


if __name__ == '__main__':
    main()
