#!/usr/bin/env python
"""
Run DeCIR inference on CIRR dataset.

This script performs composed image retrieval on the CIRR dataset using the
DeCIR pipeline with FLUX.1-dev for inpainting.

Usage:
    python scripts/inference/run_cirr.py \
        --data_root data/cirr \
        --split test \
        --gallery_embeddings data/cirr/gallery_embeddings_test.npy \
        --output_dir outputs/cirr_test \
        --alpha 0.6 --beta 0.2 --gamma 0.2 \
        --batch_size 8
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from decir.datasets.cirr import CIRRDataset
from decir.models.clip_encoder import CLIPEncoder, CLIPConfig
from decir.models.qwen_client import Qwen3VLClient
from decir.models.flux_client import FluxInpaintClient, FluxConfig, MockFluxClient
from decir.core.pipeline import DeCIRPipeline


def load_gallery_embeddings(gallery_path: str) -> np.ndarray:
    """Load pre-computed gallery embeddings."""
    print(f"Loading gallery embeddings from {gallery_path}...")
    embeddings = np.load(gallery_path)
    print(f"Loaded {len(embeddings)} gallery embeddings, shape: {embeddings.shape}")
    return embeddings


def compute_similarities(query_emb: np.ndarray, gallery_embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between query and gallery.

    Args:
        query_emb: Query embedding of shape (D,)
        gallery_embs: Gallery embeddings of shape (N, D)

    Returns:
        Similarities of shape (N,)
    """
    # Normalize
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    gallery_norm = gallery_embs / (np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-8)

    # Compute similarities
    similarities = np.dot(gallery_norm, query_norm)

    return similarities


def rank_gallery(similarities: np.ndarray, top_k: int = 50) -> List[int]:
    """
    Rank gallery by similarity scores.

    Args:
        similarities: Similarity scores of shape (N,)
        top_k: Number of top results to return

    Returns:
        List of top-K gallery indices
    """
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return ranked_indices.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run DeCIR inference on CIRR dataset")

    # Dataset arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of CIRR dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to use')

    # Gallery embeddings
    parser.add_argument('--gallery_embeddings', type=str, required=True,
                        help='Path to pre-computed gallery embeddings (.npy)')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')

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

    # Pipeline parameters
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Weight for reference image features')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Weight for text features')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Weight for visual delta features')
    parser.add_argument('--delta_mode', type=str, default='diff',
                        choices=['diff', 'patch', 'none'],
                        help='Delta computation mode')

    # Processing
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Number of top results to save')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    # Options
    parser.add_argument('--mock_flux', action='store_true',
                        help='Use mock FLUX for faster testing (no actual editing)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip samples that already have results')

    args = parser.parse_args()

    # Setup
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DeCIR Inference on CIRR")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Split: {args.split}")
    print(f"Gallery embeddings: {args.gallery_embeddings}")
    print(f"Output dir: {output_dir}")
    print(f"Fusion weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"Delta mode: {args.delta_mode}")
    print(f"Device: {args.device}")
    print("="*60)

    # Load dataset
    print("\n[1/5] Loading CIRR dataset...")
    dataset = CIRRDataset(data_root=str(data_root), split=args.split)
    print(f"Loaded {len(dataset)} queries")

    # Load gallery embeddings
    print("\n[2/5] Loading gallery embeddings...")
    gallery_embeddings = load_gallery_embeddings(args.gallery_embeddings)

    # Load models
    print("\n[3/5] Loading models...")

    print(f"  - Loading Qwen3-VL: {args.qwen_model}")
    qwen_client = Qwen3VLClient(
        model_path=args.qwen_model,
        device=args.device
    )

    print(f"  - Loading CLIP: {args.clip_model}")
    clip_config = CLIPConfig(
        model_name=args.clip_model,
        device=args.device
    )
    clip_encoder = CLIPEncoder(clip_config)

    if args.mock_flux:
        print(f"  - Using Mock FLUX (no actual editing)")
        flux_client = MockFluxClient()
    else:
        print(f"  - Loading FLUX.1-dev: {args.flux_model}")
        flux_config = FluxConfig(
            model_name=args.flux_model,
            device=args.device
        )
        flux_client = FluxInpaintClient(flux_config)

    # Create pipeline
    print("\n[4/5] Creating DeCIR pipeline...")
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
    print("\n[5/5] Running inference...")

    all_predictions = {}
    query_embeddings_list = []

    for idx, sample in enumerate(tqdm(dataset, desc="Processing queries")):
        pair_id = sample['pair_id']

        # Skip if exists and skip_existing is True
        if args.skip_existing:
            pred_file = output_dir / f"{pair_id}_prediction.json"
            if pred_file.exists():
                continue

        try:
            # Get paths
            ref_img_path = dataset.get_image_path(sample['reference'])
            caption = sample['caption']

            # Run DeCIR pipeline
            query_embedding = pipeline(
                reference_image=str(ref_img_path),
                modification_text=caption
            )

            # Store query embedding
            query_embeddings_list.append(query_embedding)

            # Compute similarities with gallery
            similarities = compute_similarities(query_embedding, gallery_embeddings)

            # Rank gallery
            ranked_indices = rank_gallery(similarities, top_k=args.top_k)

            # Store predictions
            all_predictions[pair_id] = ranked_indices

        except Exception as e:
            print(f"\nError processing {pair_id}: {e}")
            # Use zeros as fallback
            query_embeddings_list.append(np.zeros(gallery_embeddings.shape[1]))
            all_predictions[pair_id] = list(range(args.top_k))

    # Save results
    print("\n" + "="*60)
    print("Saving results...")

    # Save all query embeddings
    query_embeddings_array = np.array(query_embeddings_list)
    query_emb_path = output_dir / "query_embeddings.npy"
    np.save(query_emb_path, query_embeddings_array)
    print(f"  - Query embeddings: {query_emb_path}")

    # Save predictions
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"  - Predictions: {predictions_path}")

    # Create submission file (CIRR server format)
    submission = []
    for pair_id, ranked_indices in all_predictions.items():
        # Get image names from indices
        # Note: You may need to map indices to actual image names
        # based on your gallery structure
        submission.append({
            "query_id": pair_id,
            "ranking": ranked_indices  # or map to image names
        })

    submission_path = output_dir / "submission.json"
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"  - Submission file: {submission_path}")

    print("\n" + "="*60)
    print("✓ Done!")
    print(f"Processed {len(all_predictions)} queries")
    print(f"Results saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Evaluate: python scripts/evaluation/evaluate_cirr.py --predictions {predictions_path}")
    print(f"  2. Submit to CIRR server: {submission_path}")
    print("="*60)


if __name__ == '__main__':
    main()
