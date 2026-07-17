#!/usr/bin/env python
"""
Evaluate CIRR validation set predictions.

Note: CIRR test set evaluation must be done via the official server at:
https://cirr.cecs.anu.edu.au/test_process/

This script only evaluates validation set results locally.

Usage:
    python scripts/evaluation/evaluate_cirr.py \
        --data_root data/cirr \
        --predictions outputs/cirr_val/predictions.json \
        --split val
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple


def load_cirr_annotations(data_root: Path, split: str) -> List[Dict]:
    """
    Load CIRR annotations.

    Args:
        data_root: Root directory of CIRR dataset
        split: Split name ('val' or 'test')

    Returns:
        List of annotation dictionaries
    """
    if split == 'val':
        cap_file = data_root / 'cirr' / 'captions' / 'cap.rc2.val.json'
    elif split == 'test':
        cap_file = data_root / 'cirr' / 'captions' / 'cap.rc2.test1.json'
    else:
        raise ValueError(f"Unknown split: {split}")

    if not cap_file.exists():
        raise FileNotFoundError(f"Caption file not found: {cap_file}")

    with open(cap_file, 'r') as f:
        annotations = json.load(f)

    return annotations


def get_image_id_to_index_map(image_dir: Path) -> Dict[str, int]:
    """
    Create a mapping from image ID to gallery index.

    Args:
        image_dir: Directory containing images

    Returns:
        Dictionary mapping image ID to index
    """
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    image_ids = [f.stem for f in image_files]

    id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}

    return id_to_index


def evaluate_cirr(
    predictions: Dict[str, List[int]],
    annotations: List[Dict],
    image_id_to_index: Dict[str, int],
    top_k: Tuple[int, ...] = (1, 5, 10, 50)
) -> Dict[int, float]:
    """
    Evaluate CIRR predictions.

    Args:
        predictions: Dictionary mapping pair_id to list of ranked gallery indices
        annotations: List of annotation dictionaries
        image_id_to_index: Mapping from image ID to gallery index
        top_k: Tuple of k values for recall@k computation

    Returns:
        Dictionary mapping k to recall@k percentage
    """
    # Create annotation lookup
    anno_dict = {item.get('pairid', f"{item['reference']}_{item['target']}"): item
                 for item in annotations}

    num_samples = len(anno_dict)
    recall_metrics = {k: 0.0 for k in top_k}
    correct_counts = {k: 0 for k in top_k}

    processed = 0

    for pair_id, ranked_indices in predictions.items():
        if pair_id not in anno_dict:
            print(f"Warning: pair_id {pair_id} not found in annotations")
            continue

        anno = anno_dict[pair_id]

        # Get target image ID
        target_id = anno.get('target_hard', anno.get('target'))

        if target_id is None:
            print(f"Warning: No target found for {pair_id}")
            continue

        # Get target index
        if target_id not in image_id_to_index:
            print(f"Warning: Target image {target_id} not found in gallery")
            continue

        target_index = image_id_to_index[target_id]

        # Check if target is in top-k predictions
        for k in top_k:
            if target_index in ranked_indices[:k]:
                correct_counts[k] += 1

        processed += 1

    # Calculate recall@k percentages
    for k in top_k:
        recall_metrics[k] = (correct_counts[k] / processed) * 100.0 if processed > 0 else 0.0

    return recall_metrics, processed


def main():
    parser = ArgumentParser(description="Evaluate CIRR validation set predictions")

    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of CIRR dataset')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Dataset split (only val can be evaluated locally)')
    parser.add_argument('--top-k', type=int, nargs='+', default=[1, 5, 10, 50],
                        help='List of k values for recall@k computation')

    args = parser.parse_args()

    if args.split == 'test':
        print("="*60)
        print("WARNING: CIRR test set evaluation must be done via official server!")
        print("Please submit your results to:")
        print("https://cirr.cecs.anu.edu.au/test_process/")
        print("="*60)
        print("\nContinuing with local evaluation (for format checking)...")

    data_root = Path(args.data_root)

    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # Load annotations
    print(f"Loading CIRR {args.split} set annotations...")
    annotations = load_cirr_annotations(data_root, args.split)
    print(f"Loaded {len(annotations)} annotations")

    # Build image ID to index mapping
    if args.split == 'val':
        image_dir = data_root / 'dev'
    else:
        image_dir = data_root / 'test1'

    print(f"Building image ID to index mapping from: {image_dir}")
    image_id_to_index = get_image_id_to_index_map(image_dir)
    print(f"Found {len(image_id_to_index)} gallery images")

    # Evaluate
    print(f"\nEvaluating predictions...")
    recall_metrics, processed = evaluate_cirr(
        predictions=predictions,
        annotations=annotations,
        image_id_to_index=image_id_to_index,
        top_k=tuple(args.top_k)
    )

    # Print results
    print("\n" + "="*60)
    print(f"CIRR {args.split.upper()} Set Evaluation Results")
    print("="*60)
    print(f"Processed queries: {processed}")
    print("-"*60)
    for k in sorted(recall_metrics.keys()):
        print(f"Recall@{k:2d}: {recall_metrics[k]:6.2f}%")
    print("="*60)

    # Save results
    results_file = Path(args.predictions).parent / f"{Path(args.predictions).stem}_evaluation_results.json"
    results_data = {
        "recall_metrics": recall_metrics,
        "num_queries": processed,
        "num_predictions": len(predictions),
        "split": args.split,
        "evaluation_config": {
            "top_k_values": args.top_k,
            "predictions_file": args.predictions,
            "data_root": str(data_root)
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    if args.split == 'val':
        print("\nNote: These are VALIDATION set results.")
        print("For TEST set evaluation, please submit to the official server.")


if __name__ == '__main__':
    main()
