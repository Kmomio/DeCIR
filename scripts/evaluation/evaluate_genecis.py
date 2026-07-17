#!/usr/bin/env python
"""
Evaluate GeneCIS submission results.

This script computes Recall@K metrics for GeneCIS predictions.

Usage:
    python scripts/evaluation/evaluate_genecis.py \
        --submission-file outputs/genecis/predictions.json \
        --dataset-path data/genecis/test_queries.json \
        --top-k 1 2 3
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple


def load_ground_truth_dataset(dataset_path: str) -> List[Dict]:
    """
    Load the ground truth dataset JSON file.

    :param dataset_path: Path to the dataset JSON file
    :return: List of dataset samples
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def evaluate_submission(
    submission_dict: Dict[str, List[int]],
    dataset_samples: List[Dict],
    top_k: Tuple[int, ...] = (1, 2, 3)
) -> Dict[int, float]:
    """
    Evaluate a submission dictionary against ground truth.

    :param submission_dict: Dictionary mapping pairid to list of ranked gallery indices
    :param dataset_samples: List of ground truth dataset samples
    :param top_k: Tuple of k values for recall@k computation
    :return: Dictionary mapping k to recall@k percentage
    """
    num_samples = len(dataset_samples)
    recall_metrics = {k: 0.0 for k in top_k}

    # The target is always at index 0 in the gallery by construction
    target_rank = 0

    correct_counts = {k: 0 for k in top_k}

    for pairid, ranked_indices in submission_dict.items():
        # Convert pairid to int if it's a string
        pairid_int = int(pairid) if isinstance(pairid, str) else pairid

        # Check if pairid is valid
        if pairid_int >= len(dataset_samples):
            print(f"Warning: pairid {pairid_int} is out of range for dataset of size {len(dataset_samples)}")
            continue

        # For each k, check if target_rank (which is 0) is in the top-k predictions
        for k in top_k:
            if target_rank in ranked_indices[:k]:
                correct_counts[k] += 1

    # Calculate recall@k percentages
    for k in top_k:
        recall_metrics[k] = (correct_counts[k] / num_samples) * 100.0

    return recall_metrics


def main():
    parser = ArgumentParser(description="Evaluate GeneCIS submission file against ground truth dataset")
    parser.add_argument("--submission-file", type=str, required=True,
                        help="Path to the submission JSON file")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the ground truth dataset JSON file")
    parser.add_argument("--top-k", type=int, nargs='+', default=[1, 2, 3],
                        help="List of k values for recall@k computation (default: 1 2 3)")

    args = parser.parse_args()

    # Load submission file
    print(f"Loading submission file: {args.submission_file}")
    with open(args.submission_file, 'r') as f:
        submission_dict = json.load(f)

    # Load ground truth dataset
    print(f"Loading dataset: {args.dataset_path}")
    dataset_samples = load_ground_truth_dataset(args.dataset_path)

    print(f"Evaluating {len(submission_dict)} predictions against {len(dataset_samples)} samples")

    # Evaluate submission
    recall_metrics = evaluate_submission(
        submission_dict=submission_dict,
        dataset_samples=dataset_samples,
        top_k=tuple(args.top_k)
    )

    # Print results
    print("\nGeneCIS Evaluation Results:")
    print("=" * 40)
    for k in sorted(recall_metrics.keys()):
        print(f"Recall@{k}: {recall_metrics[k]:.2f}%")
    print("=" * 40)

    # Save results to file
    results_file = Path(args.submission_file).parent / f"{Path(args.submission_file).stem}_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(recall_metrics, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
