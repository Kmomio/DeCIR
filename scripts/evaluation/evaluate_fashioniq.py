#!/usr/bin/env python
"""
Evaluate FashionIQ submission results.

This script computes Recall@K metrics for FashionIQ predictions.

Usage:
    python scripts/evaluation/evaluate_fashioniq.py \
        --submission-file outputs/fashioniq_dress/predictions.json \
        --dataset-path data/fashioniq/captions/cap.dress.test.json \
        --top-k 10 50
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple


def load_ground_truth_dataset(dataset_path: str) -> List[Dict]:
    """
    Load the ground truth dataset JSON file.
    Expected format:
    [
        {
            "target": "B0084Y8XIU",
            "candidate": "B005X4PL1G",
            "captions": ["caption1", "caption2"]
        },
        ...
    ]

    :param dataset_path: Path to the dataset JSON file
    :return: List of dataset samples
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def evaluate_submission(
    submission_dict: Dict[str, List[str]],
    dataset_samples: List[Dict],
    top_k: Tuple[int, ...] = (1, 5, 10, 50)
) -> Dict[int, float]:
    """
    Evaluate a submission dictionary against ground truth.

    :param submission_dict: Dictionary mapping query index (as string) to list of ranked image IDs
    :param dataset_samples: List of ground truth dataset samples with 'target' and 'candidate' fields
    :param top_k: Tuple of k values for recall@k computation
    :return: Dictionary mapping k to recall@k percentage
    """
    num_samples = len(dataset_samples)
    recall_metrics = {k: 0.0 for k in top_k}

    correct_counts = {k: 0 for k in top_k}

    for query_idx_str, ranked_image_ids in submission_dict.items():
        # Convert query index to int
        query_idx = int(query_idx_str)

        # Check if query index is valid
        if query_idx >= len(dataset_samples):
            print(f"Warning: query index {query_idx} is out of range for dataset of size {len(dataset_samples)}")
            continue

        # Get the target image ID for this query
        target_image_id = dataset_samples[query_idx]["target"]

        # For each k, check if target image ID is in the top-k predictions
        for k in top_k:
            if target_image_id in ranked_image_ids[:k]:
                correct_counts[k] += 1

    # Calculate recall@k percentages
    for k in top_k:
        recall_metrics[k] = (correct_counts[k] / num_samples) * 100.0

    return recall_metrics


def main():
    parser = ArgumentParser(description="Evaluate FashionIQ submission file against ground truth dataset")
    parser.add_argument("--submission-file", type=str, required=True,
                        help="Path to the submission JSON file")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the ground truth dataset JSON file")
    parser.add_argument("--top-k", type=int, nargs='+', default=[1, 5, 10, 50],
                        help="List of k values for recall@k computation (default: 1 5 10 50)")

    args = parser.parse_args()

    # Load submission file
    print(f"Loading submission file: {args.submission_file}")
    with open(args.submission_file, 'r') as f:
        submission_dict = json.load(f)

    # Load ground truth dataset
    print(f"Loading dataset: {args.dataset_path}")
    dataset_samples = load_ground_truth_dataset(args.dataset_path)

    print(f"Dataset contains {len(dataset_samples)} samples")
    print(f"Submission contains {len(submission_dict)} predictions")

    # Validate that submission covers all dataset samples
    expected_queries = set(str(i) for i in range(len(dataset_samples)))
    submission_queries = set(submission_dict.keys())

    if submission_queries != expected_queries:
        missing_queries = expected_queries - submission_queries
        extra_queries = submission_queries - expected_queries

        if missing_queries:
            print(f"Warning: Missing predictions for {len(missing_queries)} queries: {sorted(list(missing_queries))[:10]}...")
        if extra_queries:
            print(f"Warning: Extra predictions for {len(extra_queries)} queries: {sorted(list(extra_queries))[:10]}...")

    print(f"Evaluating against {len(dataset_samples)} samples")

    # Evaluate submission
    recall_metrics = evaluate_submission(
        submission_dict=submission_dict,
        dataset_samples=dataset_samples,
        top_k=tuple(args.top_k)
    )

    # Print results
    print("\nFashionIQ Evaluation Results:")
    print("=" * 40)
    for k in sorted(recall_metrics.keys()):
        print(f"Recall@{k:2d}: {recall_metrics[k]:6.2f}%")
    print("=" * 40)

    # Save results to file
    results_file = Path(args.submission_file).parent / f"{Path(args.submission_file).stem}_evaluation_results.json"
    results_data = {
        "recall_metrics": recall_metrics,
        "num_queries": len(dataset_samples),
        "num_predictions": len(submission_dict),
        "evaluation_config": {
            "top_k_values": args.top_k,
            "submission_file": args.submission_file,
            "dataset_path": args.dataset_path
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == '__main__':
    main()
