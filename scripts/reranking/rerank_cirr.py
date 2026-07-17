#!/usr/bin/env python
"""
Rerank CIRR retrieval results using Qwen3-VL MLLM.

This script takes initial retrieval results and uses a multi-modal LLM
to rerank the top-K candidates for improved accuracy.

Usage:
    python scripts/reranking/rerank_cirr.py \
        --data_root data/cirr \
        --predictions outputs/cirr_test/predictions.json \
        --output_dir outputs/cirr_test_reranked \
        --split test \
        --grid_k 4 \
        --qwen_model Qwen/Qwen2-VL-7B-Instruct
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

matplotlib.use("Agg")


def load_qwen_model(model_path: str, device: str = "cuda") -> Tuple:
    """Load Qwen3-VL model for reranking.

    Args:
        model_path: Path to Qwen model
        device: Device for inference

    Returns:
        (model, processor) tuple
    """
    print(f"Loading Qwen3-VL model from {model_path}...")
    t_start = time.time()

    # Configure device map for multi-GPU
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    max_memory = None

    if n_gpus >= 3:
        # Optimized 3-GPU setup
        max_memory = {
            0: "4GiB",   # Visual encoder
            1: "4GiB",   # Buffer
            2: "30GiB",  # Main model
        }
    elif n_gpus == 2:
        max_memory = {0: "4GiB", 1: "28GiB"}

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="eager"  # Stable mode
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    t_end = time.time()
    print(f"Model loaded successfully! ({t_end - t_start:.2f}s)")
    return model, processor


def create_grid_image(
    candidate_names: List[str],
    img_name_to_path: Dict[str, Path],
    output_path: str,
    grid_k: int = 4,
) -> str:
    """Create a grid visualization of top-K candidates.

    Args:
        candidate_names: List of image names (top-K from retrieval)
        img_name_to_path: Mapping from image name to file path
        output_path: Where to save the grid image
        grid_k: Grid size (k x k)

    Returns:
        Path to saved grid image
    """
    max_candidates = grid_k * grid_k
    num_candidates = min(len(candidate_names), max_candidates)
    actual_names = candidate_names[:num_candidates]

    # Color palette for borders
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan',
              'magenta', 'gold', 'lime', 'teal', 'navy', 'coral', 'olive', 'indigo']

    fig_size = max(10.0, grid_k * 3.5)
    fig, axes = plt.subplots(grid_k, grid_k, figsize=(fig_size, fig_size))

    if grid_k == 1:
        axes = [axes]
    else:
        axes = axes.flat

    for idx, ax in enumerate(axes):
        if idx < len(actual_names):
            img_name = actual_names[idx]
            try:
                if img_name not in img_name_to_path:
                    ax.axis('off')
                    continue

                img_path = img_name_to_path[img_name]
                img = cv2.imread(str(img_path))
                if img is None:
                    ax.axis('off')
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.axis('off')

                # Add colored border
                color = colors[idx % len(colors)]
                rect = patches.Rectangle(
                    (0, 0), img.shape[1], img.shape[0],
                    linewidth=6, edgecolor=color, facecolor='none',
                    transform=ax.transData
                )
                ax.add_patch(rect)

                # Add index label
                ax.text(
                    0.02, 0.98, str(idx),
                    color=color, fontsize=24, fontweight='bold',
                    ha='left', va='top', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=color,
                              linewidth=2, boxstyle='round,pad=0.4'),
                )
            except Exception as e:
                print(f"Warning: Failed to load image {img_name}: {e}")
                ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def create_rerank_prompt(modification_text: str) -> str:
    """Create prompt for MLLM reranking.

    Args:
        modification_text: Modification caption

    Returns:
        Prompt string
    """
    prompt = f'''I want you help me do the composed image retrieval. Given the reference image (first image) and modified text: "{modification_text}".
There are some candidates in the second image, please rank all the images in order from most to least likely to be the target image.
**Just respond with the image IDs (numbers), and make sure not to miss any of them.**
'''
    return prompt


def generate_ranking(
    model,
    processor,
    reference_path: str,
    grid_path: str,
    prompt_text: str,
    device: str = "cuda"
) -> str:
    """Generate reranking using Qwen3-VL.

    Args:
        model: Qwen model
        processor: Qwen processor
        reference_path: Path to reference image
        grid_path: Path to grid image
        prompt_text: Reranking prompt
        device: Device for inference

    Returns:
        Model response text
    """
    # Clear cache before inference
    if device == "cuda":
        torch.cuda.empty_cache()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for image retrieval tasks."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": reference_path},
                {"type": "image", "image": grid_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    # Load images
    ref_img = Image.open(reference_path).convert("RGB")
    grid_img = Image.open(grid_path).convert("RGB")

    inputs = processor(
        text=[text_prompt],
        images=[ref_img, grid_img],
        padding=True,
        return_tensors="pt"
    )

    if device == "cuda":
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    # Decode
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs['input_ids'], output_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    return response_text


def parse_ranking_response(response_text: str) -> List[int]:
    """Parse MLLM response to extract ranking indices.

    Args:
        response_text: Model response

    Returns:
        List of indices in ranked order
    """
    if not response_text:
        return []
    try:
        # Extract all numbers from response
        numbers = [int(num) for num in re.findall(r'\d+', response_text)]
        return numbers
    except Exception as e:
        print(f"Error parsing ranking: {e}")
        return []


def apply_reordering(
    original_ranking: List[str],
    new_order: List[int],
    max_candidates: int
) -> List[str]:
    """Apply reordering to original ranking.

    Args:
        original_ranking: Original ranked list
        new_order: New order indices (0-based)
        max_candidates: Number of candidates being reranked

    Returns:
        Reordered ranking
    """
    global_array = np.array(original_ranking)
    num_processed = min(len(original_ranking), max_candidates)

    if num_processed == 0:
        return original_ranking

    # Extract subrange to reorder
    subrange = global_array[:num_processed]

    # Filter valid indices
    unique_order = []
    seen = set()
    for i in new_order:
        if 0 <= i < len(subrange) and i not in seen:
            unique_order.append(i)
            seen.add(i)

    if not unique_order:
        return original_ranking

    # Reorder
    reordered_part = subrange[unique_order]

    # Append remaining (not in new order)
    remaining_indices = [i for i in range(len(subrange)) if i not in unique_order]
    remaining_part = subrange[remaining_indices]

    # Combine
    final_subrange = np.concatenate([reordered_part, remaining_part])
    global_array[:num_processed] = final_subrange

    return global_array.tolist()


def main():
    parser = argparse.ArgumentParser(description="Rerank CIRR results using Qwen3-VL")

    # Dataset arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of CIRR dataset')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to initial predictions JSON file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for reranked results')

    # Model
    parser.add_argument('--qwen_model', type=str,
                        default='Qwen/Qwen2-VL-7B-Instruct',
                        help='Qwen model path')

    # Reranking parameters
    parser.add_argument('--grid_k', type=int, default=4,
                        help='Grid size (k x k candidates to rerank)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    # Checkpointing
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save results every N samples')

    args = parser.parse_args()

    # Setup
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(exist_ok=True)

    print("="*60)
    print("CIRR Reranking with Qwen3-VL")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Predictions: {args.predictions}")
    print(f"Output dir: {output_dir}")
    print(f"Grid size: {args.grid_k}x{args.grid_k}")
    print("="*60)

    # Load data
    print("\n[1/5] Loading CIRR data...")

    # Load annotations
    if args.split == 'val':
        anno_file = data_root / 'cirr' / 'captions' / 'cap.rc2.val.json'
        img_split_file = data_root / 'cirr' / 'image_splits' / 'split.rc2.val.json'
    else:
        anno_file = data_root / 'cirr' / 'captions' / 'cap.rc2.test1.json'
        img_split_file = data_root / 'cirr' / 'image_splits' / 'split.rc2.test1.json'

    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    with open(img_split_file, 'r') as f:
        name_to_relpath = json.load(f)

    # Build image path mapping
    img_name_to_path = {}
    for img_name, rel_path in name_to_relpath.items():
        full_path = data_root / rel_path
        img_name_to_path[img_name] = full_path

    print(f"Loaded {len(annotations)} triplets and {len(img_name_to_path)} images")

    # Load initial predictions
    print("\n[2/5] Loading initial predictions...")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions")

    # Load model
    print("\n[3/5] Loading Qwen3-VL model...")
    model, processor = load_qwen_model(args.qwen_model, args.device)

    # Check for existing results (resumable)
    output_file = output_dir / "reranked_predictions.json"
    if output_file.exists():
        print(f"\nFound existing results at {output_file}")
        with open(output_file, 'r') as f:
            reranked_predictions = json.load(f)
        print(f"Loaded {len(reranked_predictions)} already processed")
    else:
        reranked_predictions = {}

    # Rerank
    print(f"\n[4/5] Reranking predictions...")
    print(f"Processing {len(annotations)} samples ({len(reranked_predictions)} already done)")

    max_candidates = args.grid_k * args.grid_k
    save_counter = 0

    for triplet in tqdm(annotations, desc="Reranking"):
        pair_id = str(triplet.get('pairid', f"{triplet['reference']}_{triplet['target']}"))

        # Skip if already processed
        if pair_id in reranked_predictions:
            continue

        reference_name = triplet['reference']
        caption = triplet['caption']

        # Get initial ranking
        if pair_id in predictions:
            original_ranking = predictions[pair_id]
        else:
            print(f"Warning: pair_id {pair_id} not in predictions, skipping")
            continue

        # Get top-K candidates
        candidates = original_ranking[:max_candidates]

        if not candidates or reference_name not in img_name_to_path:
            reranked_predictions[pair_id] = original_ranking
            continue

        try:
            # Create grid image
            grid_path = grid_dir / f"cirr_pair_{pair_id}_grid.jpg"
            if not grid_path.exists():
                create_grid_image(candidates, img_name_to_path, str(grid_path), args.grid_k)

            # Get reference image path
            reference_path = img_name_to_path[reference_name]

            # Generate reranking
            prompt_text = create_rerank_prompt(caption)
            response_text = generate_ranking(
                model, processor,
                str(reference_path),
                str(grid_path),
                prompt_text,
                args.device
            )

            # Parse and apply reordering
            new_order = parse_ranking_response(response_text)
            if new_order:
                reranked = apply_reordering(original_ranking, new_order, max_candidates)
                reranked_predictions[pair_id] = reranked
            else:
                reranked_predictions[pair_id] = original_ranking

        except Exception as e:
            print(f"\nError processing {pair_id}: {e}")
            reranked_predictions[pair_id] = original_ranking

        finally:
            # Clean up grid image to save disk space
            if grid_path.exists():
                try:
                    os.remove(grid_path)
                except OSError:
                    pass

        # Save checkpoint
        save_counter += 1
        if save_counter % args.save_every == 0:
            with open(output_file, 'w') as f:
                json.dump(reranked_predictions, f, indent=2)

    # Final save
    print("\n[5/5] Saving final results...")
    with open(output_file, 'w') as f:
        json.dump(reranked_predictions, f, indent=2)

    print("\n" + "="*60)
    print("Done!")
    print(f"Reranked {len(reranked_predictions)} predictions")
    print(f"Results saved to: {output_file}")
    print("="*60)


if __name__ == '__main__':
    main()
