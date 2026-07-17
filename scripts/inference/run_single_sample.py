#!/usr/bin/env python
"""Run DeCIR pipeline on a single sample.

Example usage:
    python scripts/inference/run_single_sample.py \\
        --reference_image examples/car.jpg \\
        --modification_text "change color to blue" \\
        --output_dir outputs/demo
"""

import argparse
from pathlib import Path
import json

from decir.models.qwen_client import Qwen3VLClient
from decir.models.clip_encoder import CLIPEncoder, CLIPConfig
from decir.models.sdxl_client import SDXLInpaintClient, SDXLConfig, MockSDXLClient
from decir.core.pipeline import DeCIRPipeline


def main():
    parser = argparse.ArgumentParser(description="Run DeCIR on a single sample")
    parser.add_argument("--reference_image", type=str, required=True,
                        help="Path to reference image")
    parser.add_argument("--modification_text", type=str, required=True,
                        help="Modification description")
    parser.add_argument("--output_dir", type=str, default="outputs/demo",
                        help="Output directory")

    # Model paths
    parser.add_argument("--qwen_model", type=str,
                        default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Qwen model path")
    parser.add_argument("--clip_model", type=str,
                        default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
                        help="CLIP model name")
    parser.add_argument("--sdxl_model", type=str,
                        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        help="SDXL model name")

    # Pipeline params
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Weight for image features")
    parser.add_argument("--beta", type=float, default=0.2,
                        help="Weight for text features")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Weight for delta features")
    parser.add_argument("--delta_mode", type=str, default="diff",
                        choices=["diff", "patch", "none"],
                        help="Delta computation mode")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--mock_sdxl", action="store_true",
                        help="Use mock SDXL (faster for testing)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DeCIR: Dual-modal Semantic Decoupling")
    print("="*60)
    print(f"Reference: {args.reference_image}")
    print(f"Modification: {args.modification_text}")
    print(f"Output: {output_dir}")
    print("="*60)

    # Initialize models
    print("\n[1/4] Loading models...")
    print(f"  - Qwen3-VL: {args.qwen_model}")
    qwen_client = Qwen3VLClient(
        model_path=args.qwen_model,
        device=args.device
    )

    print(f"  - CLIP: {args.clip_model}")
    clip_config = CLIPConfig(
        model_name=args.clip_model,
        device=args.device
    )
    clip_encoder = CLIPEncoder(clip_config)

    if args.mock_sdxl:
        print(f"  - SDXL: Mock mode (no actual editing)")
        sdxl_client = MockSDXLClient()
    else:
        print(f"  - SDXL: {args.sdxl_model}")
        sdxl_config = SDXLConfig(
            model_name=args.sdxl_model,
            device=args.device
        )
        sdxl_client = SDXLInpaintClient(sdxl_config)

    # Create pipeline
    print("\n[2/4] Creating pipeline...")
    pipeline = DeCIRPipeline(
        qwen_client=qwen_client,
        clip_encoder=clip_encoder,
        sdxl_client=sdxl_client,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta_mode=args.delta_mode
    )

    # Run inference
    print("\n[3/4] Running DeCIR pipeline...")
    query_embedding, intermediate = pipeline(
        reference_image=args.reference_image,
        modification_text=args.modification_text,
        return_intermediate=True
    )

    # Save results
    print("\n[4/4] Saving results...")

    # Save query embedding
    import numpy as np
    np.save(output_dir / "query_embedding.npy", query_embedding)
    print(f"  - Query embedding: {output_dir / 'query_embedding.npy'}")

    # Save edited image
    if intermediate.get('edited_image'):
        edited_path = output_dir / "edited_image.png"
        intermediate['edited_image'].save(edited_path)
        print(f"  - Edited image: {edited_path}")

    # Save intent
    intent_path = output_dir / "intent.json"
    with open(intent_path, 'w') as f:
        json.dump(intermediate.get('intent', {}), f, indent=2)
    print(f"  - Intent: {intent_path}")

    # Save masks
    if intermediate.get('masks'):
        from PIL import Image as PILImage
        for i, mask in enumerate(intermediate['masks']):
            mask_path = output_dir / f"mask_{i}.png"
            PILImage.fromarray(mask).save(mask_path)
            print(f"  - Mask {i}: {mask_path}")

    print("\n" + "="*60)
    print("✓ Done!")
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
