"""FLUX.1-dev Inpainting client for DeCIR.

This module provides a wrapper around FLUX.1-dev Fill pipeline
for localized image editing based on text prompts using rectified flow.

FLUX.1-dev achieves superior inpainting quality compared to SDXL-Inpaint
as demonstrated in the DeCIR paper.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image


@dataclass
class FluxConfig:
    """Configuration for FLUX.1-dev inpainting.

    Attributes:
        model_name: HuggingFace model name or local path.
        device: Device for inference ("cuda", "cpu", or "auto").
        dtype: Data type for inference ("fp16", "bf16", or "fp32").
        num_inference_steps: Number of denoising steps (default: 50).
        guidance_scale: Classifier-free guidance scale (default: 30.0).
        max_sequence_length: Maximum sequence length for prompts (default: 512).
        seed: Random seed for reproducibility (default: None).
        enable_vae_slicing: Whether to enable VAE slicing for memory efficiency (default: True).
        device_map: Device map strategy for multi-GPU ("balanced", "auto", or None).
    """
    model_name: str = "black-forest-labs/FLUX.1-dev"
    device: str = "cuda"
    dtype: str = "fp16"
    num_inference_steps: int = 50
    guidance_scale: float = 30.0
    max_sequence_length: int = 512
    seed: Optional[int] = None
    enable_vae_slicing: bool = True
    device_map: Optional[str] = "balanced"


class FluxInpaintClient:
    """FLUX.1-dev Inpainting client for localized image editing.

    This client uses FLUX.1-dev Fill pipeline to modify specific
    regions of images based on text prompts and masks. FLUX uses
    rectified flow transformers for superior quality.

    Attributes:
        config: FLUX configuration.
        pipeline: Diffusers FLUX Fill pipeline.

    Example:
        >>> config = FluxConfig(
        ...     model_name="black-forest-labs/FLUX.1-dev",
        ...     device="cuda",
        ...     guidance_scale=30.0,
        ...     seed=42
        ... )
        >>> client = FluxInpaintClient(config)
        >>> edited_img = client.inpaint(
        ...     image=original_image,
        ...     mask=edit_mask,
        ...     prompt="a blue car"
        ... )
    """

    def __init__(self, config: FluxConfig):
        """Initialize FLUX inpainting client.

        Args:
            config: FLUX configuration object.
        """
        self.config = config
        self._pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize FLUX Fill pipeline."""
        import torch
        from diffusers import FluxFillPipeline

        # Map dtype string to torch dtype
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float16)

        # Determine device
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load pipeline
        # FLUX supports device_map for multi-GPU setups
        if self.config.device_map and device == "cuda":
            self._pipeline = FluxFillPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
                device_map=self.config.device_map,
            )
        else:
            self._pipeline = FluxFillPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
            ).to(device)

        # Enable VAE slicing for memory efficiency
        if self.config.enable_vae_slicing:
            try:
                self._pipeline.enable_vae_slicing()
            except Exception as e:
                print(f"[WARNING] Failed to enable VAE slicing: {e}")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        max_sequence_length: Optional[int] = None,
    ) -> Image.Image:
        """Perform inpainting on masked image region using FLUX.

        This method edits the masked region of the input image according to
        the text prompt while preserving unmasked regions using rectified flow.

        Args:
            image: Input image (PIL RGB Image).
            mask: Binary mask (PIL L Image with 0-255 values).
                  White (255) = editable region, Black (0) = keep original.
            prompt: Text description of desired edit.
            negative_prompt: Text description to avoid (optional, not heavily used in FLUX).
            num_inference_steps: Override config num_inference_steps (optional).
            guidance_scale: Override config guidance_scale (optional).
            max_sequence_length: Override config max_sequence_length (optional).

        Returns:
            Edited image as PIL RGB Image.

        Raises:
            ValueError: If inputs are invalid.

        Example:
            >>> # Edit a car's color using FLUX
            >>> original = Image.open("car.jpg")
            >>> mask = Image.open("car_mask.png")  # White on car, black elsewhere
            >>> edited = client.inpaint(
            ...     image=original,
            ...     mask=mask,
            ...     prompt="a blue sports car",
            ...     guidance_scale=30.0
            ... )
            >>> edited.save("blue_car.jpg")

        Note:
            - Input image and mask should have the same dimensions
            - Mask should be single-channel (grayscale)
            - FLUX uses higher guidance_scale (30.0) compared to SDXL (7.5)
            - FLUX uses rectified flow for superior quality
        """
        # Validate inputs
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        if not isinstance(mask, Image.Image):
            raise ValueError(f"Expected PIL Image mask, got {type(mask)}")

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Ensure mask is single-channel
        if mask.mode != "L":
            mask = mask.convert("L")

        # Check dimensions match
        if image.size != mask.size:
            raise ValueError(
                f"Image size {image.size} does not match mask size {mask.size}"
            )

        # Use config defaults if not overridden
        num_steps = num_inference_steps or self.config.num_inference_steps
        guidance = guidance_scale or self.config.guidance_scale
        max_seq_len = max_sequence_length or self.config.max_sequence_length

        # Set up generator for reproducibility
        import torch
        generator = None
        if self.config.seed is not None:
            # For multi-GPU setups, generator should be on CPU
            device = "cpu" if self.config.device_map else self.config.device
            generator = torch.Generator(device=device).manual_seed(
                self.config.seed
            )

        # Run inpainting with FLUX Fill pipeline
        output = self._pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance,
            num_inference_steps=num_steps,
            max_sequence_length=max_seq_len,
            generator=generator,
        )

        # Return first image from output
        return output.images[0]


class MockFluxClient:
    """Mock FLUX client for testing without loading real models.

    This client returns the original image unmodified, useful for testing
    pipelines without GPU resources or waiting for FLUX generation.

    Example:
        >>> client = MockFluxClient()
        >>> edited = client.inpaint(image, mask, "test prompt")
        >>> # Returns original image unchanged
    """

    def __init__(self):
        """Initialize mock client."""
        pass

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Image.Image:
        """Return original image without modification.

        Args:
            image: Input image.
            mask: Mask (ignored).
            prompt: Prompt (ignored).
            negative_prompt: Negative prompt (ignored).
            **kwargs: Additional arguments (ignored).

        Returns:
            Original input image.
        """
        print(f"[MOCK] FLUX inpaint called with prompt: '{prompt}'")
        return image.copy()
