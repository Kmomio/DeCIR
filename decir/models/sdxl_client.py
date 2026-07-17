"""SDXL Inpainting client for DeCIR.

This module provides a wrapper around Stable Diffusion XL inpainting pipeline
for localized image editing based on text prompts.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image


@dataclass
class SDXLConfig:
    """Configuration for SDXL inpainting.

    Attributes:
        model_name: HuggingFace model name or local path.
        device: Device for inference ("cuda", "cpu", or "auto").
        dtype: Data type for inference ("fp16", "bf16", or "fp32").
        num_inference_steps: Number of denoising steps (default: 30).
        guidance_scale: Classifier-free guidance scale (default: 7.5).
        strength: Inpainting strength (0-1, default: 0.99).
        seed: Random seed for reproducibility (default: None).
        enable_xformers: Whether to enable xformers optimization (default: True).
    """
    model_name: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    device: str = "cuda"
    dtype: str = "fp16"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.99
    seed: Optional[int] = None
    enable_xformers: bool = True


class SDXLInpaintClient:
    """SDXL Inpainting client for localized image editing.

    This client uses Stable Diffusion XL inpainting to modify specific
    regions of images based on text prompts and masks.

    Attributes:
        config: SDXL configuration.
        pipeline: Diffusers inpainting pipeline.

    Example:
        >>> config = SDXLConfig(
        ...     model_name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        ...     device="cuda",
        ...     seed=42
        ... )
        >>> client = SDXLInpaintClient(config)
        >>> edited_img = client.inpaint(
        ...     image=original_image,
        ...     mask=edit_mask,
        ...     prompt="a blue car"
        ... )
    """

    def __init__(self, config: SDXLConfig):
        """Initialize SDXL inpainting client.

        Args:
            config: SDXL configuration object.
        """
        self.config = config
        self._pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize SDXL inpainting pipeline."""
        import torch
        from diffusers import AutoPipelineForInpainting

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
        self._pipeline = AutoPipelineForInpainting.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None,
        ).to(device)

        # Enable memory optimizations
        if self.config.enable_xformers and device == "cuda":
            try:
                self._pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[WARNING] Failed to enable xformers: {e}")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
    ) -> Image.Image:
        """Perform inpainting on masked image region.

        This method edits the masked region of the input image according to
        the text prompt while preserving unmasked regions.

        Args:
            image: Input image (PIL RGB Image).
            mask: Binary mask (PIL L Image with 0-255 values).
                  White (255) = editable region, Black (0) = keep original.
            prompt: Text description of desired edit.
            negative_prompt: Text description to avoid (optional).
            num_inference_steps: Override config num_inference_steps (optional).
            guidance_scale: Override config guidance_scale (optional).
            strength: Override config strength (optional).

        Returns:
            Edited image as PIL RGB Image.

        Raises:
            ValueError: If inputs are invalid.

        Example:
            >>> # Edit a car's color
            >>> original = Image.open("car.jpg")
            >>> mask = Image.open("car_mask.png")  # White on car, black elsewhere
            >>> edited = client.inpaint(
            ...     image=original,
            ...     mask=mask,
            ...     prompt="a blue sports car",
            ...     negative_prompt="blurry, low quality"
            ... )
            >>> edited.save("blue_car.jpg")

        Note:
            - Input image and mask should have the same dimensions
            - Mask should be single-channel (grayscale)
            - Higher guidance_scale = stronger adherence to prompt
            - Higher strength = more modification to masked region
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
        inpaint_strength = strength or self.config.strength

        # Set up generator for reproducibility
        import torch
        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(
                self.config.seed
            )

        # Run inpainting
        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance,
            num_inference_steps=num_steps,
            generator=generator,
            strength=inpaint_strength,
        )

        # Return first image from output
        return output.images[0]


class MockSDXLClient:
    """Mock SDXL client for testing without loading real models.

    This client returns the original image unmodified, useful for testing
    pipelines without GPU resources or waiting for diffusion generation.

    Example:
        >>> client = MockSDXLClient()
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
        print(f"[MOCK] SDXL inpaint called with prompt: '{prompt}'")
        return image.copy()
