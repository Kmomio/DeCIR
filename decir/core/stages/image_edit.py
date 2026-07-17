"""Stage 4: Image Editor for DeCIR.

Performs localized image editing using SDXL Inpainting.
"""

from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from decir.models.sdxl_client import SDXLInpaintClient, MockSDXLClient
from decir.utils.image_ops import union_masks


class ImageEditor:
    """Stage 4: Image Editor.

    Applies modifications to image regions using SDXL inpainting based on
    target descriptions and edit masks.

    Example:
        >>> from decir.models.sdxl_client import SDXLConfig, SDXLInpaintClient
        >>> sdxl_config = SDXLConfig(device="cuda")
        >>> sdxl_client = SDXLInpaintClient(sdxl_config)
        >>> editor = ImageEditor(sdxl_client)
        >>> edited_img = editor.edit(
        ...     image="photo.jpg",
        ...     masks=[mask_array],
        ...     target_prompt="a blue car"
        ... )
    """

    def __init__(self, sdxl_client: Union[SDXLInpaintClient, MockSDXLClient]):
        """Initialize Image Editor.

        Args:
            sdxl_client: SDXL inpainting client.
        """
        self.client = sdxl_client

    def edit(
        self,
        image: Union[str, Path, Image.Image],
        masks: List[np.ndarray],
        target_prompt: str,
        negative_prompt: Optional[str] = None
    ) -> Image.Image:
        """Edit image using inpainting.

        Args:
            image: Reference image (path or PIL Image).
            masks: List of binary masks (uint8 arrays with 0/255 values).
            target_prompt: Target description for inpainting.
            negative_prompt: Negative prompt (optional).

        Returns:
            Edited image as PIL Image.

        Example:
            >>> edited = editor.edit(
            ...     image="car.jpg",
            ...     masks=[car_mask],
            ...     target_prompt="a blue sports car",
            ...     negative_prompt="blurry, low quality"
            ... )
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Union all masks
        if not masks:
            # No mask - return original image
            return image.copy()

        union_mask = union_masks(masks)
        if union_mask is None:
            return image.copy()

        # Convert mask to PIL
        mask_pil = Image.fromarray(union_mask).convert("L")

        # Perform inpainting
        try:
            edited_image = self.client.inpaint(
                image=image,
                mask=mask_pil,
                prompt=target_prompt,
                negative_prompt=negative_prompt
            )
            return edited_image
        except Exception as e:
            print(f"[ERROR] Image editing failed: {e}")
            return image.copy()


class MockImageEditor:
    """Mock image editor for testing."""

    def edit(
        self,
        image: Union[str, Path, Image.Image],
        masks: List[np.ndarray],
        target_prompt: str,
        negative_prompt: Optional[str] = None
    ) -> Image.Image:
        """Return original image (mock editing)."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        print(f"[MOCK] Image edit called with prompt: '{target_prompt}'")
        return image.copy()
