"""Stage 2B: Region Caption for DeCIR.

Generates textual descriptions of localized image regions.
"""

from typing import Union, List, Dict
from pathlib import Path
from PIL import Image

from decir.models.qwen_client import Qwen3VLClient
from decir.utils.image_ops import bbox1000_to_xyxy_pixels, crop_region


class RegionCaption:
    """Stage 2B: Region Caption.

    Generates textual descriptions of image regions defined by bounding boxes.

    Example:
        >>> captioner = RegionCaption(qwen_client)
        >>> description = captioner.caption(
        ...     image="photo.jpg",
        ...     bbox=[100, 150, 500, 600]  # Normalized 1000-scale
        ... )
        >>> print(description)
        "A red sports car parked on the street"
    """

    def __init__(self, qwen_client: Qwen3VLClient):
        """Initialize Region Caption stage.

        Args:
            qwen_client: Initialized Qwen3-VL client.
        """
        self.client = qwen_client

    def caption(
        self,
        image: Union[str, Path, Image.Image],
        bbox: List[int]
    ) -> str:
        """Generate caption for image region.

        Args:
            image: Input image (path or PIL Image).
            bbox: Bounding box in normalized 1000-scale [x1, y1, x2, y2].

        Returns:
            Textual description of the region.
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Convert bbox to pixel coordinates
        bbox_pixels = bbox1000_to_xyxy_pixels(bbox, image)
        if bbox_pixels is None:
            return "empty region"

        # Crop region
        region = crop_region(image, bbox_pixels)

        # Generate caption
        try:
            caption = self.client.generate(
                system_prompt="You are a precise image captioning assistant.",
                user_text="Describe what you see in this image region in one concise sentence.",
                image=region
            )
            return caption.strip()
        except Exception as e:
            print(f"[ERROR] Region captioning failed: {e}")
            return "unknown region"


class MockRegionCaption:
    """Mock region captioner for testing."""

    def caption(self, image: Union[str, Path, Image.Image], bbox: List[int]) -> str:
        """Generate mock caption."""
        return f"mock_caption_for_bbox_{bbox[0]}_{bbox[1]}"
