"""Image operation utilities for DeCIR.

This module provides utility functions for image manipulation, including:
- Bounding box transformations
- Mask generation and manipulation
- Image cropping and resizing
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Union
import numpy as np
from PIL import Image


def load_image(image_path: str, mode: str = "RGB") -> Image.Image:
    """Load an image from disk.

    Args:
        image_path: Path to the image file.
        mode: PIL image mode (default: "RGB").

    Returns:
        PIL Image object.

    Raises:
        FileNotFoundError: If image file does not exist.
    """
    return Image.open(image_path).convert(mode)


def save_image(image: Image.Image, save_path: str) -> None:
    """Save an image to disk.

    Args:
        image: PIL Image object to save.
        save_path: Destination file path.
    """
    image.save(save_path)


def get_image_hw(image: Union[Image.Image, np.ndarray]) -> Tuple[int, int]:
    """Get image height and width.

    Args:
        image: PIL Image or numpy array.

    Returns:
        Tuple of (height, width).
    """
    if isinstance(image, Image.Image):
        w, h = image.size
        return h, w
    arr = np.asarray(image)
    return arr.shape[0], arr.shape[1]


def bbox1000_to_xyxy_pixels(
    bbox_1000: List[int],
    image: Union[Image.Image, np.ndarray]
) -> Optional[Tuple[int, int, int, int]]:
    """Convert normalized bbox (0-1000 scale) to pixel coordinates.

    This function converts bounding boxes in the normalized 1000-scale format
    (commonly used by vision-language models) to pixel coordinates based on
    the actual image dimensions.

    Args:
        bbox_1000: Bounding box in format [x1, y1, x2, y2] with values in [0, 1000].
        image: Reference image (PIL Image or numpy array) for dimensions.

    Returns:
        Pixel coordinates as (x1, y1, x2, y2), or None if invalid bbox.

    Example:
        >>> bbox = [100, 200, 500, 800]  # Normalized bbox
        >>> img = Image.open("example.jpg")  # 512x512 image
        >>> bbox_pixels = bbox1000_to_xyxy_pixels(bbox, img)
        >>> print(bbox_pixels)  # (51, 102, 256, 409)
    """
    if not isinstance(bbox_1000, (list, tuple)) or len(bbox_1000) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_1000]
    except (ValueError, TypeError):
        return None

    # Validate bbox
    if x2 <= x1 or y2 <= y1:
        return None

    h, w = get_image_hw(image)

    # Clamp to valid normalized range [0, 1000]
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))

    # Convert to pixel coordinates
    px1 = int(round(x1 / 1000.0 * w))
    py1 = int(round(y1 / 1000.0 * h))
    px2 = int(round(x2 / 1000.0 * w))
    py2 = int(round(y2 / 1000.0 * h))

    # Clamp to image bounds
    px1 = max(0, min(w - 1, px1))
    py1 = max(0, min(h - 1, py1))
    px2 = max(0, min(w, px2))
    py2 = max(0, min(h, py2))

    # Validate pixel bbox
    if px2 <= px1 or py2 <= py1:
        return None

    return (px1, py1, px2, py2)


def rect_mask_from_bbox(
    bbox_xyxy: Tuple[int, int, int, int],
    image: Union[Image.Image, np.ndarray]
) -> np.ndarray:
    """Create a rectangular mask from bounding box.

    Args:
        bbox_xyxy: Bounding box in pixel coordinates (x1, y1, x2, y2).
        image: Reference image for dimensions.

    Returns:
        Binary mask as uint8 numpy array with values 0 or 255.
    """
    h, w = get_image_hw(image)
    x1, y1, x2, y2 = bbox_xyxy
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def union_masks(masks: List[np.ndarray]) -> Optional[np.ndarray]:
    """Compute union of multiple binary masks.

    Args:
        masks: List of binary masks (uint8 numpy arrays).

    Returns:
        Union mask, or None if input list is empty.
    """
    if not masks:
        return None

    out = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out = np.maximum(out, m.astype(np.uint8))
    return out


def dilate_mask(mask: np.ndarray, dilation_ratio: float) -> np.ndarray:
    """Dilate a binary mask based on its size.

    The dilation kernel size is computed as dilation_ratio * max(bbox_height, bbox_width),
    which ensures the dilation is proportional to the mask size.

    Args:
        mask: Binary mask (uint8 numpy array with values 0/255).
        dilation_ratio: Ratio of dilation relative to mask size (e.g., 0.1 for 10%).

    Returns:
        Dilated mask with same shape as input.

    Note:
        Uses OpenCV for efficient dilation. Falls back to naive implementation
        if OpenCV is not available.
    """
    if mask is None:
        return None
    if dilation_ratio <= 0:
        return mask

    # Locate mask bounding box
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return mask

    # Compute kernel size based on mask dimensions
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    base = max(h, w)
    k = int(round(base * dilation_ratio))

    # Ensure minimum kernel size
    if k < 3:
        k = 3
    # Ensure odd kernel size
    if k % 2 == 0:
        k += 1

    try:
        import cv2
        kernel = np.ones((k, k), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    except ImportError:
        # Fallback: naive dilation via max filter
        pad = k // 2
        padded = np.pad(
            mask.astype(np.uint8),
            ((pad, pad), (pad, pad)),
            mode="constant"
        )
        out = np.zeros_like(mask, dtype=np.uint8)
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                window = padded[y:y + k, x:x + k]
                out[y, x] = 255 if window.max() > 0 else 0
        return out


def crop_region(
    image: Union[Image.Image, np.ndarray],
    bbox_xyxy: Tuple[int, int, int, int]
) -> Union[Image.Image, np.ndarray]:
    """Crop a region from an image using bounding box.

    Args:
        image: Input image (PIL Image or numpy array).
        bbox_xyxy: Bounding box in pixel coordinates (x1, y1, x2, y2).

    Returns:
        Cropped image region (same type as input).
    """
    x1, y1, x2, y2 = bbox_xyxy

    if isinstance(image, Image.Image):
        return image.crop((x1, y1, x2, y2))
    else:
        return image[y1:y2, x1:x2]


def resize_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int],
    resample: int = Image.BILINEAR
) -> Union[Image.Image, np.ndarray]:
    """Resize image to target size.

    Args:
        image: Input image (PIL Image or numpy array).
        target_size: Target (width, height).
        resample: Resampling filter (for PIL Images).

    Returns:
        Resized image (same type as input).
    """
    if isinstance(image, Image.Image):
        return image.resize(target_size, resample=resample)
    else:
        import cv2
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
