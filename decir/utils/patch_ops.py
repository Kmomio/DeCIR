"""Visual delta patch extraction utilities for DeCIR.

This module provides functions to extract visual delta patches - the semantic
difference between reference and edited images localized to the modification region.
"""

from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image


def _to_pil_rgb(image) -> Image.Image:
    """Convert image to PIL RGB format.

    Args:
        image: PIL Image or numpy array.

    Returns:
        PIL Image in RGB mode.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _mask_union_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Compute bounding box of mask region.

    Args:
        mask: Binary mask (HxW uint8 array with values 0/255).

    Returns:
        Bounding box (x1, y1, x2, y2) in pixel coordinates, or None if mask is empty.

    Note:
        Returns bbox in inclusive-exclusive format: [x1, x2), [y1, y2).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2)


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    w: int,
    h: int,
    expand_ratio: float
) -> Tuple[int, int, int, int]:
    """Expand bounding box by a ratio of its size.

    This function expands the bbox symmetrically to include context around
    the edit region, which is important for capturing semantic changes.

    Args:
        bbox: Input bounding box (x1, y1, x2, y2).
        w: Image width for boundary clamping.
        h: Image height for boundary clamping.
        expand_ratio: Expansion ratio relative to max(bbox_width, bbox_height).

    Returns:
        Expanded bounding box clamped to image boundaries.

    Example:
        >>> bbox = (100, 100, 200, 200)  # 100x100 box
        >>> expanded = _expand_bbox(bbox, 512, 512, expand_ratio=0.2)
        >>> # Expands by 20 pixels on each side: (80, 80, 220, 220)
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    # Compute padding based on max dimension
    pad = int(round(max(bw, bh) * expand_ratio))

    # Expand and clamp to image boundaries
    nx1 = max(0, x1 - pad)
    ny1 = max(0, y1 - pad)
    nx2 = min(w, x2 + pad)
    ny2 = min(h, y2 + pad)

    return (nx1, ny1, nx2, ny2)


def extract_visual_delta_patch(
    ref_image: Image.Image,
    edited_image: Image.Image,
    masks: List[np.ndarray],
    expand_ratio: float = 0.15,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """Extract visual delta patches from reference and edited images.

    This is a core function for DeCIR's dual-modal fusion. It extracts
    corresponding patches from the reference and edited images, localized
    to the modification region. The visual delta (edit_patch - ref_patch)
    captures the semantic change introduced by the edit.

    Strategy:
        1. Union all provided masks to get overall edit region
        2. Compute bounding box of union mask
        3. Expand bbox by expand_ratio to include context
        4. Crop both images using the same bbox

    Args:
        ref_image: Reference image (PIL Image or numpy array).
        edited_image: Edited image (same size as reference).
        masks: List of binary masks indicating edit regions.
        expand_ratio: Bbox expansion ratio (default: 0.15 = 15%).

    Returns:
        Tuple of (ref_patch, edit_patch, bbox_xyxy):
            - ref_patch: Cropped patch from reference image
            - edit_patch: Cropped patch from edited image
            - bbox_xyxy: Pixel coordinates (x1, y1, x2, y2) of the patch

    Example:
        >>> ref_img = Image.open("reference.jpg")
        >>> edit_img = Image.open("edited.jpg")
        >>> masks = [mask1, mask2]  # Binary masks
        >>> ref_patch, edit_patch, bbox = extract_visual_delta_patch(
        ...     ref_img, edit_img, masks, expand_ratio=0.2
        ... )
        >>> # Use patches for computing visual delta embedding
        >>> delta_emb = clip_encode(edit_patch) - clip_encode(ref_patch)
    """
    ref_pil = _to_pil_rgb(ref_image)
    edit_pil = _to_pil_rgb(edited_image)
    w, h = ref_pil.size

    if not masks:
        # Fallback: use full image if no masks provided
        bbox = (0, 0, w, h)
        return ref_pil, edit_pil, bbox

    # Compute union of all masks
    union = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        union = np.maximum(union, m.astype(np.uint8))

    # Get bounding box of union mask
    bbox = _mask_union_bbox(union)
    if bbox is None:
        # Fallback if mask is empty
        bbox = (0, 0, w, h)

    # Expand bbox to include context
    bbox = _expand_bbox(bbox, w=w, h=h, expand_ratio=expand_ratio)

    # Extract patches
    ref_patch = ref_pil.crop(bbox)
    edit_patch = edit_pil.crop(bbox)

    return ref_patch, edit_patch, bbox
