"""Stage 3: Mask Engine for DeCIR.

Generates edit masks based on intent and bounding boxes.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

from decir.utils.image_ops import (
    bbox1000_to_xyxy_pixels,
    rect_mask_from_bbox,
    union_masks,
    dilate_mask,
    get_image_hw
)


class MaskStrategy(Enum):
    """Mask generation strategy."""
    SINGLE_OBJECT = auto()          # Single object mask
    MULTI_OBJECT_UNION = auto()     # Union of multiple objects
    MULTI_OBJECT_SEPARATE = auto()  # Separate masks for each object
    BACKGROUND = auto()             # Inverted mask (everything except objects)
    GLOBAL = auto()                 # Full image mask
    SKIP = auto()                   # No mask needed
    INTERACTION_EXPAND = auto()     # Expanded mask for interaction/add operations


@dataclass
class MaskDecision:
    """Mask generation decision.

    Attributes:
        strategy: Chosen mask strategy.
        dilation_ratio: Mask dilation ratio (0-1).
    """
    strategy: MaskStrategy
    dilation_ratio: float = 0.0


class MaskEngine:
    """Stage 3: Mask Engine.

    Decides mask strategy based on edit intent and generates corresponding masks.

    Example:
        >>> engine = MaskEngine()
        >>> masks = engine.generate_masks(
        ...     image="photo.jpg",
        ...     bboxes=[{"bbox": [100, 150, 500, 600]}],
        ...     intent={"edit_type": "subject_only"}
        ... )
    """

    def decide_strategy(self, intent: Dict, bboxes: List[Dict]) -> MaskDecision:
        """Decide mask generation strategy.

        Args:
            intent: Parsed edit intent.
            bboxes: List of grounded bounding boxes.

        Returns:
            MaskDecision with strategy and dilation ratio.
        """
        edit_type = (intent.get("edit_type") or "").lower()

        # Global edits
        if edit_type == "global" or intent.get("needs_global_redraw", False):
            return MaskDecision(MaskStrategy.GLOBAL, 0.0)

        # Background edits
        if edit_type == "background_only" or intent.get("needs_background_mask", False):
            return MaskDecision(MaskStrategy.BACKGROUND, 0.0)

        # Check for interaction/add operations
        if self._has_interaction_or_add(intent):
            return MaskDecision(MaskStrategy.INTERACTION_EXPAND, 0.40)

        # Multiple objects
        if len(bboxes) > 1:
            return MaskDecision(MaskStrategy.MULTI_OBJECT_UNION, 0.10)

        # Single object
        if len(bboxes) == 1:
            return MaskDecision(MaskStrategy.SINGLE_OBJECT, 0.05)

        # No bboxes
        return MaskDecision(MaskStrategy.SKIP, 0.0)

    def generate_masks(
        self,
        image: Image.Image,
        bboxes: List[Dict],
        intent: Dict
    ) -> List[np.ndarray]:
        """Generate edit masks.

        Args:
            image: Reference image.
            bboxes: List of bounding boxes (normalized 1000-scale).
            intent: Parsed edit intent.

        Returns:
            List of binary masks (uint8 numpy arrays with 0/255 values).
        """
        decision = self.decide_strategy(intent, bboxes)

        # Global mask
        if decision.strategy == MaskStrategy.GLOBAL:
            h, w = get_image_hw(image)
            mask = np.ones((h, w), dtype=np.uint8) * 255
            return [mask]

        # Skip
        if decision.strategy == MaskStrategy.SKIP or not bboxes:
            return []

        # Convert bboxes to pixel coordinates
        masks = []
        for bbox_dict in bboxes:
            bbox_1000 = bbox_dict.get("bbox")
            if not bbox_1000:
                continue

            bbox_pixels = bbox1000_to_xyxy_pixels(bbox_1000, image)
            if bbox_pixels is None:
                continue

            mask = rect_mask_from_bbox(bbox_pixels, image)
            masks.append(mask)

        if not masks:
            return []

        # Apply strategy
        if decision.strategy == MaskStrategy.SINGLE_OBJECT:
            result_mask = masks[0]

        elif decision.strategy in [MaskStrategy.MULTI_OBJECT_UNION, MaskStrategy.INTERACTION_EXPAND]:
            result_mask = union_masks(masks)

        elif decision.strategy == MaskStrategy.BACKGROUND:
            # Invert: mask everything except objects
            union_mask = union_masks(masks)
            result_mask = 255 - union_mask

        else:
            result_mask = masks[0] if masks else None

        # Apply dilation
        if result_mask is not None and decision.dilation_ratio > 0:
            result_mask = dilate_mask(result_mask, decision.dilation_ratio)

        return [result_mask] if result_mask is not None else []

    @staticmethod
    def _has_interaction_or_add(intent: Dict) -> bool:
        """Check if intent contains interaction or add operations."""
        operations = intent.get("operations", [])
        interaction_keywords = {"hug", "hugging", "hold", "holding", "ride", "riding", "next to", "beside"}

        for op in operations:
            # Check action
            if op.get("action") == "add":
                return True

            # Check attributes for interaction keywords
            attributes = op.get("attributes", {}) or {}
            for value in attributes.values():
                if isinstance(value, str):
                    if any(kw in value.lower() for kw in interaction_keywords):
                        return True

        return False


class MockMaskEngine:
    """Mock mask engine for testing."""

    def decide_strategy(self, intent: Dict, bboxes: List[Dict]) -> MaskDecision:
        """Decide mock strategy."""
        if len(bboxes) == 0:
            return MaskDecision(MaskStrategy.SKIP, 0.0)
        elif len(bboxes) == 1:
            return MaskDecision(MaskStrategy.SINGLE_OBJECT, 0.05)
        else:
            return MaskDecision(MaskStrategy.MULTI_OBJECT_UNION, 0.10)

    def generate_masks(
        self,
        image: Image.Image,
        bboxes: List[Dict],
        intent: Dict
    ) -> List[np.ndarray]:
        """Generate mock masks."""
        if not bboxes:
            return []

        h, w = get_image_hw(image)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h//4:3*h//4, w//4:3*w//4] = 255  # Center region
        return [mask]
