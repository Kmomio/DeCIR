"""Stage 2A: Visual Grounding for DeCIR.

This module implements visual grounding - localizing target objects in the
reference image using vision-language models (Qwen3-VL).
"""

import re
import json
from typing import List, Dict, Union, Tuple
from pathlib import Path
from PIL import Image

from decir.models.qwen_client import Qwen3VLClient


def extract_json_list(text: str) -> List[Dict]:
    """Extract JSON list from text response.

    Tries to extract JSON from code fences or raw brackets.

    Args:
        text: Text containing JSON list.

    Returns:
        Parsed JSON list, or empty list if extraction fails.
    """
    # Try code fence first
    fence_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw bracket matching
    bracket_match = re.search(r"\[[\s\S]*?\]", text)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except json.JSONDecodeError:
            pass

    return []


class VisualGrounding:
    """Stage 2A: Visual Grounding.

    Localizes target objects in reference image using Qwen3-VL grounding capabilities.
    Returns bounding boxes in normalized 1000-scale format.

    Attributes:
        qwen_client: Qwen3-VL client for visual grounding.

    Example:
        >>> grounding = VisualGrounding(qwen_client)
        >>> bboxes = grounding.ground(
        ...     image="car.jpg",
        ...     categories=["car", "person"]
        ... )
        >>> print(bboxes)
        [
            {"label": "car", "bbox": [100, 200, 500, 800]},
            {"label": "person", "bbox": [550, 150, 700, 900]}
        ]
    """

    def __init__(self, qwen_client: Qwen3VLClient):
        """Initialize Visual Grounding stage.

        Args:
            qwen_client: Initialized Qwen3-VL client.
        """
        self.client = qwen_client

    def ground(
        self,
        image: Union[str, Path, Image.Image],
        categories: List[str]
    ) -> List[Dict]:
        """Localize objects in image.

        Args:
            image: Reference image (path or PIL Image).
            categories: List of object categories to localize (e.g., ["car", "person"]).

        Returns:
            List of bounding boxes with labels. Each bbox is a dict:
            {"label": str, "bbox": [x1, y1, x2, y2]} where coordinates
            are in normalized 1000-scale.

        Example:
            >>> bboxes = grounding.ground("photo.jpg", ["dog", "cat"])
            >>> for bbox in bboxes:
            ...     print(f"{bbox['label']}: {bbox['bbox']}")
            dog: [100, 150, 400, 600]
            cat: [500, 200, 750, 650]
        """
        if not categories:
            return []

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Construct grounding prompt
        cat_str = ", ".join(categories)
        prompt = (
            f"Locate all instances of [{cat_str}] visible in the image. "
            "Return JSON array: [{\"label\": \"object_name\", \"bbox\": [xmin, ymin, xmax, ymax]}]. "
            "Bbox coordinates should be normalized to 1000 scale."
        )

        # Generate grounding result
        try:
            response = self.client.generate(
                system_prompt="You are a precise object localization assistant.",
                user_text=prompt,
                image=image
            )
        except Exception as e:
            print(f"[ERROR] Visual grounding failed: {e}")
            return []

        # Extract JSON
        results = extract_json_list(response)

        # Validate and filter results
        valid_results = []
        for item in results:
            if not isinstance(item, dict):
                continue

            label = item.get("label")
            bbox = item.get("bbox")

            if not label or not bbox or len(bbox) != 4:
                continue

            # Ensure bbox values are valid
            try:
                bbox = [int(float(x)) for x in bbox]
                x1, y1, x2, y2 = bbox

                # Validate bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                if any(c < 0 or c > 1000 for c in bbox):
                    continue

                valid_results.append({"label": label, "bbox": bbox})
            except (ValueError, TypeError):
                continue

        return valid_results

    def ground_from_intent(
        self,
        image: Union[str, Path, Image.Image],
        intent: Dict
    ) -> List[Dict]:
        """Ground objects based on parsed intent.

        Extracts target objects from intent structure and performs grounding.

        Args:
            image: Reference image.
            intent: Intent dictionary from Stage 1.

        Returns:
            List of grounded bounding boxes.

        Example:
            >>> intent = {
            ...     "edit_type": "subject_only",
            ...     "operations": [{"object": "car", ...}]
            ... }
            >>> bboxes = grounding.ground_from_intent("photo.jpg", intent)
        """
        # Extract object names from operations
        categories = []
        for op in intent.get("operations", []):
            obj_name = op.get("object")
            if obj_name and obj_name != "unknown":
                categories.append(obj_name)

        # Remove duplicates while preserving order
        categories = list(dict.fromkeys(categories))

        if not categories:
            return []

        return self.ground(image, categories)


class MockVisualGrounding:
    """Mock visual grounding for testing without real models.

    Returns dummy bounding boxes based on deterministic hashing.

    Example:
        >>> grounding = MockVisualGrounding()
        >>> bboxes = grounding.ground("test.jpg", ["car"])
        >>> print(bboxes)
        [{"label": "car", "bbox": [100, 100, 500, 500]}]
    """

    def ground(
        self,
        image: Union[str, Path, Image.Image],
        categories: List[str]
    ) -> List[Dict]:
        """Generate mock bounding boxes."""
        results = []
        for i, cat in enumerate(categories):
            # Generate deterministic bbox based on category name
            h = hash(cat)
            x1 = (abs(h) % 300) + 50
            y1 = (abs(h >> 8) % 300) + 50
            x2 = x1 + 300
            y2 = y1 + 400

            results.append({
                "label": cat,
                "bbox": [x1, y1, x2, y2]
            })

        return results

    def ground_from_intent(
        self,
        image: Union[str, Path, Image.Image],
        intent: Dict
    ) -> List[Dict]:
        """Ground from intent (mock)."""
        categories = []
        for op in intent.get("operations", []):
            obj_name = op.get("object")
            if obj_name and obj_name != "unknown":
                categories.append(obj_name)

        categories = list(dict.fromkeys(categories))
        return self.ground(image, categories)
