"""Stage 9: Query Builder for DeCIR.

This module implements the core innovation of DeCIR: Dual-modal Semantic Fusion.
It constructs the final retrieval query by combining:
1. Global visual features from reference image
2. Textual semantics from modification description
3. Visual delta from localized edit region

The query embedding is: E_query = α·E_img + β·E_txt + γ·E_delta
"""

from typing import Optional, Tuple, List, Union
from pathlib import Path
import numpy as np
from PIL import Image

from decir.models.clip_encoder import CLIPEncoder
from decir.utils.patch_ops import extract_visual_delta_patch


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector.

    Args:
        vector: Input vector.

    Returns:
        L2-normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


class QueryBuilder:
    """Stage 9: Query Builder.

    Constructs final retrieval query using dual-modal semantic fusion.

    The query combines three components:
    - **E_image**: CLIP encoding of the reference image (global context)
    - **E_text**: CLIP encoding of the modification text (semantic intent)
    - **E_delta**: Visual semantic delta from localized edit region

    **Dual-modal Fusion Formula:**
    ```
    E_query = α·E_image + β·E_text + γ·E_delta
    ```

    Where:
    - α (alpha): Weight for reference image features
    - β (beta): Weight for modification text features
    - γ (gamma): Weight for visual delta features

    **Delta Modes:**
    - "diff": E_delta = E(edit_patch) - E(ref_patch)  [Recommended]
    - "patch": E_delta = E(edit_patch)
    - "none": E_delta = 0 (no visual delta)

    Attributes:
        clip_encoder: CLIP encoder for embeddings.
        alpha: Weight for image features (default: 0.6).
        beta: Weight for text features (default: 0.2).
        gamma: Weight for delta features (default: 0.2).
        delta_mode: Delta computation mode.
        patch_expand_ratio: Expansion ratio for delta patch extraction.

    Example:
        >>> from decir.models.clip_encoder import CLIPConfig, CLIPEncoder
        >>> clip_config = CLIPConfig(model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        >>> clip_encoder = CLIPEncoder(clip_config)
        >>> builder = QueryBuilder(
        ...     clip_encoder=clip_encoder,
        ...     alpha=0.6, beta=0.2, gamma=0.2
        ... )
        >>> query_emb = builder.build(
        ...     reference_image="car.jpg",
        ...     modification_text="change color to blue",
        ...     edited_image=edited_car,
        ...     masks=[car_mask]
        ... )
    """

    def __init__(
        self,
        clip_encoder: CLIPEncoder,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        delta_mode: str = "diff",
        patch_expand_ratio: float = 0.15
    ):
        """Initialize Query Builder.

        Args:
            clip_encoder: CLIP encoder for computing embeddings.
            alpha: Weight for reference image (default: 0.6).
            beta: Weight for modification text (default: 0.2).
            gamma: Weight for visual delta (default: 0.2).
            delta_mode: Delta mode ("diff", "patch", or "none").
            patch_expand_ratio: Patch expansion ratio (default: 0.15).

        Note:
            Weights should sum to approximately 1.0 for balanced fusion.
            Adjust based on dataset characteristics:
            - Higher alpha: Preserve more reference image context
            - Higher beta: Focus more on text modification
            - Higher gamma: Emphasize visual changes more
        """
        self.clip_encoder = clip_encoder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_mode = delta_mode
        self.patch_expand_ratio = patch_expand_ratio

    def build(
        self,
        reference_image: Union[str, Path, Image.Image],
        modification_text: str,
        edited_image: Optional[Image.Image] = None,
        masks: Optional[List[np.ndarray]] = None,
        weights: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """Build retrieval query embedding.

        Args:
            reference_image: Reference image (path or PIL Image).
            modification_text: Modification description text.
            edited_image: Edited image from Stage 4 (optional, for delta).
            masks: Edit masks from Stage 3 (optional, for delta).
            weights: Override (alpha, beta, gamma) weights (optional).

        Returns:
            Query embedding as L2-normalized numpy array.

        Example:
            >>> # Without visual delta (text + image only)
            >>> query = builder.build(
            ...     reference_image="photo.jpg",
            ...     modification_text="add a hat"
            ... )
            >>>
            >>> # With full dual-modal fusion
            >>> query = builder.build(
            ...     reference_image="photo.jpg",
            ...     modification_text="change car color to blue",
            ...     edited_image=edited_photo,
            ...     masks=[car_mask]
            ... )
        """
        # Use provided weights or defaults
        if weights:
            alpha, beta, gamma = weights
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma

        # Load reference image if path
        if isinstance(reference_image, (str, Path)):
            reference_image = Image.open(reference_image).convert("RGB")

        # 1. Encode reference image (global visual context)
        E_image = self.clip_encoder.encode_image(reference_image)

        # 2. Encode modification text (semantic intent)
        E_text = self.clip_encoder.encode_text(modification_text)

        # 3. Compute visual delta (localized semantic change)
        E_delta = np.zeros_like(E_image, dtype=np.float32)

        if self.delta_mode != "none" and edited_image is not None and masks:
            E_delta = self._compute_visual_delta(
                reference_image=reference_image,
                edited_image=edited_image,
                masks=masks
            )

        # 4. Fuse components with weighted sum
        E_query = alpha * E_image + beta * E_text + gamma * E_delta

        # 5. L2-normalize for retrieval
        return l2_normalize(E_query.astype(np.float32))

    def _compute_visual_delta(
        self,
        reference_image: Image.Image,
        edited_image: Image.Image,
        masks: List[np.ndarray]
    ) -> np.ndarray:
        """Compute visual semantic delta.

        Extracts corresponding patches from reference and edited images,
        then computes their embedding difference.

        Args:
            reference_image: Reference image.
            edited_image: Edited image.
            masks: Edit masks.

        Returns:
            Visual delta embedding.
        """
        try:
            # Extract patches from edit region
            ref_patch, edit_patch, _bbox = extract_visual_delta_patch(
                ref_image=reference_image,
                edited_image=edited_image,
                masks=masks,
                expand_ratio=self.patch_expand_ratio
            )

            # Encode patches
            E_ref_patch = self.clip_encoder.encode_image(ref_patch)
            E_edit_patch = self.clip_encoder.encode_image(edit_patch)

            # Compute delta based on mode
            if self.delta_mode == "diff":
                # Difference mode: captures semantic change
                E_delta = (E_edit_patch - E_ref_patch).astype(np.float32)
            elif self.delta_mode == "patch":
                # Patch mode: uses only edited patch
                E_delta = E_edit_patch.astype(np.float32)
            else:
                raise ValueError(f"Unknown delta_mode: {self.delta_mode}")

            return E_delta

        except Exception as e:
            print(f"[WARNING] Visual delta computation failed: {e}")
            # Fallback to zero delta
            E_image = self.clip_encoder.encode_image(reference_image)
            return np.zeros_like(E_image, dtype=np.float32)


class MockQueryBuilder:
    """Mock query builder for testing."""

    def __init__(self, embedding_dim: int = 768):
        """Initialize mock builder."""
        self.embedding_dim = embedding_dim

    def build(
        self,
        reference_image: Union[str, Path, Image.Image],
        modification_text: str,
        edited_image: Optional[Image.Image] = None,
        masks: Optional[List[np.ndarray]] = None,
        weights: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """Generate mock query embedding."""
        # Deterministic pseudo-embedding based on text
        v = np.zeros(self.embedding_dim, dtype=np.float32)
        h = abs(hash(modification_text)) % self.embedding_dim
        v[h] = 1.0
        return l2_normalize(v)
