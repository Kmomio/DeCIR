"""CLIP encoder for image and text embedding in DeCIR.

This module provides a unified interface for CLIP-based encoding with support
for multiple models and backends.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
from PIL import Image


@dataclass
class CLIPConfig:
    """Configuration for CLIP encoder.

    Attributes:
        model_name: HuggingFace model name or path.
        device: Device for inference ("cuda", "cpu", or "auto").
        embedding_dim: Expected embedding dimension (auto-detected if None).
        dtype: Data type for inference (default: float32).
    """
    model_name: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    device: str = "cuda"
    embedding_dim: Optional[int] = None
    dtype: str = "float32"


class CLIPEncoder:
    """CLIP encoder for multi-modal embeddings.

    This encoder supports both image and text encoding using CLIP models
    from HuggingFace or OpenAI. All embeddings are L2-normalized by default.

    Attributes:
        config: CLIP configuration.
        model: Loaded CLIP model.
        processor: CLIP processor for inputs.

    Example:
        >>> config = CLIPConfig(
        ...     model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        ...     device="cuda"
        ... )
        >>> encoder = CLIPEncoder(config)
        >>> img_emb = encoder.encode_image("image.jpg")
        >>> txt_emb = encoder.encode_text("a photo of a cat")
        >>> similarity = np.dot(img_emb, txt_emb)
    """

    def __init__(self, config: CLIPConfig):
        """Initialize CLIP encoder.

        Args:
            config: CLIP configuration object.
        """
        self.config = config
        self._model = None
        self._processor = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize CLIP model and processor."""
        import torch
        from transformers import CLIPModel, CLIPProcessor

        # Load processor
        self._processor = CLIPProcessor.from_pretrained(self.config.model_name)

        # Load model
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = CLIPModel.from_pretrained(self.config.model_name).to(device)
        self._model.eval()

        # Auto-detect embedding dimension if not specified
        if self.config.embedding_dim is None:
            self.config.embedding_dim = self._model.config.projection_dim

    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """Encode text into CLIP embedding.

        Args:
            text: Input text or list of texts.
            normalize: Whether to L2-normalize embeddings (default: True).

        Returns:
            Embedding array of shape (embedding_dim,) for single text,
            or (batch_size, embedding_dim) for batch.

        Example:
            >>> emb = encoder.encode_text("a red car")
            >>> emb.shape
            (768,)
            >>> batch_emb = encoder.encode_text(["cat", "dog", "bird"])
            >>> batch_emb.shape
            (3, 768)
        """
        import torch

        # Ensure text is a list
        is_single = isinstance(text, str)
        text_list = [text] if is_single else text

        # Tokenize
        inputs = self._processor(
            text=text_list,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self._model.device)

        # Encode
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)

            # Normalize
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        embeddings = text_features.detach().cpu().float().numpy()

        # Return single embedding if single input
        return embeddings[0] if is_single else embeddings

    def encode_image(
        self,
        image: Union[str, Image.Image, np.ndarray, List],
        normalize: bool = True
    ) -> np.ndarray:
        """Encode image into CLIP embedding.

        Args:
            image: Input image (path, PIL Image, numpy array) or list of images.
            normalize: Whether to L2-normalize embeddings (default: True).

        Returns:
            Embedding array of shape (embedding_dim,) for single image,
            or (batch_size, embedding_dim) for batch.

        Example:
            >>> emb = encoder.encode_image("photo.jpg")
            >>> emb.shape
            (768,)
            >>> batch_emb = encoder.encode_image(["img1.jpg", "img2.jpg"])
            >>> batch_emb.shape
            (2, 768)
        """
        import torch

        # Convert to list of PIL Images
        is_single = not isinstance(image, list)
        image_list = [image] if is_single else image
        pil_images = [self._to_pil_image(img) for img in image_list]

        # Process images
        inputs = self._processor(
            images=pil_images,
            return_tensors="pt"
        ).to(self._model.device)

        # Encode
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

            # Normalize
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        embeddings = image_features.detach().cpu().float().numpy()

        # Return single embedding if single input
        return embeddings[0] if is_single else embeddings

    def compute_similarity(
        self,
        image_emb: np.ndarray,
        text_emb: np.ndarray
    ) -> float:
        """Compute cosine similarity between image and text embeddings.

        Args:
            image_emb: Image embedding (must be normalized).
            text_emb: Text embedding (must be normalized).

        Returns:
            Cosine similarity score in [-1, 1].

        Example:
            >>> img_emb = encoder.encode_image("cat.jpg")
            >>> txt_emb = encoder.encode_text("a cat")
            >>> sim = encoder.compute_similarity(img_emb, txt_emb)
            >>> print(f"Similarity: {sim:.4f}")
        """
        return float(np.dot(image_emb, text_emb))

    @staticmethod
    def _to_pil_image(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Convert various image formats to PIL Image.

        Args:
            image: Input image (path, PIL Image, or numpy array).

        Returns:
            PIL Image in RGB mode.

        Raises:
            TypeError: If image format is not supported.
        """
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")


class MockCLIPEncoder:
    """Mock CLIP encoder for testing without loading real models.

    This encoder generates deterministic pseudo-embeddings based on input hashing,
    useful for testing pipelines without GPU resources.

    Example:
        >>> encoder = MockCLIPEncoder(embedding_dim=512)
        >>> emb = encoder.encode_text("test")
        >>> emb.shape
        (512,)
    """

    def __init__(self, embedding_dim: int = 768):
        """Initialize mock encoder.

        Args:
            embedding_dim: Embedding dimension (default: 768).
        """
        self.embedding_dim = embedding_dim

    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Generate mock text embedding."""
        is_single = isinstance(text, str)
        text_list = [text] if is_single else text

        embeddings = []
        for t in text_list:
            v = np.zeros(self.embedding_dim, dtype=np.float32)
            h = abs(hash(t)) % self.embedding_dim
            v[h] = 1.0
            if normalize:
                v = v / np.linalg.norm(v)
            embeddings.append(v)

        embeddings = np.array(embeddings)
        return embeddings[0] if is_single else embeddings

    def encode_image(
        self,
        image: Union[str, Image.Image, np.ndarray, List],
        normalize: bool = True
    ) -> np.ndarray:
        """Generate mock image embedding."""
        is_single = not isinstance(image, list)
        image_list = [image] if is_single else image

        embeddings = []
        for img in image_list:
            # Get image dimensions for deterministic hashing
            if isinstance(img, str):
                img = Image.open(img)
            if isinstance(img, Image.Image):
                w, h = img.size
            else:
                arr = np.asarray(img)
                h, w = arr.shape[:2]

            v = np.zeros(self.embedding_dim, dtype=np.float32)
            idx = (w * 131 + h * 17) % self.embedding_dim
            v[idx] = 1.0
            if normalize:
                v = v / np.linalg.norm(v)
            embeddings.append(v)

        embeddings = np.array(embeddings)
        return embeddings[0] if is_single else embeddings

    def compute_similarity(self, image_emb: np.ndarray, text_emb: np.ndarray) -> float:
        """Compute similarity between embeddings."""
        return float(np.dot(image_emb, text_emb))
