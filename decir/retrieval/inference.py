"""Retrieval inference module for DeCIR."""

from typing import List, Dict
import numpy as np
from tqdm import tqdm

from decir.core.pipeline import DeCIRPipeline
from decir.models.clip_encoder import CLIPEncoder


class RetrievalInference:
    """Composed Image Retrieval inference.

    Performs retrieval by computing query embeddings for reference-caption pairs
    and comparing against a gallery of target images.

    Example:
        >>> pipeline = DeCIRPipeline(qwen_client, clip_encoder, sdxl_client)
        >>> inference = RetrievalInference(pipeline, clip_encoder)
        >>>
        >>> # Encode gallery
        >>> gallery_embs = inference.encode_gallery(gallery_images)
        >>>
        >>> # Retrieve for queries
        >>> results = inference.retrieve(
        ...     queries=[{"reference": "img1.jpg", "caption": "make it blue"}],
        ...     gallery_embeddings=gallery_embs,
        ...     top_k=50
        ... )
    """

    def __init__(self, pipeline: DeCIRPipeline, clip_encoder: CLIPEncoder):
        """Initialize retrieval inference.

        Args:
            pipeline: DeCIR pipeline for query encoding.
            clip_encoder: CLIP encoder for gallery encoding.
        """
        self.pipeline = pipeline
        self.clip_encoder = clip_encoder

    def encode_gallery(self, gallery_images: List[str]) -> np.ndarray:
        """Encode gallery images.

        Args:
            gallery_images: List of image paths.

        Returns:
            Gallery embeddings array of shape (N, D).
        """
        print(f"Encoding {len(gallery_images)} gallery images...")
        embeddings = []

        for img_path in tqdm(gallery_images):
            emb = self.clip_encoder.encode_image(img_path)
            embeddings.append(emb)

        return np.array(embeddings)

    def retrieve(
        self,
        queries: List[Dict],
        gallery_embeddings: np.ndarray,
        gallery_ids: List[str],
        top_k: int = 50
    ) -> List[Dict]:
        """Perform retrieval for queries.

        Args:
            queries: List of query dicts with 'reference' and 'caption' keys.
            gallery_embeddings: Pre-computed gallery embeddings (N, D).
            gallery_ids: List of gallery image IDs.
            top_k: Number of top results to return.

        Returns:
            List of result dicts with 'query_id', 'reference', 'caption', 'results'.
        """
        results = []

        print(f"Processing {len(queries)} queries...")
        for query in tqdm(queries):
            # Encode query
            query_emb = self.pipeline(
                reference_image=query['reference'],
                modification_text=query['caption']
            )

            # Compute similarities
            similarities = np.dot(gallery_embeddings, query_emb)

            # Get top-k
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            top_k_ids = [gallery_ids[i] for i in top_k_indices]
            top_k_scores = [float(similarities[i]) for i in top_k_indices]

            results.append({
                'query_id': query.get('pair_id', ''),
                'reference': query['reference'],
                'caption': query['caption'],
                'results': top_k_ids,
                'scores': top_k_scores
            })

        return results
