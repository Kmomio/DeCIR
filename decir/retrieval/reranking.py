"""MLLM-based reranking for DeCIR."""

from typing import List, Dict
from decir.models.qwen_client import Qwen3VLClient


class MLLMReranker:
    """MLLM-based reranker for refining initial retrieval results.

    Uses multi-modal LLM to perform joint visual-semantic reasoning
    across top-k candidates.

    Example:
        >>> reranker = MLLMReranker(qwen_client)
        >>> reranked = reranker.rerank(
        ...     reference_image="car.jpg",
        ...     modification_text="make it blue",
        ...     candidates=["result1.jpg", "result2.jpg", ...],
        ...     top_k=10
        ... )
    """

    def __init__(self, qwen_client: Qwen3VLClient):
        """Initialize reranker.

        Args:
            qwen_client: Qwen3-VL client for reranking.
        """
        self.client = qwen_client

    def rerank(
        self,
        reference_image: str,
        modification_text: str,
        candidates: List[str],
        top_k: int = 10
    ) -> List[str]:
        """Rerank candidates using MLLM.

        Args:
            reference_image: Reference image path.
            modification_text: Modification description.
            candidates: List of candidate image paths.
            top_k: Number of top results to return.

        Returns:
            Reranked list of candidate image paths.

        Note:
            This is a simplified implementation. Full implementation would
            use grid-based visual prompting as described in the paper.
        """
        # For now, return candidates as-is
        # Full implementation would construct image grid and use Qwen3-VL
        print(f"[NOTE] MLLM reranking not fully implemented. Returning top-{top_k} as-is.")
        return candidates[:top_k]
