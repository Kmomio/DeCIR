"""Stage 2.5: Target Rewrite for DeCIR.

Synthesizes target description by combining current region description
with edit intent.
"""

from typing import Dict, Optional

from decir.models.qwen_client import Qwen3VLClient


class TargetRewrite:
    """Stage 2.5: Target Rewrite.

    Combines current region description with modification intent to generate
    target description for image editing.

    Example:
        >>> rewriter = TargetRewrite(qwen_client)
        >>> target_desc = rewriter.rewrite(
        ...     current_description="a red car",
        ...     modification_text="change color to blue",
        ...     intent=intent_dict
        ... )
        >>> print(target_desc)
        "a blue car"
    """

    def __init__(self, qwen_client: Optional[Qwen3VLClient] = None, mode: str = "simple"):
        """Initialize Target Rewrite stage.

        Args:
            qwen_client: Qwen3-VL client for LLM-based rewriting (optional).
            mode: Rewrite mode ("simple" or "llm").
        """
        self.client = qwen_client
        self.mode = mode

    def rewrite(
        self,
        current_description: str,
        modification_text: str,
        intent: Optional[Dict] = None
    ) -> str:
        """Generate target description.

        Args:
            current_description: Description of current region state.
            modification_text: User's modification request.
            intent: Parsed intent dictionary (optional, for advanced logic).

        Returns:
            Target description for image editing.

        Example:
            >>> target = rewriter.rewrite(
            ...     "a person wearing a red shirt",
            ...     "change shirt color to blue"
            ... )
            >>> print(target)
            "a person wearing a blue shirt"
        """
        if self.mode == "simple":
            # Simple concatenation
            return f"{current_description}, {modification_text}"

        elif self.mode == "llm" and self.client:
            # LLM-based rewriting
            prompt = (
                f"Current: {current_description}\n"
                f"Modification: {modification_text}\n\n"
                "Generate the target description after applying the modification. "
                "Output only the target description, nothing else."
            )

            try:
                response = self.client.generate(
                    system_prompt="You are a text rewriting assistant.",
                    user_text=prompt,
                    image=None  # Text-only task
                )
                return response.strip()
            except Exception as e:
                print(f"[ERROR] LLM rewrite failed: {e}, falling back to simple mode")
                return f"{current_description}, {modification_text}"

        else:
            # Fallback
            return modification_text


class MockTargetRewrite:
    """Mock target rewriter for testing."""

    def rewrite(
        self,
        current_description: str,
        modification_text: str,
        intent: Optional[Dict] = None
    ) -> str:
        """Generate mock target description."""
        return f"mock_target: {modification_text}"
