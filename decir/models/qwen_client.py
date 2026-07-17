"""Qwen3-VL model client for DeCIR.

This module provides a wrapper around the Qwen2-VL model for multi-modal
understanding tasks including intent parsing, visual grounding, and captioning.
"""

import torch
import gc
from typing import Union, Optional
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class Qwen3VLClient:
    """Client for Qwen3-VL multi-modal model inference.

    This client provides optimized inference for NVIDIA GPUs with automatic
    OOM (Out-of-Memory) handling. When OOM is detected, the client automatically
    retries with downscaled images.

    Attributes:
        model: Loaded Qwen3-VL model.
        processor: Qwen3-VL processor for inputs.
        max_new_tokens: Maximum tokens to generate.
        do_sample: Whether to use sampling (vs greedy decoding).

    Example:
        >>> client = Qwen3VLClient(
        ...     model_path="Qwen/Qwen2-VL-7B-Instruct",
        ...     device="cuda"
        ... )
        >>> response = client.generate(
        ...     system_prompt="You are a helpful vision assistant.",
        ...     user_text="Describe this image.",
        ...     image="path/to/image.jpg"
        ... )
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        dtype: torch.dtype = torch.float16
    ):
        """Initialize Qwen3-VL client.

        Args:
            model_path: HuggingFace model path or local checkpoint directory.
            device: Device placement ("auto", "cuda", "cpu").
            max_new_tokens: Maximum tokens to generate (default: 256).
            do_sample: Whether to use sampling for generation (default: False).
            dtype: Model data type (default: float16 for GPU efficiency).
        """
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        # Map device to device_map format
        device_map = device
        if device in ("cuda", "cuda:0", "0"):
            device_map = "auto"

        # Load model with float16 precision for memory efficiency
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

    def _prepare_inputs(
        self,
        system_prompt: str,
        user_text: str,
        image: Union[str, Image.Image],
        target_size: Optional[int] = None
    ) -> dict:
        """Prepare inputs for model inference.

        Args:
            system_prompt: System prompt text.
            user_text: User query text.
            image: Input image (path or PIL Image).
            target_size: Optional target size for image resizing (OOM handling).

        Returns:
            Dictionary of model inputs.
        """
        # Load and convert image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected str or PIL.Image, got {type(image)}")

        # Resize image if target size specified (OOM mitigation)
        if target_size:
            image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        # Construct messages in chat format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt.strip()}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text}
                ]
            }
        ]

        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Ensure float16 precision for all float tensors
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                inputs[k] = v.to(torch.float16)

        return inputs

    @torch.no_grad()
    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image: Union[str, Image.Image],
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate text response from multi-modal inputs.

        This method automatically handles CUDA OOM errors by retrying with
        downscaled images (640x640).

        Args:
            system_prompt: System prompt defining assistant behavior.
            user_text: User query or instruction.
            image: Input image (file path or PIL Image).
            max_new_tokens: Override default max_new_tokens.

        Returns:
            Generated text response.

        Raises:
            torch.cuda.OutOfMemoryError: If OOM occurs even with downscaled image.
            RuntimeError: If inference fails for other reasons.

        Example:
            >>> response = client.generate(
            ...     system_prompt="You are a vision assistant.",
            ...     user_text="What objects are in this image?",
            ...     image="scene.jpg"
            ... )
            >>> print(response)
            "The image contains a person, a car, and a tree."
        """
        try:
            # First attempt with original image
            return self._run_inference(
                system_prompt, user_text, image,
                target_size=None,
                max_new_tokens=max_new_tokens
            )

        except torch.cuda.OutOfMemoryError:
            print("[WARNING] CUDA OOM detected. Retrying with resized image (640x640)...")
            self._clear_gpu_cache()

            try:
                # Retry with downscaled image
                return self._run_inference(
                    system_prompt, user_text, image,
                    target_size=640,
                    max_new_tokens=max_new_tokens
                )
            except torch.cuda.OutOfMemoryError:
                self._clear_gpu_cache()
                raise  # Let caller handle if even 640x640 fails

    def _run_inference(
        self,
        system_prompt: str,
        user_text: str,
        image: Union[str, Image.Image],
        target_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """Internal inference method.

        Args:
            system_prompt: System prompt text.
            user_text: User query text.
            image: Input image.
            target_size: Optional image resize target.
            max_new_tokens: Override generation length.

        Returns:
            Generated text (response only, without input prompt).
        """
        inputs = self._prepare_inputs(system_prompt, user_text, image, target_size)

        mnt = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        # Generate tokens
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=mnt,
            do_sample=self.do_sample
        )

        # Trim input tokens to get only generated response
        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

        # Decode to text
        return self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    def _clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
