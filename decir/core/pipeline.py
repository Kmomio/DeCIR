"""DeCIR Pipeline - End-to-end composed image retrieval.

This module provides the main pipeline that integrates all stages of DeCIR.
"""

from typing import Optional, Dict, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image

from decir.models.qwen_client import Qwen3VLClient
from decir.models.clip_encoder import CLIPEncoder
from decir.models.sdxl_client import SDXLInpaintClient
from decir.models.flux_client import FluxInpaintClient

from decir.core.stages import (
    IntentParser,
    VisualGrounding,
    RegionCaption,
    TargetRewrite,
    MaskEngine,
    ImageEditor,
    QueryBuilder
)


class DeCIRPipeline:
    """DeCIR end-to-end pipeline.

    This pipeline executes all stages of DeCIR to produce a retrieval query
    embedding from a reference image and modification text.

    **Pipeline Stages:**
    1. Intent Parser - Analyze modification intent
    2. Visual Grounding - Localize target objects
    3. Region Caption - Describe localized regions
    4. Target Rewrite - Generate target descriptions
    5. Mask Engine - Generate edit masks
    6. Image Editor - Apply modifications via inpainting
    7. Query Builder - Construct retrieval query

    Attributes:
        intent_parser: Stage 1 parser.
        visual_grounding: Stage 2A grounding.
        region_caption: Stage 2B captioner.
        target_rewrite: Stage 2.5 rewriter.
        mask_engine: Stage 3 mask generator.
        image_editor: Stage 4 editor.
        query_builder: Stage 9 query builder.

    Example:
        >>> from decir.models import Qwen3VLClient, CLIPEncoder, FluxInpaintClient
        >>> # Initialize models
        >>> qwen_client = Qwen3VLClient(model_path="Qwen/Qwen2-VL-7B-Instruct")
        >>> clip_encoder = CLIPEncoder(CLIPConfig(model_name="laion/CLIP-ViT-L-14"))
        >>> flux_client = FluxInpaintClient(FluxConfig())  # FLUX.1-dev (recommended)
        >>>
        >>> # Create pipeline
        >>> pipeline = DeCIRPipeline(
        ...     qwen_client=qwen_client,
        ...     clip_encoder=clip_encoder,
        ...     inpaint_client=flux_client
        ... )
        >>>
        >>> # Run inference
        >>> query_emb, intermediate = pipeline(
        ...     reference_image="car.jpg",
        ...     modification_text="change the color to blue"
        ... )
    """

    def __init__(
        self,
        qwen_client: Qwen3VLClient,
        clip_encoder: CLIPEncoder,
        inpaint_client: Union[FluxInpaintClient, SDXLInpaintClient],
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        delta_mode: str = "diff"
    ):
        """Initialize DeCIR pipeline.

        Args:
            qwen_client: Qwen3-VL client for multi-modal understanding.
            clip_encoder: CLIP encoder for embeddings.
            inpaint_client: Inpainting client (FluxInpaintClient recommended, or SDXLInpaintClient).
            alpha: Weight for image features in query fusion.
            beta: Weight for text features in query fusion.
            gamma: Weight for delta features in query fusion.
            delta_mode: Delta computation mode ("diff", "patch", "none").

        Note:
            FLUX.1-dev is recommended over SDXL-Inpaint for superior quality as shown in the paper.
        """
        # Initialize all stages
        self.intent_parser = IntentParser(qwen_client)
        self.visual_grounding = VisualGrounding(qwen_client)
        self.region_caption = RegionCaption(qwen_client)
        self.target_rewrite = TargetRewrite(qwen_client, mode="simple")
        self.mask_engine = MaskEngine()
        self.image_editor = ImageEditor(inpaint_client)
        self.query_builder = QueryBuilder(
            clip_encoder=clip_encoder,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta_mode=delta_mode
        )

    def __call__(
        self,
        reference_image: Union[str, Path, Image.Image],
        modification_text: str,
        return_intermediate: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Run full DeCIR pipeline.

        Args:
            reference_image: Reference image (path or PIL Image).
            modification_text: Text description of desired modification.
            return_intermediate: Whether to return intermediate stage outputs.

        Returns:
            If return_intermediate=False: query_embedding (np.ndarray)
            If return_intermediate=True: (query_embedding, intermediate_dict)

        Example:
            >>> query_emb = pipeline("photo.jpg", "add sunglasses to the person")
            >>> # Or get intermediate results
            >>> query_emb, intermediate = pipeline(
            ...     "photo.jpg",
            ...     "add sunglasses",
            ...     return_intermediate=True
            ... )
            >>> print(intermediate.keys())
            dict_keys(['intent', 'bboxes', 'masks', 'edited_image'])
        """
        intermediate = {}

        # Load image
        if isinstance(reference_image, (str, Path)):
            reference_image = Image.open(reference_image).convert("RGB")

        # Stage 1: Parse intent
        intent = self.intent_parser.parse(
            reference_image=reference_image,
            modification_text=modification_text
        )
        intermediate['intent'] = intent

        # Stage 2A: Ground objects
        bboxes = self.visual_grounding.ground_from_intent(
            image=reference_image,
            intent=intent
        )
        intermediate['bboxes'] = bboxes

        # Stage 2B: Caption regions (optional, for target rewrite)
        if bboxes:
            region_desc = self.region_caption.caption(
                image=reference_image,
                bbox=bboxes[0]['bbox']  # Use first bbox
            )
        else:
            region_desc = "image"

        # Stage 2.5: Rewrite target
        target_prompt = self.target_rewrite.rewrite(
            current_description=region_desc,
            modification_text=modification_text,
            intent=intent
        )
        intermediate['target_prompt'] = target_prompt

        # Stage 3: Generate masks
        masks = self.mask_engine.generate_masks(
            image=reference_image,
            bboxes=bboxes,
            intent=intent
        )
        intermediate['masks'] = masks

        # Stage 4: Edit image
        edited_image = self.image_editor.edit(
            image=reference_image,
            masks=masks,
            target_prompt=target_prompt
        )
        intermediate['edited_image'] = edited_image

        # Stage 9: Build query
        query_embedding = self.query_builder.build(
            reference_image=reference_image,
            modification_text=modification_text,
            edited_image=edited_image,
            masks=masks
        )

        if return_intermediate:
            return query_embedding, intermediate
        else:
            return query_embedding
