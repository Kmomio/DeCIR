"""Stage modules for the DeCIR pipeline."""

from decir.core.stages.intent_parser import IntentParser
from decir.core.stages.visual_grounding import VisualGrounding
from decir.core.stages.region_caption import RegionCaption
from decir.core.stages.target_rewrite import TargetRewrite
from decir.core.stages.mask_engine import MaskEngine, MaskStrategy
from decir.core.stages.image_edit import ImageEditor
from decir.core.stages.query_builder import QueryBuilder

__all__ = [
    "IntentParser",
    "VisualGrounding",
    "RegionCaption",
    "TargetRewrite",
    "MaskEngine",
    "MaskStrategy",
    "ImageEditor",
    "QueryBuilder",
]
