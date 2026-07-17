"""
DeCIR: Dual-modal Semantic Decoupling for Composed Image Retrieval.

This package provides a training-free framework for composed image retrieval
using dual-modal semantic decoupling.
"""

__version__ = "0.1.0"
__author__ = "DeCIR Authors"

from decir.core.pipeline import DeCIRPipeline
from decir.core.stages.query_builder import QueryBuilder

__all__ = [
    "DeCIRPipeline",
    "QueryBuilder",
]
