"""
Extraction module for converting Minecraft .mca files to .npz training data.

This module provides functionality to parse Minecraft region files and convert
them into compressed numpy arrays suitable for machine learning training.
"""

from .chunk_extractor import ChunkExtractor

__all__ = ["ChunkExtractor"]
