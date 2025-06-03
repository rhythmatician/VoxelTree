"""
LODValidator - LOD alignment validation utilities

This module provides validation utilities for LOD (Level of Detail) alignment
between parent and child voxel pairs.
"""

import logging

# Re-export LODValidator from patch_pairer for backward compatibility
from .patch_pairer import LODValidator

logger = logging.getLogger(__name__)

__all__ = ["LODValidator"]
