"""
LODValidator - LOD alignment validation utilities

This module provides validation utilities for LOD (Level of Detail) alignment
between parent and child voxel pairs.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Re-export LODValidator from patch_pairer for backward compatibility
from .patch_pairer import LODValidator

__all__ = ["LODValidator"]
