"""Compatibility shim for legacy imports.

Historically this module was called ``deploy_models.py``. It has since been
renamed to ``deploy.py`` to provide a consistent CLI entrypoint.

This module re-exports the public API of ``VoxelTree.scripts.deploy`` so that
existing imports continue to work.
"""

from __future__ import annotations

from VoxelTree.scripts.deploy import *  # noqa: F401,F403
