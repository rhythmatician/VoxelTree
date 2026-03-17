"""server_config.py — Load and validate servers.yaml role configurations.

``servers.yaml`` lives next to the ``profiles/`` directory in the VoxelTree repo
root.  It defines one entry per server role (e.g. ``"train"``, ``"validate"``),
each specifying the Minecraft world seed, level-name, and network ports.

Typical ``servers.yaml``::

    train:
      seed: 8675309
      level_name: train
      server_port: 25565
      rcon_port: 25575
      rcon_password: voxeltree

    validate:
      seed: 3141592
      level_name: validate
      server_port: 25566
      rcon_port: 25576
      rcon_password: voxeltree

The profile YAML then references the desired role via ``server_role: train``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_ROLE = "train"


def _find_servers_yaml() -> Path:
    """Search upward from this module for servers.yaml."""
    cur = Path(__file__).resolve()
    for _ in range(6):
        candidate = cur.parent / "servers.yaml"
        if candidate.exists():
            return candidate
        cur = cur.parent
    # Fallback: assume repo root is three levels up from gui/
    return Path(__file__).resolve().parents[3] / "servers.yaml"


@dataclass(frozen=True)
class ServerRole:
    """Configuration for one named server role.

    Attributes
    ----------
    name          : Role identifier, e.g. ``"train"`` or ``"validate"``.
    seed          : Minecraft world seed (int or string).
    level_name    : Minecraft ``level-name`` property (world folder name).
    server_port   : Java edition listen port shown to players / clients.
    rcon_port     : RCON TCP port for command injection.
    rcon_password : RCON authentication password.
    """

    name: str
    seed: int | str
    level_name: str
    server_port: int = 25565
    rcon_port: int = 25575
    rcon_password: str = "voxeltree"

    @property
    def voxy_save_key(self) -> str:
        """Voxy stores data under ``<host>_<port>``; this is the subfolder key."""
        return f"localhost_{self.server_port}"


_BUILTIN_DEFAULTS: dict[str, ServerRole] = {
    "train": ServerRole(
        name="train",
        seed=8675309,
        level_name="train",
        server_port=25565,
        rcon_port=25575,
        rcon_password="voxeltree",
    )
}


def load_server_roles() -> dict[str, ServerRole]:
    """Load and parse ``servers.yaml`` into a ``{role_name: ServerRole}`` dict.

    Falls back to a single built-in ``"train"`` role if the file does not exist.
    """
    path = _find_servers_yaml()
    if not path.exists():
        return dict(_BUILTIN_DEFAULTS)

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    roles: dict[str, ServerRole] = {}
    for role_name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        roles[role_name] = ServerRole(
            name=role_name,
            seed=cfg.get("seed", 0),
            level_name=str(cfg.get("level_name", role_name)),
            server_port=int(cfg.get("server_port", 25565)),
            rcon_port=int(cfg.get("rcon_port", 25575)),
            rcon_password=str(cfg.get("rcon_password", "voxeltree")),
        )
    return roles


def get_role(role_name: str | None = None) -> ServerRole:
    """Return the :class:`ServerRole` for *role_name*, defaulting to ``"train"``.

    Falls back to the first available role if *role_name* is not found.
    """
    roles = load_server_roles()
    name = role_name or _DEFAULT_ROLE
    if name in roles:
        return roles[name]
    if roles:
        return next(iter(roles.values()))
    return _BUILTIN_DEFAULTS["train"]


def list_role_names() -> list[str]:
    """Return a sorted list of available role names from ``servers.yaml``."""
    return sorted(load_server_roles().keys())
