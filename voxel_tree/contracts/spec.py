"""Model I/O contract data structures with revision tracking.

A ``ModelContract`` is an immutable description of a model's inputs, outputs,
and the pipeline stages that produce / consume it.  When the I/O changes, you
create a **new revision** of the same model name rather than inventing a new
model name (``density_v3``, ``density_v4`` …).

Immutability contract
---------------------
``ModelContract`` is a frozen dataclass.  Once registered, a contract can
never be mutated in-place — only superseded by a higher revision.

Validation
----------
Call ``contract.validate_forward(inputs, outputs)`` to assert that live
tensors match the declared shapes — useful as a training-time sanity check
and as an export-time gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch


# ---------------------------------------------------------------------------
# TensorSpec — one named tensor in a contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """Describes one input or output tensor.

    Parameters
    ----------
    name     : ONNX-visible tensor name (e.g. ``'climate_input'``).
    shape    : Shape tuple.  Use strings for dynamic axes (``('batch', 6)``).
    dtype    : Numpy-style dtype string (``'float32'``, ``'int64'``).
    channels : Optional ordered list of semantic channel names.
    channel_indices : Optional list of indices into a source tensor
        (e.g. which RouterField indices this input reads).
    description : Free-text note for humans / sidecar JSON.
    """

    name: str
    shape: tuple[str | int, ...]
    dtype: str = "float32"
    channels: tuple[str, ...] | None = None
    channel_indices: tuple[int, ...] | None = None
    description: str = ""

    # -- validation helpers ------------------------------------------------

    def matches_tensor(self, t: torch.Tensor, *, batch_dim: int = 0) -> bool:
        """Return True if *t*'s shape is compatible with this spec.

        Dynamic dimensions (string entries in ``self.shape``) match any size.
        """
        if t.ndim != len(self.shape):
            return False
        for i, (declared, actual) in enumerate(zip(self.shape, t.shape)):
            if isinstance(declared, str):
                continue  # dynamic axis — anything goes
            if declared != actual:
                return False
        return True

    def shape_mismatch_msg(self, t: torch.Tensor) -> str | None:
        """Return a human-readable mismatch message, or None if OK."""
        if self.matches_tensor(t):
            return None
        return f"Tensor '{self.name}': expected shape {self.shape}, " f"got {tuple(t.shape)}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (for sidecar files)."""
        d: dict[str, Any] = {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }
        if self.channels is not None:
            d["channels"] = list(self.channels)
        if self.channel_indices is not None:
            d["channel_indices"] = list(self.channel_indices)
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TensorSpec":
        """Deserialize from a sidecar dict."""
        return cls(
            name=d["name"],
            shape=tuple(d["shape"]),
            dtype=d.get("dtype", "float32"),
            channels=tuple(d["channels"]) if d.get("channels") else None,
            channel_indices=tuple(d["channel_indices"]) if d.get("channel_indices") else None,
            description=d.get("description", ""),
        )


# ---------------------------------------------------------------------------
# ModelContract — full I/O contract for one model revision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelContract:
    """Immutable description of a model's I/O contract at a specific revision.

    Parameters
    ----------
    model_name : Stable model family name (e.g. ``'density'``, ``'sparse_octree'``).
                 This is the *semantic* name — it never changes when you bump
                 the I/O contract.
    revision   : Monotonically increasing integer.  Bump whenever any input
                 or output changes shape, dtype, name, or semantics.
    inputs     : Ordered list of input ``TensorSpec``s.
    outputs    : Ordered list of output ``TensorSpec``s.
    onnx_opset : ONNX opset this contract targets.
    contract_id : Dot-delimited contract string written into sidecars
                  (e.g. ``'lodiffusion.v7.density'``).
    description : Human-readable summary.
    changelog   : What changed from the previous revision.
    build_pairs_fn : Dotted import path of the ``build_pairs`` entry point
                     that produces compatible training data.
    train_fn       : Dotted import path of the ``train`` entry point.
    export_fn      : Dotted import path of the ``export`` entry point.
    extra          : Arbitrary extra metadata (class names, thresholds, etc.).
    """

    model_name: str
    revision: int
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    onnx_opset: int = 18
    contract_id: str = ""
    description: str = ""
    changelog: str = ""

    # Pipeline entry points (dotted import paths).
    build_pairs_fn: str = ""
    train_fn: str = ""
    export_fn: str = ""

    # Arbitrary extra metadata (not validated).
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-generate contract_id if not provided.
        if not self.contract_id:
            object.__setattr__(
                self,
                "contract_id",
                f"lodiffusion.r{self.revision}.{self.model_name}",
            )

    # -- fingerprint -------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 hash of the I/O shape contract.

        The fingerprint changes whenever any tensor name, shape, dtype, or
        channel list changes.  It does NOT include description/changelog —
        only the structural bits that matter for compatibility.
        """
        payload = {
            "model_name": self.model_name,
            "revision": self.revision,
            "inputs": [s.to_dict() for s in self.inputs],
            "outputs": [s.to_dict() for s in self.outputs],
            "onnx_opset": self.onnx_opset,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # -- validation --------------------------------------------------------

    def validate_forward(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
    ) -> None:
        """Assert live tensors match the declared specs.

        Raises ``ContractViolation`` on mismatch.
        """
        if len(inputs) != len(self.inputs):
            raise ContractViolation(
                f"[{self.contract_id}] Expected {len(self.inputs)} inputs, " f"got {len(inputs)}"
            )
        if len(outputs) != len(self.outputs):
            raise ContractViolation(
                f"[{self.contract_id}] Expected {len(self.outputs)} outputs, " f"got {len(outputs)}"
            )
        errors: list[str] = []
        for spec, t in zip(self.inputs, inputs):
            msg = spec.shape_mismatch_msg(t)
            if msg:
                errors.append(f"  INPUT  {msg}")
        for spec, t in zip(self.outputs, outputs):
            msg = spec.shape_mismatch_msg(t)
            if msg:
                errors.append(f"  OUTPUT {msg}")
        if errors:
            raise ContractViolation(f"[{self.contract_id}] Shape mismatches:\n" + "\n".join(errors))

    # -- serialization -----------------------------------------------------

    def to_sidecar(self, **training_meta: Any) -> dict[str, Any]:
        """Build the sidecar JSON dict for export scripts.

        This replaces the hand-rolled sidecar dicts in each export script.
        Pass training metrics (epoch, val_loss, …) as keyword args.
        """
        d: dict[str, Any] = {
            "contract": self.contract_id,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "model_name": self.model_name,
            "description": self.description,
            "onnx_opset": self.onnx_opset,
            "inputs": [s.to_dict() for s in self.inputs],
            "outputs": [s.to_dict() for s in self.outputs],
        }
        if self.extra:
            d["extra"] = self.extra
        if training_meta:
            d["training"] = training_meta
        return d

    def to_checkpoint_meta(self) -> dict[str, Any]:
        """Minimal metadata dict to embed in PyTorch checkpoints.

        This lets any script verify that a checkpoint was trained under
        a specific contract revision.
        """
        return {
            "contract_id": self.contract_id,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "model_name": self.model_name,
        }

    @classmethod
    def from_sidecar(cls, d: dict[str, Any]) -> "ModelContract":
        """Reconstruct a (partial) contract from a sidecar JSON dict."""
        return cls(
            model_name=d.get("model_name", "unknown"),
            revision=d.get("revision", 0),
            inputs=tuple(TensorSpec.from_dict(s) for s in d.get("inputs", [])),
            outputs=tuple(TensorSpec.from_dict(s) for s in d.get("outputs", [])),
            onnx_opset=d.get("onnx_opset", 18),
            contract_id=d.get("contract", ""),
            description=d.get("description", ""),
            extra=d.get("extra", {}),
        )


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ContractViolation(Exception):
    """Raised when live tensors don't match the declared contract."""
