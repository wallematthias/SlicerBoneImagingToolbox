"""Batch execution backend registry.

The public toolbox owns the Batch Processor UI and local execution. Optional
private packages can register remote execution backends without copying or
patching the public module.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Protocol


PRIVATE_BACKEND_MODULES_ENV_VAR = "SLICER_BONE_BATCH_BACKEND_MODULES"


class BatchBackendProtocol(Protocol):
    """Minimal public contract for optional batch execution backends."""

    key: str
    label: str


@dataclass(frozen=True)
class LocalBatchBackend:
    """Default in-process/local Slicer batch execution backend."""

    key: str = "local"
    label: str = "Local"


_BACKENDS: dict[str, BatchBackendProtocol] = {}
_PRIVATE_MODULES_LOADED = False


def register_batch_backend(backend: BatchBackendProtocol) -> None:
    """Register a batch backend by key.

    Private toolbox modules should call this during import. Backends only need
    to expose ``key`` and ``label`` for UI discovery; richer methods can be
    used by code that knows that backend type.
    """
    key = str(getattr(backend, "key", "") or "").strip()
    label = str(getattr(backend, "label", "") or "").strip()
    if not key:
        raise ValueError("Batch backend needs a non-empty key.")
    if not label:
        raise ValueError(f"Batch backend {key!r} needs a non-empty label.")
    _BACKENDS[key] = backend


def available_batch_backends(*, include_private: bool = True) -> dict[str, BatchBackendProtocol]:
    """Return registered backends keyed by backend id."""
    _ensure_default_backend()
    if include_private:
        _load_private_backend_modules()
    return dict(_BACKENDS)


def get_batch_backend(key: str) -> BatchBackendProtocol:
    """Return one registered backend."""
    backends = available_batch_backends()
    backend_key = str(key or "local").strip() or "local"
    try:
        return backends[backend_key]
    except KeyError as exc:
        raise KeyError(f"Unknown batch backend: {backend_key}") from exc


def _ensure_default_backend() -> None:
    if "local" not in _BACKENDS:
        register_batch_backend(LocalBatchBackend())


def _load_private_backend_modules() -> None:
    global _PRIVATE_MODULES_LOADED
    if _PRIVATE_MODULES_LOADED:
        return
    _PRIVATE_MODULES_LOADED = True
    for module_name in _private_backend_module_names():
        try:
            importlib.import_module(module_name)
        except Exception:
            continue


def _private_backend_module_names() -> tuple[str, ...]:
    configured = os.environ.get(PRIVATE_BACKEND_MODULES_ENV_VAR, "")
    names = [name.strip() for name in configured.split(os.pathsep) if name.strip()]
    names.extend(
        [
            "SlicerBoneImagingToolboxPrivate.batch_backends",
            "SlicerBoneImagingToolbox_private.batch_backends",
            "slicer_bone_imaging_toolbox_private.batch_backends",
        ]
    )
    unique: list[str] = []
    for name in names:
        if name and name not in unique:
            unique.append(name)
    return tuple(unique)


def reset_batch_backends_for_tests() -> None:
    """Reset registry state for tests."""
    global _PRIVATE_MODULES_LOADED
    _BACKENDS.clear()
    _PRIVATE_MODULES_LOADED = False
    _ensure_default_backend()


__all__ = [
    "BatchBackendProtocol",
    "LocalBatchBackend",
    "PRIVATE_BACKEND_MODULES_ENV_VAR",
    "available_batch_backends",
    "get_batch_backend",
    "register_batch_backend",
    "reset_batch_backends_for_tests",
]
