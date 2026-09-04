from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

_DERIVATIVES_IMPORT_ERROR = None
try:
    from bone_imaging_derivatives import (  # type: ignore
        ArtifactIndex,
        ArtifactRecord,
        apply_overrides,
        build_naming_rows,
        build_rename_plan,
        discover_artifacts as discover_shared_artifacts,
        execute_rename_plan,
        split_identity_metadata,
        normalize_role,
        normalize_session_id,
        normalize_site,
        normalize_subject_id,
        read_manifest as read_shared_manifest,
        site_category,
        suggested_filename,
        suggested_mids_relative_path,
        suggested_mids_relative_paths,
        undo_rename_manifest,
    )
    import bone_imaging_derivatives.naming as _naming_api  # type: ignore

    if not hasattr(_naming_api, "apply_naming_row_overrides"):
        import importlib

        _naming_api = importlib.reload(_naming_api)
    apply_naming_row_overrides = _naming_api.apply_naming_row_overrides
except Exception as exc:
    _DERIVATIVES_IMPORT_ERROR = exc
    ArtifactIndex = ArtifactRecord = object

    def _missing_derivatives_runtime(*_args, **_kwargs):
        raise RuntimeError(
            "The Bone Imaging Derivative Contract runtime package is not installed. "
            "Open Bone Imaging > Setup and install/update runtime packages."
        ) from _DERIVATIVES_IMPORT_ERROR

    apply_overrides = _missing_derivatives_runtime
    apply_naming_row_overrides = _missing_derivatives_runtime
    build_naming_rows = _missing_derivatives_runtime
    build_rename_plan = _missing_derivatives_runtime
    discover_shared_artifacts = _missing_derivatives_runtime
    execute_rename_plan = _missing_derivatives_runtime
    normalize_role = _missing_derivatives_runtime
    normalize_session_id = _missing_derivatives_runtime
    normalize_site = _missing_derivatives_runtime
    normalize_subject_id = _missing_derivatives_runtime
    read_shared_manifest = _missing_derivatives_runtime
    site_category = _missing_derivatives_runtime
    split_identity_metadata = _missing_derivatives_runtime
    suggested_filename = _missing_derivatives_runtime
    suggested_mids_relative_path = _missing_derivatives_runtime
    suggested_mids_relative_paths = _missing_derivatives_runtime
    undo_rename_manifest = _missing_derivatives_runtime


@dataclass(frozen=True)
class DerivativeRecord:
    derivative: str
    role: str
    subject_id: str
    site: str
    session_id: str
    stack_index: int | None
    space: str
    path: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DerivativeManifest:
    workflow: str
    version: str
    dataset_root: str
    records: list[DerivativeRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _record_from_mapping(payload: dict[str, Any]) -> DerivativeRecord:
    return DerivativeRecord(
        derivative=str(payload.get("derivative", "")),
        role=str(payload.get("role", "")),
        subject_id=str(payload.get("subject_id", "")),
        site=str(payload.get("site", "")),
        session_id=str(payload.get("session_id", "")),
        stack_index=int(payload.get("stack_index", 1) or 1),
        space=str(payload.get("space", "")),
        path=str(payload.get("path", "")),
        source=str(payload.get("source", "")),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _manifest_from_mapping(payload: dict[str, Any]) -> DerivativeManifest:
    records = [_record_from_mapping(item) for item in payload.get("records", []) or []]
    return DerivativeManifest(
        workflow=str(payload.get("workflow", "")),
        version=str(payload.get("version", "")),
        dataset_root=str(payload.get("dataset_root", "")),
        records=records,
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def write_manifest(path: str | Path, manifest: DerivativeManifest) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def read_manifest(path: str | Path) -> DerivativeManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Derivative manifest must contain a JSON object: {path}")
    return _manifest_from_mapping(payload)


def discover_manifests(root: str | Path) -> list[DerivativeManifest]:
    base = Path(root)
    if not base.exists():
        return []
    manifests = []
    for path in sorted(base.rglob("manifest.json")):
        # Discover each file independently: a dataset may contain both
        # schema-v1 package manifests and pre-contract Slicer manifests.
        try:
            manifest = read_shared_manifest(path)
        except (OSError, ValueError):
            manifests.append(read_manifest(path))
            continue
        manifests.append(
            DerivativeManifest(
                workflow=manifest.derivative_family,
                version=str(manifest.schema_version),
                dataset_root=str(manifest.dataset_root),
                records=[
                    DerivativeRecord(
                        record.derivative, record.role, record.subject_id,
                        record.site, str(record.session_id or ""),
                        record.stack_index, record.space,
                        str(record.path), record.source, dict(record.metadata),
                    )
                    for record in manifest.records
                ],
                metadata={"shared_contract": True},
            )
        )
    return manifests


def discover_artifacts(root: str | Path, *, include_derivatives: bool = True) -> ArtifactIndex:
    """Discover loose images, masks, transforms, and tables with shared semantics."""
    return discover_shared_artifacts(root, include_derivatives=include_derivatives)


def find_records(manifest: DerivativeManifest, **filters: Any) -> list[DerivativeRecord]:
    records: Iterable[DerivativeRecord] = manifest.records
    for field_name, expected in filters.items():
        if expected is None:
            continue
        records = [record for record in records if getattr(record, field_name) == expected]
    return list(records)
