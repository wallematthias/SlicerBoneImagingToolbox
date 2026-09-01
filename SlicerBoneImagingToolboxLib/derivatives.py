from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from bone_imaging_derivatives import read_manifest as read_shared_manifest


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


def find_records(manifest: DerivativeManifest, **filters: Any) -> list[DerivativeRecord]:
    records: Iterable[DerivativeRecord] = manifest.records
    for field_name, expected in filters.items():
        if expected is None:
            continue
        records = [record for record in records if getattr(record, field_name) == expected]
    return list(records)
