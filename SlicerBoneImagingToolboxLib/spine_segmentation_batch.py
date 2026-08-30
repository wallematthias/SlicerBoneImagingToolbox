"""Pure helpers for CT spine segmentation batch runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .derivatives import DerivativeManifest, DerivativeRecord, discover_manifests, read_manifest, write_manifest


IMAGE_SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".nhdr")
DERIVATIVE_NAME = "SpineSegmentationCT"


@dataclass(frozen=True)
class SpineSegmentationImage:
    role: str
    path: str
    source: str
    derivative: str = ""
    space: str = "native"


@dataclass(frozen=True)
class SpineSegmentationBatchCase:
    subject_id: str
    site: str
    session_id: str
    images: tuple[SpineSegmentationImage, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject_id, self.site, self.session_id)

    def first_image(self, role: str = "") -> SpineSegmentationImage | None:
        role_text = str(role or "").strip()
        if role_text:
            for image in self.images:
                if image.role == role_text:
                    return image
            return None
        for role_candidate in ("calibrated_image", "density_image", "image", "raw_image"):
            image = self.first_image(role_candidate)
            if image is not None:
                return image
        return self.images[0] if self.images else None


@dataclass(frozen=True)
class SpineSegmentationBatchCommand:
    case: SpineSegmentationBatchCase
    input_path: Path
    output_dir: Path
    mode: str
    device: str
    cli_args: list[str]


def discover_spine_segmentation_batch_cases(
    dataset_root: str | Path,
    *,
    subject_id: str = "",
    site: str = "",
    session_id: str = "",
) -> list[SpineSegmentationBatchCase]:
    """Discover CT image inputs grouped by subject, site, and session."""
    root = Path(dataset_root).expanduser().resolve()
    images_by_key: dict[tuple[str, str, str], list[SpineSegmentationImage]] = {}

    def add_image(key: tuple[str, str, str], image: SpineSegmentationImage) -> None:
        if subject_id and key[0] != _clean_token(subject_id):
            return
        if site and key[1] != _clean_token(site):
            return
        if session_id and key[2] != _clean_token(session_id):
            return
        if not key[0] or not key[1]:
            return
        bucket = images_by_key.setdefault(key, [])
        if image.path not in {existing.path for existing in bucket}:
            bucket.append(image)

    for manifest in discover_manifests(root / "derivatives"):
        for record in manifest.records:
            role = _normalize_image_role(record.role, record.path)
            if role is None:
                continue
            path = _resolve_record_path(record.path, dataset_root=root, manifest_root=manifest.dataset_root)
            key = (
                _clean_token(record.subject_id),
                _clean_token(record.site or "spine"),
                _clean_token(record.session_id),
            )
            add_image(
                key,
                SpineSegmentationImage(
                    role=role,
                    path=str(path),
                    source=record.source or "manifest",
                    derivative=record.derivative,
                    space=record.space or "native",
                ),
            )

    for path in _iter_image_paths(root):
        tokens = _tokens_from_path(path, root)
        key = (tokens["subject_id"], tokens["site"], tokens["session_id"])
        add_image(
            key,
            SpineSegmentationImage(
                role=_normalize_image_role("", str(path)) or "image",
                path=str(path),
                source="dataset",
            ),
        )

    return [
        SpineSegmentationBatchCase(
            subject_id=key[0],
            site=key[1],
            session_id=key[2],
            images=tuple(_sort_images(images)),
        )
        for key, images in sorted(images_by_key.items())
    ]


def build_spine_segmentation_batch_commands(
    dataset_root: str | Path,
    cases: Iterable[SpineSegmentationBatchCase],
    *,
    image_role: str = "",
    mode: str = "full",
    device: str = "auto",
) -> list[SpineSegmentationBatchCommand]:
    """Build one spine-segment command per case without copying input images."""
    root = Path(dataset_root).expanduser().resolve()
    commands: list[SpineSegmentationBatchCommand] = []
    for case in cases:
        image = case.first_image(image_role)
        if image is None:
            continue
        output_dir = (
            root
            / "derivatives"
            / DERIVATIVE_NAME
            / f"sub-{_clean_token(case.subject_id)}"
            / f"site-{_clean_token(case.site)}"
            / f"ses-{_clean_token(case.session_id)}"
        )
        cli_args = [
            str(Path(image.path).expanduser()),
            "--output",
            str(output_dir),
            "--device",
            str(device or "auto"),
            "--overwrite",
        ]
        normalized_mode = str(mode or "full").strip().lower()
        if normalized_mode == "localization":
            cli_args.append("--localization-only")
        elif normalized_mode == "level":
            cli_args.append("--level-only")
        commands.append(
            SpineSegmentationBatchCommand(
                case=case,
                input_path=Path(image.path).expanduser(),
                output_dir=output_dir,
                mode=normalized_mode,
                device=str(device or "auto"),
                cli_args=cli_args,
            )
        )
    return commands


def write_spine_segmentation_manifest(
    dataset_root: str | Path,
    commands: Iterable[SpineSegmentationBatchCommand],
    *,
    module_version: str,
) -> Path:
    """Write or update the SpineSegmentationCT derivative manifest."""
    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / "derivatives" / DERIVATIVE_NAME / "manifest.json"
    existing_records = []
    if manifest_path.exists():
        try:
            existing_records = list(read_manifest(manifest_path).records)
        except Exception:
            existing_records = []
    records_by_key = {
        (
            record.subject_id,
            record.site,
            record.session_id,
            record.role,
            str(Path(record.path).expanduser()),
        ): record
        for record in existing_records
    }
    for command in commands:
        for record in _records_for_command(command):
            records_by_key[
                (
                    record.subject_id,
                    record.site,
                    record.session_id,
                    record.role,
                    str(Path(record.path).expanduser()),
                )
            ] = record
    manifest = DerivativeManifest(
        workflow=DERIVATIVE_NAME,
        version=str(module_version),
        dataset_root=str(root),
        records=list(records_by_key.values()),
        metadata={"interface": "slicer-batch"},
    )
    return write_manifest(manifest_path, manifest)


def discovered_image_roles(cases: Iterable[SpineSegmentationBatchCase]) -> list[str]:
    """Return discovered image roles in preferred UI order."""
    roles = {image.role for case in cases for image in case.images}
    order = ("calibrated_image", "density_image", "image", "raw_image")
    ordered = [role for role in order if role in roles]
    return ordered + sorted(roles.difference(ordered))


def _records_for_command(command: SpineSegmentationBatchCommand) -> list[DerivativeRecord]:
    stem = _strip_image_suffix(command.input_path.name)
    outputs = {
        "vertebral_level_segmentation": command.output_dir / f"{stem}_vertebral-level.nii.gz",
        "process_body_segmentation": command.output_dir / f"{stem}_process-body.nii.gz",
        "cort_trab_segmentation": command.output_dir / f"{stem}_cort-trab.nii.gz",
        "vertebral_centroids": command.output_dir / f"{stem}_centroids.json",
    }
    records = []
    for role, path in outputs.items():
        if not path.exists():
            continue
        records.append(
            DerivativeRecord(
                derivative=DERIVATIVE_NAME,
                role=role,
                subject_id=command.case.subject_id,
                site=command.case.site,
                session_id=command.case.session_id,
                stack_index=None,
                space="native",
                path=str(path),
                source="spine-segment",
                metadata={"mode": command.mode, "device": command.device},
            )
        )
    return records


def _iter_image_paths(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or not _has_image_suffix(path):
            continue
        parts = {part.lower() for part in path.parts}
        if "derivatives" in parts and DERIVATIVE_NAME.lower() in parts:
            continue
        paths.append(path)
    return sorted(paths)


def _has_image_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in IMAGE_SUFFIXES)


def _resolve_record_path(path: str, *, dataset_root: Path, manifest_root: str = "") -> Path:
    record_path = Path(path).expanduser()
    if record_path.is_absolute():
        return record_path
    base = Path(manifest_root).expanduser() if str(manifest_root or "").strip() else dataset_root
    return (base / record_path).resolve()


def _normalize_image_role(role: str, path: str) -> str | None:
    text = str(role or "").strip().lower().replace("-", "_")
    name = Path(path).name.lower().replace("-", "_")
    if text in {"calibrated_image", "density_image", "image", "raw_image"}:
        return text
    if text in {"ct", "ct_image", "volume", "scalar_volume"}:
        return "image"
    if "calibrated" in name:
        return "calibrated_image"
    if "density" in name or "bmd" in name:
        return "density_image"
    if _has_image_suffix(Path(path)) and not any(mask in name for mask in ("mask", "seg", "label")):
        return "image"
    return None


def _tokens_from_path(path: Path, root: Path) -> dict[str, str]:
    rel_parts = path.relative_to(root).parts if path.is_relative_to(root) else path.parts
    subject = ""
    session = ""
    site = "spine"
    for part in rel_parts:
        clean = _clean_token(part)
        lower = clean.lower()
        if lower.startswith("sub-"):
            subject = clean[4:]
        elif lower.startswith("ses-"):
            session = clean[4:]
        elif lower.startswith("site-"):
            site = clean[5:]
    stem = _strip_image_suffix(path.name)
    for match in re.finditer(r"(?i)(?:^|[_-])(sub-[A-Za-z0-9_.-]+|ses-[A-Za-z0-9_.-]+|site-[A-Za-z0-9_.-]+)(?=[_-]|$)", stem):
        token = match.group(1)
        lower = token.lower()
        if lower.startswith("sub-"):
            subject = _clean_token(token[4:])
        elif lower.startswith("ses-"):
            session = _clean_token(token[4:])
        elif lower.startswith("site-"):
            site = _clean_token(token[5:])
    if not subject:
        subject = _clean_token(path.parent.parent.name if path.parent.parent != root else path.parent.name)
    if not session:
        session = _clean_token(path.parent.name)
    return {"subject_id": subject, "site": site or "spine", "session_id": session}


def _strip_image_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in sorted(IMAGE_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _sort_images(images: Iterable[SpineSegmentationImage]) -> list[SpineSegmentationImage]:
    order = {"calibrated_image": 0, "density_image": 1, "image": 2, "raw_image": 3}
    return sorted(images, key=lambda image: (order.get(image.role, 99), image.path))


def _clean_token(value: str) -> str:
    token = str(value or "").strip()
    if token.lower().startswith("sub-"):
        token = token[4:]
    elif token.lower().startswith("ses-"):
        token = token[4:]
    elif token.lower().startswith("site-"):
        token = token[5:]
    token = re.sub(r"[^0-9A-Za-z_.-]+", "-", token)
    return token.strip("-")
