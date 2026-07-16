from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


TOOLBOX_NAME = "SlicerBoneImagingToolbox"
TOOLBOX_DISPLAY_NAME = "Bone Imaging Toolbox"
MANIFEST_NAME = "toolbox_modules.json"
DEFAULT_BUILTIN_MODULE_DIRS = (
    "HRpQCTTools/TimelapsedHRpQCT",
    "HRpQCTTools/MotionScoreHRpQCT",
    "HRpQCTTools/SegmentationHRpQCT",
    "IOTools/ScancoIO",
)
DEFAULT_EXTERNAL_MODULE_ROOTS = ("ExternalModules",)


@dataclass(frozen=True)
class ToolboxModule:
    path: str
    title: str
    section: str
    source: str = "built-in"


def _read_manifest(root: Path) -> dict:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{MANIFEST_NAME} must contain a JSON object.")
    return payload


def builtin_module_dirs(root: str | Path) -> tuple[str, ...]:
    payload = _read_manifest(Path(root))
    modules = payload.get("modules")
    if not isinstance(modules, list):
        return DEFAULT_BUILTIN_MODULE_DIRS
    paths = []
    for module in modules:
        if isinstance(module, dict) and module.get("source", "built-in") == "built-in":
            path = str(module.get("path") or "").strip()
            if path:
                paths.append(path)
    return tuple(paths) or DEFAULT_BUILTIN_MODULE_DIRS


def external_module_roots(root: str | Path) -> tuple[str, ...]:
    payload = _read_manifest(Path(root))
    roots = payload.get("external_module_roots")
    if not isinstance(roots, list):
        return DEFAULT_EXTERNAL_MODULE_ROOTS
    clean_roots = tuple(str(item).strip() for item in roots if str(item).strip())
    return clean_roots or DEFAULT_EXTERNAL_MODULE_ROOTS


def listed_modules(root: str | Path) -> tuple[ToolboxModule, ...]:
    payload = _read_manifest(Path(root))
    modules = payload.get("modules")
    if not isinstance(modules, list):
        return tuple(
            ToolboxModule(path=path, title=path, section="HR-pQCT")
            for path in DEFAULT_BUILTIN_MODULE_DIRS
        )
    listed: list[ToolboxModule] = []
    for module in modules:
        if not isinstance(module, dict):
            continue
        path = str(module.get("path") or "").strip()
        if not path:
            continue
        listed.append(
            ToolboxModule(
                path=path,
                title=str(module.get("title") or path).strip(),
                section=str(module.get("section") or "Other").strip(),
                source=str(module.get("source") or "built-in").strip(),
            )
        )
    return tuple(listed)


def is_slicer_scripted_module_dir(path: str | Path) -> bool:
    module_dir = Path(path)
    module_name = module_dir.name
    return (
        module_dir.is_dir()
        and (module_dir / "CMakeLists.txt").is_file()
        and (module_dir / f"{module_name}.py").is_file()
    )


def discover_external_module_dirs(root: str | Path) -> tuple[Path, ...]:
    toolbox_root = Path(root)
    discovered: list[Path] = []
    seen: set[Path] = set()
    for relative_root in external_module_roots(toolbox_root):
        search_root = (toolbox_root / relative_root).resolve()
        if not search_root.is_dir():
            continue
        for candidate in [search_root, *search_root.glob("*"), *search_root.glob("*/*")]:
            if not is_slicer_scripted_module_dir(candidate):
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(resolved)
    return tuple(discovered)


def toolbox_module_dirs(root: str | Path, *, include_external: bool = True) -> tuple[Path, ...]:
    toolbox_root = Path(root)
    builtins = tuple((toolbox_root / path).resolve() for path in builtin_module_dirs(toolbox_root))
    if not include_external:
        return builtins
    return (*builtins, *discover_external_module_dirs(toolbox_root))
