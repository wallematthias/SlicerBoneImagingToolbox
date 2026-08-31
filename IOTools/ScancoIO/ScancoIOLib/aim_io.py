"""Lightweight AIM read/write helpers for the Scanco I/O Slicer module."""

from __future__ import annotations

from datetime import datetime
import importlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import SimpleITK as sitk


SUPPORTED_IMAGE_EXTENSIONS = (".aim", ".isq", ".scv", ".gobj")
SUPPORTED_IMAGE_FORMATS = ("aim", "isq", "scv", "gobj")
SEGMENTATION_FILENAME_TOKENS = {
    "contour",
    "contours",
    "label",
    "labels",
    "mask",
    "regmask",
    "roi1",
    "roi2",
    "seg",
    "segmentation",
}


def _load_py_aimio():
    try:
        return importlib.import_module("py_aimio")
    except ImportError as exc:
        raise RuntimeError(
            "py_aimio is required for Scanco image import/export. Use the module install button "
            "or install the PyPI package 'aimio-py' in Slicer Python."
        ) from exc


def is_aimio_available() -> bool:
    try:
        _load_py_aimio()
        return True
    except Exception:
        return False


def log_to_dict(log: str) -> dict[str, Any]:
    py_aimio = _load_py_aimio()
    return py_aimio.log_to_dict(log or "")


def dict_to_log(log_dict: dict[str, Any]) -> str:
    py_aimio = _load_py_aimio()
    return py_aimio.dict_to_log(log_dict or {})


def _get_aim_calibration_constants_from_processing_log(
    processing_log: str,
) -> tuple[int, float, float, float, float]:
    import re

    mu_scaling_match = re.search(r"Mu_Scaling\s+(\d+)", processing_log)
    hu_mu_water_match = re.search(r"HU: mu water\s+(\d+\.\d+)", processing_log)
    density_slope_match = re.search(
        r"Density: slope\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",
        processing_log,
    )
    density_intercept_match = re.search(
        r"Density: intercept\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",
        processing_log,
    )

    if not all(
        [
            mu_scaling_match,
            hu_mu_water_match,
            density_slope_match,
            density_intercept_match,
        ]
    ):
        raise ValueError("Could not parse AIM calibration constants from processing log.")

    mu_scaling = int(mu_scaling_match.group(1))
    hu_mu_water = float(hu_mu_water_match.group(1))
    hu_mu_air = 0.0
    density_slope = float(density_slope_match.group(1))
    density_intercept = float(density_intercept_match.group(1))
    return mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept


def _normalize_scaling(scaling: str) -> str:
    normalized = scaling.lower()
    if normalized in {"none", "native", "mu", "hu", "bmd", "density"}:
        return normalized
    raise ValueError(
        f"Unsupported scaling '{scaling}'. Use one of: native, none, mu, hu, bmd, density."
    )


def _normalize_image_scaling(scaling: str) -> str:
    normalized = str(scaling or "density").strip().lower()
    if normalized == "bmd":
        return "density"
    if normalized in {"density", "native", "none", "hu"}:
        return normalized
    raise ValueError(f"Unsupported scaling '{scaling}'. Use one of: density, native, HU.")


def _apply_scaling(np_image: np.ndarray, processing_log: str, scaling: str) -> np.ndarray:
    scaling = scaling.lower()
    if scaling in {"native", "none"}:
        return np_image

    mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = (
        _get_aim_calibration_constants_from_processing_log(processing_log)
    )

    if scaling == "mu":
        return np_image.astype(np.float32) / float(mu_scaling)

    if scaling == "hu":
        m = 1000.0 / (mu_scaling * (hu_mu_water - hu_mu_air))
        b = -1000.0 * hu_mu_water / (hu_mu_water - hu_mu_air)
        return np_image.astype(np.float32) * m + b

    if scaling in {"bmd", "density"}:
        return (
            np_image.astype(np.float32) / float(mu_scaling) * float(density_slope)
            + float(density_intercept)
        )

    raise ValueError(
        f"Unsupported scaling '{scaling}'. Use one of: native, none, mu, hu, bmd, density."
    )


def _as_zyx(array: np.ndarray, dimensions_xyz: tuple[int, int, int] | None) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected 3D AIM array, got shape {array.shape}.")

    if dimensions_xyz is None:
        return array

    expected_zyx = (dimensions_xyz[2], dimensions_xyz[1], dimensions_xyz[0])
    expected_xyz = dimensions_xyz
    if tuple(array.shape) == expected_zyx:
        return array
    if tuple(array.shape) == expected_xyz:
        return np.transpose(array, (2, 1, 0))
    return array


def supported_image_extensions() -> tuple[str, ...]:
    return SUPPORTED_IMAGE_EXTENSIONS


def resolve_image_format(path: Path | str, image_format: str = "auto") -> str:
    normalized = str(image_format or "auto").strip().lower()
    if normalized != "auto":
        if normalized not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError("image_format must be one of: auto, aim, isq, scv, or gobj.")
        return normalized

    path_without_version = str(path).rsplit(";", 1)
    path_text = path_without_version[0] if len(path_without_version) == 2 and path_without_version[1].isdigit() else str(path)
    suffix = Path(path_text).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return suffix.lstrip(".")
    raise ValueError("Could not infer image format from extension; use AIM, ISQ, SCV, or GOBJ.")


def suggested_slicer_load_as(path: Path | str, metadata: dict[str, Any] | None = None) -> str:
    metadata_format = str((metadata or {}).get("format") or "").strip().lower()
    image_format = metadata_format or resolve_image_format(path)
    if image_format == "gobj":
        return "segmentation"

    stem = Path(str(path).rsplit(";", 1)[0]).stem.lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    if tokens & SEGMENTATION_FILENAME_TOKENS:
        return "segmentation"
    return "volume"


def _image_scaling_kwargs(image_format: str, scaling: str) -> dict[str, Any]:
    scaling = _normalize_image_scaling(scaling)
    if image_format == "isq":
        return {"unit": "native" if scaling == "none" else scaling}
    if image_format == "aim":
        return {
            "density": scaling == "density",
            "hu": scaling == "hu",
        }
    return {}


def _resolve_origin(meta: dict[str, Any], spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    origin_raw = meta.get("origin")
    if isinstance(origin_raw, (list, tuple)) and len(origin_raw) == 3:
        return tuple(float(v) for v in origin_raw)

    position_raw = meta.get("position")
    if isinstance(position_raw, (list, tuple)) and len(position_raw) == 3:
        offset_raw = meta.get("offset", (0, 0, 0))
        if not (isinstance(offset_raw, (list, tuple)) and len(offset_raw) == 3):
            offset_raw = (0, 0, 0)
        return tuple(
            (float(position_raw[i]) + float(offset_raw[i]) + 0.5) * float(spacing[i])
            for i in range(3)
        )

    return (0.0, 0.0, 0.0)


def _metadata_vector_for_image(
    meta: dict[str, Any],
    key: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    raw = meta.get(key)
    if isinstance(raw, (list, tuple)):
        values = tuple(float(v) for v in raw[:3])
        if len(values) == 3:
            return values
        if len(values) == 2:
            return (values[0], values[1], default[2])
    return default


def _direction_for_image(meta: dict[str, Any], dimension: int) -> tuple[float, ...]:
    raw = meta.get("direction")
    if isinstance(raw, (list, tuple)):
        if dimension == 3 and len(raw) == 9:
            return tuple(float(v) for v in raw)
        if dimension == 2 and len(raw) == 4:
            return tuple(float(v) for v in raw)
    if dimension == 2:
        return (1.0, 0.0, 0.0, 1.0)
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _image_from_array_and_metadata(
    array: np.ndarray,
    meta: dict[str, Any],
    *,
    scaling: str,
) -> tuple[sitk.Image, dict[str, Any]]:
    np_arr = np.asarray(array)
    if np_arr.ndim == 2:
        np_arr = np_arr[np.newaxis, :, :]
    elif np_arr.ndim == 3:
        dims_raw = meta.get("dimensions")
        dimensions_xyz = (
            tuple(int(v) for v in dims_raw)
            if isinstance(dims_raw, (list, tuple)) and len(dims_raw) == 3
            else None
        )
        np_arr = _as_zyx(np_arr, dimensions_xyz)
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {np_arr.shape}.")

    spacing = _metadata_vector_for_image(
        meta,
        "spacing",
        _metadata_vector_for_image(meta, "element_size", (1.0, 1.0, 1.0)),
    )
    origin_raw = meta.get("origin")
    if isinstance(origin_raw, (list, tuple)) and len(origin_raw) >= 3:
        origin = tuple(float(v) for v in origin_raw[:3])
    else:
        origin = _resolve_origin(meta, spacing)

    image = sitk.GetImageFromArray(np_arr)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(_direction_for_image(meta, image.GetDimension()))

    metadata = dict(meta)
    image_format = str(metadata.get("format") or "").upper()
    unit = str(metadata.get("unit") or "").strip()
    if not unit:
        unit = "native" if image_format in {"SCV", "GOBJ"} else _normalize_image_scaling(scaling)
    metadata.update(
        {
            "origin": origin,
            "spacing": spacing,
            "element_size": spacing,
            "direction": tuple(float(v) for v in image.GetDirection()),
            "dimensions": tuple(int(v) for v in image.GetSize()),
            "unit": unit,
        }
    )
    if image_format:
        metadata["format"] = image_format
    return image, metadata


def image_with_aim_metadata_geometry(image: sitk.Image, metadata: dict[str, Any] | None) -> sitk.Image:
    """Return a copy of image with AIM spacing/origin restored when dimensions match."""
    if not metadata:
        return image

    dimensions_raw = metadata.get("dimensions")
    if isinstance(dimensions_raw, (list, tuple)) and len(dimensions_raw) == 3:
        dimensions = tuple(int(v) for v in dimensions_raw)
        if dimensions != tuple(int(v) for v in image.GetSize()):
            return image

    spacing_raw = metadata.get("element_size", metadata.get("spacing"))
    if not (isinstance(spacing_raw, (list, tuple)) and len(spacing_raw) == 3):
        return image

    spacing = tuple(float(v) for v in spacing_raw)
    origin = _resolve_origin(metadata, spacing)
    restored = sitk.Image(image)
    restored.SetSpacing(spacing)
    restored.SetOrigin(origin)

    direction_raw = metadata.get("direction")
    if isinstance(direction_raw, (list, tuple)) and len(direction_raw) == 9:
        restored.SetDirection(tuple(float(v) for v in direction_raw))

    return restored


def _metadata_vector(
    meta: dict[str, Any],
    key: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    raw = meta.get(key)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            return tuple(float(v) for v in raw)
        except (TypeError, ValueError):
            pass
    return default


def _aim_position_from_origin(
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    offset: tuple[float, float, float],
) -> tuple[int, int, int]:
    position = []
    for origin_value, spacing_value, offset_value in zip(origin, spacing, offset):
        if spacing_value == 0:
            position.append(0)
        else:
            position.append(int(round((origin_value / spacing_value) - offset_value)))
    return tuple(position)


def _refresh_position_from_image_geometry(meta: dict[str, Any], image: sitk.Image) -> None:
    spacing = tuple(float(v) for v in image.GetSpacing())
    origin = tuple(float(v) for v in image.GetOrigin())
    offset = _metadata_vector(meta, "offset", (0.0, 0.0, 0.0))
    meta["position"] = _aim_position_from_origin(origin, spacing, offset)
    meta["offset"] = tuple(int(round(v)) for v in offset)


def _prepare_aim_metadata_for_write(meta: dict[str, Any], image: sitk.Image, *, mask: bool = False) -> None:
    if mask:
        meta["unit"] = "native"
    meta["dimensions"] = tuple(int(v) for v in image.GetSize())
    meta["spacing"] = tuple(float(v) for v in image.GetSpacing())
    meta["element_size"] = tuple(float(v) for v in image.GetSpacing())
    meta["origin"] = tuple(float(v) for v in image.GetOrigin())
    meta["direction"] = tuple(float(v) for v in image.GetDirection())
    _refresh_position_from_image_geometry(meta, image)


def read_aim(path: Path, scaling: str = "bmd") -> tuple[sitk.Image, dict[str, Any]]:
    py_aimio = _load_py_aimio()
    scaling = _normalize_scaling(scaling)
    path = Path(path)

    np_arr, meta = py_aimio.read_aim(str(path), density=False, hu=False)
    meta = dict(meta)
    processing_log_value = meta.get("processing_log")
    processing_log = str(
        meta.get("processing_log_raw")
        or (dict_to_log(processing_log_value) if isinstance(processing_log_value, dict) else processing_log_value)
        or ""
    )

    dims_xyz_raw = meta.get("dimensions")
    dimensions_xyz: tuple[int, int, int] | None
    if isinstance(dims_xyz_raw, (list, tuple)) and len(dims_xyz_raw) == 3:
        dimensions_xyz = tuple(int(v) for v in dims_xyz_raw)
    else:
        dimensions_xyz = None

    np_arr = _as_zyx(np.asarray(np_arr), dimensions_xyz)
    np_arr = _apply_scaling(np_arr, processing_log, scaling)

    spacing_raw = meta.get("element_size", meta.get("spacing", (1.0, 1.0, 1.0)))
    if isinstance(spacing_raw, (list, tuple)) and len(spacing_raw) == 3:
        spacing = tuple(float(v) for v in spacing_raw)
    else:
        spacing = (1.0, 1.0, 1.0)
    origin = _resolve_origin(meta, spacing)

    image = sitk.GetImageFromArray(np_arr)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    image.SetMetaData("processing_log", processing_log.replace("\n", "_LINEBREAK_"))
    image.SetMetaData("unit", "native" if scaling in {"none", "native"} else scaling)

    metadata: dict[str, Any] = dict(meta)
    metadata.update({
        "origin": origin,
        "spacing": spacing,
        "element_size": spacing,
        "position": meta.get("position"),
        "offset": meta.get("offset"),
        "dimensions": dimensions_xyz
        if dimensions_xyz is not None
        else (image.GetSize()[0], image.GetSize()[1], image.GetSize()[2]),
        "processing_log": processing_log_value if isinstance(processing_log_value, dict) else log_to_dict(processing_log),
        "processing_log_raw": processing_log,
        "unit": image.GetMetaData("unit"),
    })
    return image, metadata


def read_image(
    path: Path | str,
    scaling: str = "density",
    image_format: str = "auto",
    **kwargs,
) -> tuple[sitk.Image, dict[str, Any]]:
    image_format = resolve_image_format(path, image_format)
    scaling = _normalize_image_scaling(scaling)

    if image_format == "aim":
        return read_aim(Path(path), scaling="density" if scaling == "density" else scaling)

    py_aimio = _load_py_aimio()
    read_kwargs = _image_scaling_kwargs(image_format, scaling)
    read_kwargs.update(kwargs)
    array, meta = py_aimio.read_image(str(path), format=image_format, **read_kwargs)
    metadata = dict(meta)
    metadata.setdefault("format", image_format.upper())
    return _image_from_array_and_metadata(array, metadata, scaling=scaling)


def _normalize_aim_write_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    text = str(unit).strip()
    normalized = text.lower()
    if normalized in {"bmd", "density"}:
        return "BMD"
    if normalized == "hu":
        return "HU"
    if normalized in {"native", "none"}:
        return "native"
    return text


def _bmd_to_native_int16(array: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    processing_log = str(
        metadata.get("processing_log_raw") or metadata.get("processing_log") or ""
    )
    mu_scaling, _hu_mu_water, _hu_mu_air, density_slope, density_intercept = (
        _get_aim_calibration_constants_from_processing_log(processing_log)
    )
    slope = float(density_slope) / float(mu_scaling)
    native = (array.astype(np.float32) - float(density_intercept)) / slope
    native = np.rint(native)
    native = np.clip(native, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return native.astype(np.int16)


def aim_metadata_from_import_json(
    metadata_json: Path,
    image: sitk.Image,
    *,
    log: str = "",
) -> dict[str, Any]:
    payload = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    source_meta = payload.get("image_metadata") or {}
    out = dict(source_meta)
    processing_log_value = source_meta.get("processing_log")
    processing_log = str(
        source_meta.get("processing_log_raw")
        or (dict_to_log(processing_log_value) if isinstance(processing_log_value, dict) else processing_log_value)
        or ""
    )
    if log:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        processing_log = f"{processing_log}\n[{stamp}] {log}."
    out["processing_log"] = log_to_dict(processing_log)
    out["processing_log_raw"] = processing_log
    out["dimensions"] = tuple(int(v) for v in image.GetSize())
    out["spacing"] = tuple(float(v) for v in image.GetSpacing())
    out["element_size"] = tuple(float(v) for v in image.GetSpacing())
    out["origin"] = tuple(float(v) for v in image.GetOrigin())
    out["direction"] = tuple(float(v) for v in image.GetDirection())
    out.setdefault("offset", (0, 0, 0))
    _refresh_position_from_image_geometry(out, image)
    return out


def write_aim(
    image: sitk.Image,
    path: Path,
    metadata: dict[str, Any] | None = None,
    *,
    unit: str | None = None,
    mask: bool = False,
) -> None:
    py_aimio = _load_py_aimio()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(metadata or {})
    arr_zyx = sitk.GetArrayFromImage(image)
    if mask:
        arr_zyx = (127 * (arr_zyx > 0)).astype(np.int8)
        unit = "native"

    _prepare_aim_metadata_for_write(meta, image, mask=mask)

    if unit is None:
        unit = _normalize_aim_write_unit(meta.get("unit"))
    write_unit = unit
    if unit == "BMD":
        arr_zyx = _bmd_to_native_int16(arr_zyx, meta)
        write_unit = "native"
        meta["unit"] = "native"
    elif unit is not None:
        meta["unit"] = unit

    if isinstance(meta.get("processing_log"), dict):
        meta["processing_log_raw"] = dict_to_log(meta["processing_log"])

    py_aimio.write_aim(str(path), arr_zyx, meta, unit=write_unit)
