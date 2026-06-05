"""Lightweight AIM read/write helpers for the Scanco I/O Slicer module."""

from __future__ import annotations

from datetime import datetime
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


def _load_py_aimio():
    try:
        return importlib.import_module("py_aimio")
    except ImportError as exc:
        raise RuntimeError(
            "py_aimio is required for AIM import/export. Use the module install button "
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
            position.append(int(round((origin_value / spacing_value) - offset_value - 0.5)))
    return tuple(position)


def _refresh_position_from_image_geometry(meta: dict[str, Any], image: sitk.Image) -> None:
    spacing = tuple(float(v) for v in image.GetSpacing())
    origin = tuple(float(v) for v in image.GetOrigin())
    offset = _metadata_vector(meta, "offset", (0.0, 0.0, 0.0))
    meta["position"] = _aim_position_from_origin(origin, spacing, offset)
    meta["offset"] = tuple(int(round(v)) for v in offset)


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

    arr_zyx = sitk.GetArrayFromImage(image)
    if mask:
        arr_zyx = (127 * (arr_zyx > 0)).astype(np.int8)

    meta = dict(metadata or {})
    meta.setdefault("dimensions", tuple(int(v) for v in image.GetSize()))
    meta.setdefault("spacing", tuple(float(v) for v in image.GetSpacing()))
    meta.setdefault("element_size", tuple(float(v) for v in image.GetSpacing()))
    meta.setdefault("origin", tuple(float(v) for v in image.GetOrigin()))
    meta.setdefault("direction", tuple(float(v) for v in image.GetDirection()))
    _refresh_position_from_image_geometry(meta, image)

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
