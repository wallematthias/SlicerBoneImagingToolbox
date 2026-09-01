from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk


def read_image(path: str | Path) -> sitk.Image:
    return sitk.ReadImage(str(Path(path)), sitk.sitkFloat32)


def read_mask(path: str | Path) -> sitk.Image:
    image = sitk.ReadImage(str(Path(path)), sitk.sitkUInt8)
    return sitk.Cast(image > 0, sitk.sitkUInt8)


def write_mask(path: str | Path, mask: sitk.Image) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(mask > 0, sitk.sitkUInt8), str(output_path))
    return output_path
