"""Scene execution/validation only; scientific inference lives in the GPL core."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from .masks import assert_same_geometry

DEFAULT_MODEL_LABEL = "radius_tibia_final"
SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")


def model_cache_directory(app_data):
    """Persistent Slicer models, alongside MotionScore rather than in datasets."""
    root = Path(app_data) if str(app_data or "").strip() else Path.home()/".hrpqct-segmentation"
    return root/"HRpQCTSegmentation"/"models"


@dataclass(frozen=True)
class SceneCommand:
    program: str
    args: list[str]
    cwd: Path


def build_scene_command(python_exe, snapshot, output, device="auto"):
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if device not in SUPPORTED_DEVICES:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    if output.exists():
        raise FileExistsError(output)
    return SceneCommand(str(python_exe), ["-m", "bone_contouring.unet.scene_worker", str(snapshot),
                        str(output), "--device", device], output.parent)


def local_core_source(toolbox_root):
    """Development convenience matching the Toolbox's other local core packages."""
    for base in Path(toolbox_root).resolve().parents:
        source = base/"bone-contouring"/"src"
        if (source/"bone_contouring"/"unet").is_dir():
            return source
    return None


def scene_worker_args(args, local_source):
    """Prefer local code even when PythonSlicer prepends installed site-packages."""
    args = list(args)
    if local_source is None or args[:2] != ["-m", "bone_contouring.unet.scene_worker"]:
        return args
    bootstrap = ("import sys; "
                 f"sys.path.insert(0, {str(local_source)!r}); "
                 "from bone_contouring.unet.scene_worker import main; "
                 "raise SystemExit(main())")
    return ["-c", bootstrap, *args[2:]]


def align_native_aim_mask(image, metadata, reference, source_metadata):
    """Restore source geometry only for AIM-v3's sub-nanometre spacing quantization.

    No interpolation. Integer dimensions, position and offset must match exactly;
    changed reference geometry or an actual shifted/rescaled mask is rejected.
    """
    try:
        assert_same_geometry([image, reference], context="U-Net masks and source")
        return image
    except ValueError:
        if not source_metadata:
            raise
    for key in ("position", "offset", "dimensions"):
        if key not in metadata or key not in source_metadata or tuple(metadata[key]) != tuple(source_metadata[key]):
            raise ValueError(f"U-Net AIM lattice mismatch: {key}")
    if tuple(metadata["dimensions"]) != reference.GetSize() or image.GetSize() != reference.GetSize():
        raise ValueError("U-Net AIM dimensions differ from reference")
    source_spacing = np.asarray(source_metadata["element_size"], dtype=float)
    written_spacing = np.asarray(metadata["element_size"], dtype=float)
    if (not np.isfinite(source_spacing).all() or not np.isfinite(written_spacing).all()
            or np.any(source_spacing <= 0) or np.any(written_spacing <= 0)
            or not np.allclose(source_spacing, written_spacing, rtol=0, atol=1e-6)
            or not np.allclose(reference.GetSpacing(), source_spacing, rtol=0, atol=1e-9)
            or not np.allclose(image.GetSpacing(), written_spacing, rtol=0, atol=1e-9)
            or not np.allclose(reference.GetOrigin(), source_metadata["origin"], rtol=0, atol=1e-6)
            or not np.allclose(image.GetOrigin(), metadata["origin"], rtol=0, atol=1e-6)
            or not np.allclose(image.GetDirection(), reference.GetDirection(), rtol=0, atol=1e-9)):
        raise ValueError("U-Net AIM geometry differs by more than native writer quantization")
    aligned = sitk.Image(image)
    aligned.CopyInformation(reference)
    return aligned


def validate_deep_learning_compartments(cort, trab, reference):
    if any(image.GetDimension() != 3 for image in (cort,trab,reference)):
        raise ValueError("Deep-learning compartments require 3D images.")
    assert_same_geometry([cort,trab,reference], context="predicted masks and selected scan")
    arrays, masks = {}, {}
    for role, image in (("cort",cort),("trab",trab)):
        array = sitk.GetArrayViewFromImage(image)
        foreground = np.unique(array)
        foreground = foreground[foreground != 0]
        if len(foreground) != 1 or foreground[0] not in (1,127,255):
            raise ValueError(f"The {role} output must be a nonempty binary compartment mask.")
        arrays[role] = array > 0
        masks[role] = sitk.GetImageFromArray(arrays[role].astype(np.uint8))
        masks[role].CopyInformation(reference)
    if np.any(arrays["cort"] & arrays["trab"]):
        raise ValueError("Predicted cortical and trabecular compartments overlap.")
    masks["full"] = sitk.GetImageFromArray((arrays["cort"]|arrays["trab"]).astype(np.uint8))
    masks["full"].CopyInformation(reference)
    return masks
