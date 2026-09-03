# Installation

## Extension Manager

When the toolbox is listed for your Slicer version:

1. Open 3D Slicer.
2. Install `Bone Imaging Toolbox` from the Extension Manager.
3. Restart Slicer.
4. Open modules from the `Bone Imaging` category.
5. Open `Bone Imaging > Setup > Toolbox Setup` and install or update runtime packages.

## Manual Install From A Clone

Clone the repository:

```bash
git clone https://github.com/wallematthias/SlicerBoneImagingToolbox.git
```

Then run the local linking helper in Slicer's Python Interactor:

```python
script = "/path/to/SlicerBoneImagingToolbox/scripts/link_local_toolbox_modules.py"
exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
```

Restart Slicer after linking.

## Runtime Packages

The Setup module is the canonical installer for public runtime packages used by the toolbox. It checks package versions in Slicer Python and offers install/update buttons for missing or outdated packages.

Common runtime packages include:

- `aimio-py` / `py_aimio`
- `bone-imaging-derivatives`
- `bone-contouring`
- `timelapsed-hrpqct`
- `bone-microarchitecture`
- `plate-rod-thinning`
- `parosol-py`
- `bone-mechanoregulation`
- `motionscorehrpqct`
- `spine-segment`

Motion Scoring and Spine Segmentation may also require Slicer's `PyTorch` extension depending on the selected backend.

## Local Development Check

For a quick local wrapper check:

```bash
python3 -m py_compile HRpQCTTools/*/*.py IOTools/*/*.py CTTools/*/*.py Setup/*/*.py
python3 -m pytest tests/test_package_status.py tests/test_batch_processor_module.py -q
```

Some tests import Slicer-only modules and must be run inside Slicer Python.
