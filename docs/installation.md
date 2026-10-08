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

## Updating A GitHub Installation

Open `Bone Imaging > Setup > Toolbox Setup`, click **Check for updates**, then
**Update toolbox**. This updates toolbox code from GitHub, separately from its
Python runtime packages:

- A Git clone uses `git pull --ff-only` on its configured upstream. Git must be
  available, and the checkout must be clean and fast-forwardable. Local changes
  are not overwritten. For the current public toolbox, use the `main` branch
  tracking `origin/main`.
- A GitHub ZIP download is replaced with the latest `main` archive after backing
  up the previous folder beside it. Keep the download in a writable folder.
- An Extension Manager installation must be updated through Slicer's Extension
  Manager, not this button.

Restart Slicer after updating the toolbox. Then return to Toolbox Setup and use
**Check package updates** and **Install / update needed** for its runtime
packages. Updating packages alone does not update toolbox modules.

If updating fails, retain the error dialog and Setup log. Include the installed
and latest revisions, installation folder, Slicer version, and whether the
download was a Git clone or ZIP. Network restrictions, folder permissions, and
Git availability or checkout state can affect the update; a newer GitHub tag
does not itself update an installed copy.

## Runtime Packages

The Setup module is the canonical installer for public runtime packages used by the toolbox. It checks package versions in Slicer Python and offers install/update buttons for missing or outdated packages.

Toolbox installers preserve the installed SimpleITK version with a pip constraint.
Slicer's bundled build supports fast in-memory image transfer and must not be
replaced by a generic PyPI wheel. Dependency-wide force reinstalls are blocked;
targeted compiled-package reinstalls use `--no-deps`. Restart Slicer after updates.
If SimpleITK was already replaced, see [troubleshooting](troubleshooting.md#simpleitk-mrmlidimageio-warning).

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

Setup may use sibling editable checkouts for Python-only development packages.
Wheel-only compiled requirements, including Plate/Rod Morphometry, always use
the configured binary install command even when a sibling checkout exists. This
avoids triggering a local compiler build during normal Slicer package updates.

For a quick local wrapper check:

```bash
python3 -m py_compile HRpQCTTools/*/*.py IOTools/*/*.py CTTools/*/*.py Setup/*/*.py
python3 -m pytest tests/test_package_status.py tests/test_batch_processor_module.py -q
```

Some tests import Slicer-only modules and must be run inside Slicer Python.
