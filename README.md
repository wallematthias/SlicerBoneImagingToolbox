<p align="center">
  <img src="resources/SlicerBoneImagingToolbox.png" alt="Bone Imaging Toolbox logo" width="360">
</p>

# Bone Imaging Toolbox for 3D Slicer

Bone Imaging Toolbox is a 3D Slicer extension for bone-imaging workflows. It groups focused tools for longitudinal HR-pQCT analysis, motion grading, Scanco AIM import/export, and HR-pQCT segmentation into one installable Slicer extension.

The extension appears in Slicer as:

```text
Bone Imaging
  I/O
    Scanco I/O
  HR-pQCT
    Timelapsed HR-pQCT
    Motion Scoring
    Segmentation and Contours
```

## Included Tools

| Tool | Slicer Category | What It Does | Guide |
| --- | --- | --- | --- |
| Timelapsed HR-pQCT | `Bone Imaging > HR-pQCT` | Runs longitudinal HR-pQCT processing, registration, remodelling analysis, and review. | [Timelapsed HR-pQCT](docs/tools/timelapsed-hrpqct.md) |
| Motion Scoring | `Bone Imaging > HR-pQCT` | Runs and reviews HR-pQCT motion grading using MotionScore models. | [Motion Scoring](docs/tools/motion-scoring.md) |
| Segmentation and Contours | `Bone Imaging > HR-pQCT` | Creates HR-pQCT full, trabecular, cortical, binary, and material labelmaps. | [Segmentation and Contours](docs/tools/segmentation-and-contours.md) |
| Scanco I/O | `Bone Imaging > I/O` | Imports and exports Scanco AIM images, masks, and metadata. | [Scanco I/O](docs/tools/scanco-io.md) |

The Slicer modules are wrappers around core Python packages where possible:

- `TimelapsedHRpQCT`: https://github.com/wallematthias/TimelapsedHRpQCT
- `MotionScoreHRpQCT`: https://github.com/wallematthias/MotionScoreHRpQCT

Each tool guide contains its own focused workflow instructions and attribution/citation notes.

## Runtime Dependencies

Some tools need Slicer-side dependencies in addition to this toolbox:

- **Motion Scoring** requires the `PyTorch` extension from Slicer's Extension Manager. Install `PyTorch`, restart Slicer, then run Motion Scoring.
- **Timelapsed HR-pQCT** installs/updates the `timelapsed-hrpqct` Python runtime from inside the module.
- **Scanco I/O** installs/updates the lightweight `aimio-py` / `py_aimio` AIM reader-writer stack from inside the module.

## Installation

### Extension Manager

When the toolbox is listed for your Slicer version:

1. Open 3D Slicer.
2. Install `Bone Imaging Toolbox` from Extension Manager.
3. Restart 3D Slicer.
4. Open modules from the `Bone Imaging` category.

### Manual Install From A Clone

Until the toolbox is available for your Slicer version, clone this repository and add the module folders to Slicer:

```bash
git clone https://github.com/wallematthias/SlicerBoneImagingToolbox.git
```

Then in Slicer:

1. Open `View -> Python Interactor`.
2. Run the helper script below, replacing `<repo>` with the folder containing this README:

```python
script = "<repo>/scripts/link_local_toolbox_modules.py"
exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
```

3. Restart Slicer.

Example macOS path:

```python
script = "/Users/<you>/Documents/14_GitHub/active/SlicerBoneImagingToolbox/scripts/link_local_toolbox_modules.py"
exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
```

### Manual Slicer Settings Alternative

Instead of using the helper script, add the module folders in `Edit -> Application Settings -> Modules -> Additional module paths`, then restart Slicer:

- `<repo>/HRpQCTTools/TimelapsedHRpQCT`
- `<repo>/HRpQCTTools/MotionScoreHRpQCT`
- `<repo>/HRpQCTTools/SegmentationHRpQCT`
- `<repo>/IOTools/ScancoIO`

Do not add only the top-level repository folder. Slicer needs each module folder above. The helper script also discovers any vendored scripted modules under `ExternalModules/`.

## Tool Documentation

- [Timelapsed HR-pQCT](docs/tools/timelapsed-hrpqct.md): longitudinal HR-pQCT analysis, input naming, results layout, remodelling review, and citations.
- [Motion Scoring](docs/tools/motion-scoring.md): PyTorch setup, model bundle setup, prediction/review workflow, and motion-grading citation.
- [Segmentation and Contours](docs/tools/segmentation-and-contours.md): segmentation presets, Laplace-Hamming notes, mask utilities, and attribution.
- [Scanco I/O](docs/tools/scanco-io.md): AIM import/export, metadata handling, and attribution.

## Adding External Modules

External modules can be checked into `ExternalModules/` as maintained forks, git subtrees, or git submodules. A direct scripted module folder must contain `CMakeLists.txt` and a same-named `.py` file, for example:

```text
ExternalModules/
  SlicerParOSol/
    ParOSolFEA/
      CMakeLists.txt
      ParOSolFEA.py
```

The top-level `CMakeLists.txt` discovers these folders during ExtensionIndex builds. The local `scripts/link_local_toolbox_modules.py` helper also adds them to Slicer's `Modules/AdditionalPaths`.

For a module to feel native inside the toolbox, set its Slicer category to a dot-separated submenu such as `Bone Imaging.FEA`, `Bone Imaging.I/O`, or `Bone Imaging.HR-pQCT` in the vendored fork.

## Repository Layout

```text
HRpQCTTools/
  TimelapsedHRpQCT/
  MotionScoreHRpQCT/
  SegmentationHRpQCT/
IOTools/
  ScancoIO/
ExternalModules/
resources/
docs/tools/
```

## License

This extension is distributed under the MIT License. Core pipeline packages installed from PyPI or separate repositories are governed by their own license terms.
