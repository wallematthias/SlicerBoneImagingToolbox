# Spine Segmentation

`Spine Segmentation` runs the `spine-segment` PyTorch workflow on a clinical CT volume and loads the resulting labelmaps into Slicer.

Use this tool when you have a spine CT volume and want to create:

- vertebral centroid markers,
- vertebral-level labels using the VerSe label convention,
- posterior process versus vertebral body labels,
- cortical versus trabecular compartment labels.

## Requirements

- Open `Bone Imaging > CT > Spine Segmentation`.
- The main panel can run immediately when a valid runtime is already available.
- For the simple Slicer-native runtime, open `Runtime setup`, install Slicer's `PyTorch` extension from Extension Manager, restart Slicer, then click `Install Slicer Runtime`.
- For faster Apple Silicon inference, open `Runtime setup`, click `Install Conda MPS Runtime`, or point `Conda Python` to an existing arm64 environment with `torch` and `spine-segment` installed.

The first run may download the `spine-segment` model bundle into the user cache. Later runs reuse that cached bundle.

## Basic Workflow

1. Select the input CT scalar volume.
2. Choose the output set:
   - `Full segmentation + centroids` writes vertebral-level, process/body, cortical/trabecular, and centroid outputs.
   - `Vertebral levels + centroids` writes vertebral-level labels and centroid markers.
   - `Centroids only` writes centroid markers.
3. Select an output folder.
4. Click `Run`.

The module exports the selected CT to NIfTI, runs `python -m spine_segment.cli` in the selected runtime, and loads the generated centroid markers and/or NIfTI labelmaps for the selected run mode.

Centroid markers are loaded for every completed run and are named with anatomical VerSe levels such as `T12`, `L1`, and `L2`. Body/process and cortical/trabecular segmentations are generated together in full segmentation mode.

## Runtime Notes

The `Conda MPS Runtime` option is intentionally separate from Slicer Python. Current macOS Slicer builds can run under x86_64/Rosetta, while a Miniforge environment can be native arm64. The module probes the selected conda Python for `spine-segment`, PyTorch, MPS availability, and actual `Conv3D` support before using it.

This mirrors the reliable part of the nnUNet Slicer extension design: Slicer exports an input file, launches a background process, then loads the output files back into the scene. The difference is that this module can launch an external conda Python instead of a script installed inside Slicer's Python folder.

Runtime details live in the collapsed `Runtime setup` section. `Auto` probes the conda runtime first and uses it when `spine-segment` is installed and PyTorch supports 3D convolutions on MPS; otherwise it falls back to Slicer Python. The PyTorch device should usually stay on `Auto`.

## Outputs

For an input named `scan.nii.gz`, the output folder receives:

| File | Loaded Slicer node |
| --- | --- |
| `scan_vertebral-level.nii.gz` | `scan Vertebral levels` |
| `scan_process-body.nii.gz` | `scan Process/body` |
| `scan_cort-trab.nii.gz` | `scan Cortical/trabecular` |
| `scan_centroids.json` | `scan Vertebral centroids` markups |

## Attribution

This module is a Slicer wrapper around:

- `spine-segment`: https://github.com/wallematthias/spine-segment

For vertebral localization, identification, and level segmentation, cite:

```text
Payer C, Stern D, Bischof H, Urschler M.
Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net.
In: Proceedings of the 15th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2020), Volume 5: VISAPP. 2020;124-133.
doi:10.5220/0008975201240133
```

For the process/body compartment workflow, cite:

```text
Walle M, Matheson BE, Boyd SK.
Comparing linear and nonlinear finite element models of vertebral strength across the thoracolumbar spine: a benchmark from density-calibrated computed tomography.
GigaScience. 2025;14:giaf094.
doi:10.1093/gigascience/giaf094
```
