# Spine Segmentation

`Spine Segmentation` runs the `spine-segment` PyTorch workflow on a clinical CT volume and loads the resulting labelmaps into Slicer.

Use this tool when you have a spine CT volume and want to create:

- vertebral centroid markers,
- vertebral-level labels using the VerSe label convention,
- posterior process versus vertebral body labels,
- cortical versus trabecular compartment labels.

## Requirements

- Install Slicer's `PyTorch` extension from Extension Manager, then restart Slicer.
- Open `Bone Imaging > CT > Spine Segmentation`.
- Click `Install / Update Spine Segmentation` to install the `spine-segment` Python package into Slicer Python.

The first run may download the `spine-segment` model bundle into the user cache. Later runs reuse that cached bundle.

## Basic Workflow

1. Select the input CT scalar volume.
2. Choose the PyTorch device. `Auto` uses CUDA when available, then Apple MPS, then CPU.
3. Choose the run mode:
   - `Localization only` writes centroids and loads Slicer fiducial markers.
   - `Vertebral levels only` writes vertebral-level labels and centroids.
   - `Full` writes vertebral-level, process/body, cortical/trabecular, and centroid outputs.
4. Select an output folder.
5. Click `Run Spine Segmentation`.

The module exports the selected CT to NIfTI, runs `python -m spine_segment.cli`, and loads the generated centroid markers and/or NIfTI labelmaps for the selected run mode.

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
