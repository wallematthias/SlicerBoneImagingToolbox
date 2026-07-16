# Segmentation and Contours

`Segmentation and Contours` creates HR-pQCT masks, contours, and labelmaps for interactive preparation and QA.

## When To Use

Use this tool when you need to:

- create full, trabecular, cortical, and binary HR-pQCT masks,
- generate contour-supported segmentations from radius, tibia, or knee presets,
- prepare labelmaps before downstream FEA or analysis,
- derive missing compartment masks,
- validate full/trabecular/cortical consistency,
- create HOM/material labelmaps.

## Basic Workflow

1. Open `Bone Imaging > HR-pQCT > Segmentation and Contours`.
2. Select an input HR-pQCT volume.
3. Select the site preset.
4. Choose a bone segmentation method:
   - Gaussian,
   - Laplace-Hamming,
   - adaptive,
   - none.
5. Choose periosteal/outer and endosteal/inner contour strategies.
6. Adjust expert thresholds and morphology settings only when needed.
7. Click `Generate Masks And Segmentation`.
8. Optionally open Slicer's Segment Editor for manual cleanup.

Standard contours follow the selected bone segmentation method as their support. The local geodesic fracture contour can be selected as the periosteal contour for radius fracture cases.

## Laplace-Hamming Notes

Laplace-Hamming segmentation follows the same native Scanco-unit convention as the core pipeline. When the selected Slicer volume came from Scanco I/O, the module uses attached AIM calibration metadata to convert density images back to native Scanco values. Otherwise it can reload the original AIM source when that path is available.

If Laplace-Hamming produces an empty segmentation, check that:

- the input came from Scanco I/O or has AIM calibration metadata,
- the original AIM source is available,
- the threshold is in native Scanco attenuation units.

## Derive Labels Tab

The `Derive Labels` tab provides common mask utilities:

- generate a missing compartment mask from any two of `full`, `trab`, and `cort`,
- create HOM/material labelmaps from bone segmentation plus any two compartment masks,
- use default material labels `126` for trabecular bone and `127` for cortical bone,
- run boolean mask operations: union, intersection, A-minus-B, and XOR,
- relabel nonzero voxels in a mask,
- validate full/trab/cort consistency and report voxel counts.

## Attribution

Laplace-Hamming segmentation follows the Galateia Kazakia lab implementation and related work:

https://github.com/gkazakia

When publishing results from a specific segmentation method or downstream analysis, cite the method paper for that selected method in addition to the toolbox/tool documentation.
