# Scanco I/O

`Scanco I/O` imports and exports Scanco AIM images in Slicer without requiring the full timelapsed HR-pQCT pipeline.

## When To Use

Use this tool when you want to:

- import `.AIM` images into Slicer,
- load AIM data as density/BMD, native Scanco values, mu, HU, or segmentation masks,
- edit or inspect volumes using standard Slicer tools,
- export scalar volumes, labelmaps, or segmentations back to AIM,
- preserve and edit AIM metadata during round trips.

## Setup

1. Open `Bone Imaging > I/O > Scanco I/O`.
2. Click `Install / Update AIM I/O`.

This installs only the lightweight AIM dependency stack, including `aimio-py` / `py_aimio`; it does not install the full `timelapsed-hrpqct` pipeline.

## Import Workflow

1. Select the AIM file.
2. Choose the import scaling:
   - density/BMD,
   - native Scanco values,
   - mu,
   - HU,
   - segmentation from nonzero voxels.
3. Click import.
4. Inspect or edit the loaded volume in Slicer.

Imported AIM metadata is stored on the loaded Slicer volume.

## Export Workflow

1. Select a scalar volume, labelmap, or Slicer segmentation.
2. Review the processing-log field table.
3. Review or edit header metadata JSON.
4. Choose the export mode.
5. Export to AIM.

For exports from volumes that did not originate from Scanco I/O, provide an imported-stack metadata JSON from the timelapsed pipeline, paste/edit header JSON manually, or explicitly enable minimal metadata export.

Geometry fields that can be read from the selected Slicer volume, such as dimensions, spacing, origin, and direction, are refreshed at export time.

## Attribution

AIM import/export is backed by the `aimio-py` / `py_aimio` package.

No separate method paper is currently specified for this module. Cite the toolbox and any acquisition, segmentation, or downstream analysis methods used with the exported data.
