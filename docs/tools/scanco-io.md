# Scanco I/O

`Scanco I/O` imports Scanco AIM, ISQ, SCV, and GOBJ images in Slicer without requiring the full timelapsed HR-pQCT pipeline. It can also export edited grayscale or mask volumes back to AIM.

## When To Use

Use this tool when you want to:

- import `.AIM`, `.ISQ`, `.SCV`, and `.GOBJ` files into Slicer,
- load calibrated image data as density, native Scanco values, or HU,
- load nonzero image data as a Slicer segmentation,
- create a transform node from image geometry,
- edit or inspect volumes using standard Slicer tools,
- export scalar volumes, labelmaps, or segmentations back to AIM,
- preserve and edit AIM metadata during round trips.

## Setup

1. Open `Bone Imaging > Setup > Toolbox Setup`.
2. Install or update the Scanco I/O runtime if needed.
3. Open `Bone Imaging > I/O > Scanco I/O`.

This installs the lightweight Scanco image dependency stack, including `aimio-py` / `py_aimio`, without requiring the full Timelapsed pipeline.

## Import Workflow

1. Select the AIM, ISQ, SCV, or GOBJ file.
2. Choose the import scaling:
   - density,
   - native Scanco values,
   - HU.
3. Choose the Slicer target:
   - scalar volume,
   - segmentation from nonzero voxels,
   - transform from image geometry.
4. Click import.
5. Inspect or edit the loaded node in Slicer.

Imported metadata is stored on the loaded Slicer node.

Drag and drop exposes explicit Scanco reader choices in Slicer's Add Data dialog:

- `ScancoVolume`: reads native values into a scalar volume,
- `ScancoHU`: reads HU values into a scalar volume,
- `ScancoDensity`: reads density values into a scalar volume,
- `ScancoSegmentation`: reads native values into a Slicer segmentation.

## Export Workflow

1. Select a scalar volume, labelmap, or Slicer segmentation.
2. Review the processing-log field table.
3. Review or edit header metadata JSON.
4. Choose the export mode.
5. Export to AIM.

For exports from volumes that did not originate from Scanco I/O, provide an imported-stack metadata JSON from the timelapsed pipeline, paste/edit header JSON manually, or explicitly enable minimal metadata export.

Geometry fields that can be read from the selected Slicer volume, such as dimensions, spacing, origin, and direction, are refreshed at export time.

## Attribution

AIM, ISQ, SCV, and GOBJ import plus AIM export are backed by the `aimio-py` / `py_aimio` package.

No separate method paper is currently specified for this module. Cite the toolbox and any acquisition, segmentation, or downstream analysis methods used with the exported data.
