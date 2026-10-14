# Scanco I/O

`Scanco I/O` imports Scanco AIM, ISQ, SCV, GOBJ, and RAD images in Slicer without requiring the full timelapsed HR-pQCT pipeline. It can also export edited grayscale or mask volumes back to AIM.

## When To Use

Use this tool when you want to:

- import `.AIM`, `.ISQ`, `.SCV`, `.GOBJ`, and `.RAD` files into Slicer,
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
Scanco I/O requires `aimio-py>=0.4.1` for RAD viewing and repaired SCV/RSQ reading.
After updating the toolbox and runtime, restart Slicer.

## Import Workflow

1. Select the AIM, ISQ, SCV, GOBJ, or RAD file.
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

SCV scouts with repeated or uneven row coordinates load for viewing in pixel
space. Their names end in `[pixel space]`, and their metadata retains a geometry
warning and the original row coordinates. Slicer's rulers still display mm,
but those values represent pixels for these nodes: do not use them for physical
measurements or registration. Other supported SCV scouts retain the historical
IPL-compatible geometry.

Drag and drop exposes explicit Scanco reader choices in Slicer's Add Data dialog:

- `ScancoVolume`: reads native values into a scalar volume,
- `ScancoHU`: reads HU values into a scalar volume,
- `ScancoDensity`: reads density values into a scalar volume,
- `ScancoSegmentation`: reads native values into a Slicer segmentation.

### RAD radiographs

Select the `.RAD` file in Scanco I/O, or drag it into Slicer and choose
`ScancoVolume`. RAD loads as a native signed-int16, single-slice scalar volume
with header-derived mm spacing. The module automatically selects Native and
Scalar volume; HU/density conversion, segmentation import, and geometry-transform
import are unavailable for RAD. This requires an `aimio-py` version exposing
`read_rad`; older runtimes give an explicit update message.

RAD is a 2D projection, not a reconstructed CT volume. Its third-axis spacing
is a display placeholder. Zero origin and identity direction do not register it
to the CT scan; scanner reference positions remain in the node metadata. Header
pixel spacing is not a guarantee of anatomical object-plane scale for every
acquisition geometry. Do not use it as a calibrated bone-density volume.

## Export Workflow

1. Select a scalar volume, labelmap, or Slicer segmentation.
2. Review the processing-log field table.
3. Review or edit header metadata JSON.
4. Choose the export mode.
5. Export to AIM.

For exports from volumes that did not originate from Scanco I/O, provide an imported-stack metadata JSON from the timelapsed pipeline, paste/edit header JSON manually, or explicitly enable minimal metadata export.

Geometry fields that can be read from the selected Slicer volume, such as dimensions, spacing, origin, and direction, are refreshed at export time.

## Attribution

AIM, ISQ, SCV, GOBJ, and RAD import plus AIM export are backed by the `aimio-py` / `py_aimio` package. RAD header interpretation follows
[David Gobbi's vtk-dicom Scanco reader](https://github.com/dgobbi/vtk-dicom/blob/master/Source/vtkScancoCTReader.cxx).

No separate method paper is currently specified for this module. Cite the toolbox and any acquisition, segmentation, or downstream analysis methods used with the exported data.
