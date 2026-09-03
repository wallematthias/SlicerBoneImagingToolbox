# Derivative Workflow Contract

SlicerBoneImagingToolbox modules share reusable derivative products through manifest files. A module should read existing derivatives before recomputing them and should report missing prerequisites clearly in scene and batch modes.

## Modes

Scene mode uses loaded Slicer nodes as inputs and creates Slicer nodes, tables, transforms, and models as outputs. It is intended for interactive review and single-case work.

Batch mode uses dataset folders and derivative manifests as inputs, then writes derivative folders, files, manifests, and CSV outputs. It is intended for cohort processing and reproducible reruns.

Both modes should call the same backend services. The mode only changes input and output adapters.

## Derivatives

- `ImportedContours`: Scanco/IPL masks imported with a dataset, such as full, trabecular, cortical, and registration masks. These are preferred over generated masks when both are available.
- `BoneContours`: toolbox-generated bone segmentations, periosteal/endosteal masks, trabecular/cortical masks, generic ROI masks, and FEA material label maps.
- `Registration`: pairwise and composed transforms for longitudinal scans.
- `CommonRegion`: scan/FOV common-region masks derived from registered scan support.
- `Microarchitecture`: native maps, native measurements, and common-region-restricted measurements.
- `PlateRodMorphometry`: native plate/rod maps and native or common-region-restricted summaries.
- `Timelapse`: remodelling maps, longitudinal change tables, and remodelling review outputs.
- `FEA`: material maps, solver outputs, SED fields, load-history outputs, and mechanical summaries.
- `Mechanoregulation`: combined remodelling and mechanical-field outputs.
- `VoidSpace`: future void-space masks, maps, and measurements.

## Common Region

`CommonRegion` represents overlapping scan/FOV support only. It does not intersect bone, trabecular, cortical, marrow, void, or other biological masks.

Key roles:

- `scan_region_common`: common-region mask in the reference image space.
- `scan_region_native_common`: common-region mask transformed back to each native session image space.

Analysis modules apply the native common region locally:

```python
analysis_mask = biological_mask & native_common_scan_region
```

## Manifest

Each derivative writes `manifest.json` with records describing the produced files:

```json
{
  "workflow": "CommonRegion",
  "version": "1",
  "dataset_root": "/path/to/dataset",
  "records": [
    {
      "derivative": "CommonRegion",
      "role": "scan_region_native_common",
      "subject_id": "001",
      "site": "tibialeft",
      "session_id": "1",
      "stack_index": null,
      "space": "native",
      "path": "sub-001/ses-001/xct/sub-001_ses-001_voi-tibialeft_desc-scan-region-native-common_mask.nii.gz",
      "source": "generated",
      "metadata": {
        "reference_session_id": "1"
      }
    }
  ]
}
```

Consumers should prefer manifest records over filename inference. Batch tools should operate on normalized dataset names and portable relative paths so copied dataset roots remain usable.
