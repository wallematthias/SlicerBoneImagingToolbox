# Derivative Workflow Contract

SlicerBoneImagingToolbox modules share reusable derivative products through manifest files. A module should read existing derivatives before recomputing them, and long-running tools should generate missing prerequisites when inputs are available and dependency generation is enabled.

## Modes

Scene mode uses loaded Slicer nodes as inputs and creates Slicer nodes, tables, transforms, and models as outputs. It is intended for interactive review and single-case work.

Batch mode uses dataset folders and derivative manifests as inputs, then writes derivative folders, files, manifests, and CSV outputs. It is intended for cohort processing and reproducible reruns.

Both modes should call the same backend services. The mode only changes input and output adapters.

## Derivatives

- `Registration`: pairwise and composed transforms for longitudinal scans.
- `CommonRegion`: scan/FOV common-region masks derived from registered scan support.
- `Segmentation`: bone, full/periosteal, trabecular, cortical, and future mask roles.
- `Microarchitecture`: measurement tables and scalar map outputs.
- `PlateRodMorphometry`: plate/rod labels, element maps, and summaries.
- `Timelapsed`: remodelling and longitudinal change outputs.
- `FEA`: meshes, material maps, boundary conditions, solver outputs, and mechanical fields.
- `Mechanoregulation`: combined biological change and mechanical-field outputs.
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
      "subject_id": "SAMPLE001",
      "site": "tibia",
      "session_id": "1",
      "stack_index": 1,
      "space": "native",
      "path": "sub-SAMPLE001/site-tibia/stack-01/native_space/ses-1/masks/mask.nii.gz",
      "source": "generated",
      "metadata": {
        "reference_session_id": "1"
      }
    }
  ]
}
```

Consumers should prefer manifest records over filename inference. Filename discovery remains a compatibility fallback for older outputs.
