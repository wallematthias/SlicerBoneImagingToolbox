# Batch Processor

The Batch Processor is the main cohort-processing interface. It discovers normalized datasets, shows tool-specific prerequisites, queues jobs, loads completed outputs, and keeps generated files inside `derivatives/`.

Use it after the Dataset Naming Helper has normalized a dataset.

## Workflow

1. Select the dataset root.
2. Select a tool.
3. Select a profile.
4. Review the discovered rows.
5. Fix missing inputs in the source dataset or by running an upstream tool.
6. Click a row `Run` button or click `Run all`.
7. Completed rows switch to `Load` when output artifacts are found.

Rows are tool-specific. Single-session tools run one session per row. Longitudinal or registered tools group the required sessions for one subject and VOI.

## Required Inputs By Tool

| Tool | Typical required inputs | Typical outputs |
| --- | --- | --- |
| Contouring | XCT image | bone segmentation, full/trab/cort masks, material label map |
| Timelapsed Remodelling | XCT images, registration ROI, bone segmentation, analysis ROIs | transforms, common region, remodelling maps, comparison table |
| Microarchitecture | XCT/BMD image, bone segmentation, analysis ROIs | scalar maps, measurement table |
| Registered Microarchitecture profile | Microarchitecture inputs plus common region | common-region-restricted measurement table |
| Plate/Rod Morphometry | bone segmentation and trabecular ROI | plate/rod maps and summary table |
| Registered Plate/Rod profile | Plate/rod inputs plus common region | common-region-restricted summary table |
| ParOsol-FEA | material label map | SED field, mechanics table |
| Mechanoregulation | remodelling map and matching SED field | mechanoregulation table and curve figures |
| Spine Segmentation | CT image | vertebral segmentation outputs |

The table should show only inputs required by the selected tool/profile. A row with missing required inputs should not run.

## Queue Behavior

- `Run` queues one row.
- `Run all` queues all runnable rows.
- Running rows can be cancelled.
- `Skip existing` reuses compatible outputs when they already exist.
- Switching tools or profiles should not change jobs already queued.

## Outputs

Outputs are written as derivative artifacts with manifest records. Loaded outputs should appear in Slicer with readable names, compact result tables, and predictable display settings.

## Screenshot To Add

Add one screenshot of a discovered table with generic `sub-001`, `ses-001`, and `voi-radiusleft` rows.

## Citation

Cite the analysis workflow selected in the Batch Processor. For Timelapsed remodelling and mechanoregulation, cite [Walle et al., Bone 2023](https://doi.org/10.1016/j.bone.2023.116780). For multistack registration, cite [Whittier et al., Bone 2023](https://doi.org/10.1016/j.bone.2023.116893).
