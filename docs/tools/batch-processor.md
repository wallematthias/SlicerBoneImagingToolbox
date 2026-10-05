# Batch Processor

The Batch Processor is the main cohort-processing interface. It discovers normalized datasets, shows tool-specific prerequisites, queues jobs, loads completed outputs, and keeps generated files inside `derivatives/`.

!!! video "Tutorial video"
    Watch the [Batch Processor tutorial](https://www.youtube.com/watch?v=KMDTtJk_x0s), or browse [all tutorial videos](../tutorials/index.md).

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

If the dataset is not yet normalized, run the Dataset Naming Helper first. The Batch Processor is deliberately stricter than the interactive scene modules because cohort processing depends on stable subject, session, VOI, stack, and derivative paths.

Each tool-specific profile defines the command, required inputs, and expected outputs for the selected tool. Shipped profiles provide scanner or workflow defaults. Some tools also discover user-exported custom profiles.

For published CNN compartments, select **Bone Contouring → U-Net contours (Neeteson et al.) + LH SEG**
and a device. Standard/custom contour profiles remain in the same list, with
automatic per-row site detection. U-Net uses fixed published defaults and writes
full/trab/cort masks plus XCTII Laplace–Hamming tissue SEG and material labels in
one run. Completed U-Net compartments are reused when only SEG/material is
missing, without running the network or replacing those files. Load combines
full/trab/cort/SEG in one node. Scene-only U-Net and tissue-only recipes
are not offered as standard batch profiles.

## Required Inputs By Tool

| Tool | Typical required inputs | Typical outputs |
| --- | --- | --- |
| Contouring | XCT image | bone segmentation, full/trab/cort masks, material label map |
| Timelapsed Remodelling | XCT images, registration ROI, bone segmentation, analysis ROIs | transforms, common region, remodelling maps, comparison table |
| Microarchitecture | XCT/BMD image, bone segmentation, analysis ROIs | scalar maps, measurement table |
| Registered Microarchitecture profile | Microarchitecture inputs plus common region | common-region-restricted measurement table |
| Native Functional Bone profile | Microarchitecture inputs plus native large-void map | full-scan measurements excluding large voidspace, analysis mask |
| Registered Functional Bone profile | Microarchitecture inputs plus common region and large-void map | common-region measurements excluding large voidspace, analysis mask |
| Voidspace | bone segmentation and optional mask | all-void and large-void masks, measurement table |
| Registered Voidspace profile | native segmentation, native full mask, native common region | common-region-restricted all-void and large-void masks, measurement table |
| Dynamic Voidspace profile | adjacent Timelapsed registered segmentations and common region | expanded, contracted, and quiescent voidspace masks, change table |
| Plate/Rod Morphometry | bone segmentation and trabecular ROI | plate/rod maps and summary table |
| Registered Plate/Rod profile | Plate/rod inputs plus common region | common-region-restricted summary table |
| ParOsol-FEA | material label map | SED field, mechanics table |
| Mechanoregulation | remodelling map and matching SED field | mechanoregulation table and curve figures |
| Spine Segmentation | CT image | vertebral segmentation outputs |

The table should show only inputs required by the selected tool/profile. A row with missing required inputs should not run.

Both Functional Bone profiles are under **Microarchitecture**. Native requires a native Voidspace run but no registration/common region. Registered requires a native common region and accepts either native or Registered voidspace output. Their measurement directories and loaded analysis regions are distinct, while native Microarchitecture maps are shared. Existing saved `functional-bone` jobs remain registered. See [Microarchitecture](microarchitecture.md) for details.

## Queue Behavior

- `Run` queues one row.
- `Run all` queues all runnable rows.
- Running rows can be cancelled.
- `Skip existing` reuses compatible outputs when they already exist.
- Switching tools or profiles should not change jobs already queued.

The queued jobs keep the tool and profile that were active when they were added to the queue. This keeps a long run stable even if the visible Batch Processor controls are refreshed later.

## Export to CSV

After running a measurement workflow, keep its **Tool** and **Profile** selected
and click **Export to CSV**. Choose a destination; the suggested location is
`derivatives/<workflow>/aggregated/`. The export collects existing summaries for
the displayed dataset rows without rerunning analysis or averaging measurements.

The result is a wide cohort table: one row per subject, session, site and stack,
or per baseline–follow-up pair for longitudinal measurements. Identifier columns
come first, followed by the union of measurement columns. Subject/session IDs
retain their original text in the CSV; when importing into Excel, set these
columns to **Text** to preserve leading zeros. Each row includes the selected
tool/profile and a `source_files` column listing its source CSVs.

Supported exports:

| Tool | Measurement columns |
| --- | --- |
| Microarchitecture / Functional Bone | Parameter and statistic, e.g. `Tb.Th.Mean`, `Tb.Th.SD`, `Tb.Th.Units` |
| Voidspace | Large-void summary, e.g. `Large.VS.V`; dynamic profiles preserve change columns such as `expanded.VS.V` |
| Timelapsed Remodelling | Compartment, threshold and cluster size, e.g. `trab.threshold-225.cluster-1.formation_vox` |
| Plate/Rod Morphometry | Existing scalar summary columns; individual element tables are excluded |
| ParOsol-FEA | Existing scalar mechanics summaries and numbered load components; detailed curves are excluded |
| Mechanoregulation | ROI-prefixed summary columns, e.g. `full.OR_F`; classification, threshold and cluster prefixes distinguish multiple remodelling variants |

Native, registered and Functional Bone profiles use their own measurement
outputs. Missing values and non-finite numbers become blank cells; zero remains
zero. Missing/unreadable summaries and conflicting measurements are reported in
the Batch Processor log, with affected cases omitted. The completion dialog
reports the number of exported rows and any issues. Source files are not changed.
If a source table explicitly identifies a different profile, it is reported and
omitted rather than silently relabelled as the selected profile.
Mechanoregulation also requires its companion summary JSON to verify the
profile; its figures and surface-event volumes are not required for export.
Current Voidspace CSVs describe **large voidspace only**; the export does not
invent an all-void measurement from the all-void mask.

For server workflows, the button downloads CSV summaries and small discovery/
profile metadata into a fresh temporary snapshot, not image volumes. A failed download
stops the export instead of falling back to old local results. The tool/profile
selected when export begins remains fixed even if you change the controls.

## Outputs

Outputs are written as derivative artifacts with manifest records. Loaded outputs should appear in Slicer with readable names, compact result tables, and predictable display settings.

Imported cropped masks retain their own origin, spacing, and direction when
loaded; the source volume's geometry must not overwrite their crop position.
Loading contours, U-Net compartments, Voidspace, common regions, or Functional
bone analysis masks marks them as batch-owned case overlays. Loading another
case hides earlier batch overlays, while registered timepoints of the same subject
and VOI remain visible together. Manually created segmentations are left alone.
Reloading a contour result replaces only the previously tagged batch result, even
if a manual segmentation has the same name. Legacy untagged overlays are not hidden
automatically; hide them manually or start a fresh scene for testing.

## Citation

Cite the analysis workflow selected in the Batch Processor. For Timelapsed remodelling and mechanoregulation, cite [Walle et al., Bone 2023](https://doi.org/10.1016/j.bone.2023.116780). For multistack registration, cite [Whittier et al., Bone 2023](https://doi.org/10.1016/j.bone.2023.116893). For voidspace profiles, see the [Voidspace](voidspace.md) citation guidance.
