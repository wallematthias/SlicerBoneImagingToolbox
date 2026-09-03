# Timelapsed Remodelling

Timelapsed Remodelling is the Slicer front end for longitudinal HR-pQCT registration, common-region generation, and remodelling analysis. The Slicer module supports interactive scene review; cohort runs are launched from the Batch Processor.

Core processing lives in the `timelapsed-hrpqct` Python package:

https://github.com/wallematthias/TimelapsedHRpQCT

## When To Use

Use this workflow when you have two or more longitudinal scans from the same subject and VOI and want to:

- register timepoints,
- reuse or create common scan regions,
- classify formation, resorption, quiescent bone, and background,
- review remodelling maps interactively in Slicer,
- export pairwise remodelling metrics.

Prepare registration ROIs, bone segmentations, and analysis ROIs before running Timelapsed Remodelling. Generated contours belong to Contouring; imported scanner/IPL contours belong to `ImportedContours`.

## Required Inputs

| Input | Meaning |
| --- | --- |
| XCT image per session | source scan for each longitudinal timepoint |
| Registration ROI | mask used for image registration; commonly full bone contour |
| Bone segmentation | binary bone support used for remodelling classification |
| Analysis ROI masks | one or more ROIs used for reporting, commonly full, trabecular, and cortical |
| Initial transforms | optional user-provided transforms |

## Scene Mode

Use scene mode when the scans are already loaded in Slicer.

1. Discover or add loaded timepoints.
2. Select image nodes for each session.
3. Map the registration ROI, bone segmentation, and analysis ROIs.
4. Select a profile.
5. Adjust analysis options if needed.
6. Run.
7. Review the loaded remodelling map and current comparison table.
8. Use `Apply settings` to update loaded remodelling outputs interactively.

Scene mode keeps selected source scans native in the scene and loads remodelling results for review.

## Batch Mode

Use `Bone Imaging > I/O > Batch Processor` for cohort processing.

1. Select a normalized dataset root.
2. Select `Timelapsed Remodelling`.
3. Select the desired profile.
4. Review the discovered grouped rows.
5. Run one row or `Run all`.
6. Load completed rows to bring remodelling maps and compact result tables into Slicer.

Profiles include `standard`, `eth-uofc`, `ucsf`, `ped-fx`, `multistack`, `xct1-standard`, and `shriners`.

## Outputs

Timelapsed outputs are written under `derivatives/`:

```text
derivatives/
  Registration/
  CommonRegion/
  Timelapse/
```

The remodelling table loaded in Slicer focuses on the main review metrics: comparison pair, ROI, `FV/BV`, `RV/BV`, `AV/BV`, `NV/BV`, and profile.

## Input Naming

Normalized datasets should use the toolbox layout:

```text
sub-001/ses-001/xct/sub-001_ses-001_voi-radiusleft_xct.AIM
sub-001/ses-002/xct/sub-001_ses-002_voi-radiusleft_xct.AIM
```

Older lab-style filenames can be normalized with Dataset Naming Helper. Generic examples:

```text
sample001_DR_T1.AIM
sample001_DR_T1_TRAB_MASK.AIM
sample001_DR_T2.AIM

sample010_DT_STACK01_T1.AIM
sample010_DT_STACK02_T1_CORT_MASK.AIM

sample020_KN_BL.AIM
sample020_KN_FL1_REGMASK.AIM
```

## Screenshot To Add

Add two generic screenshots:

- scene role mapping with three timepoints,
- loaded remodelling map plus current comparison table.

## Citation

For Timelapsed remodelling, cite:

Walle M, Whittier DE, Schenk D, Atkins PR, Blauth M, Zysset P, Lippuner K, Müller R, Collins CJ. Precision of bone mechanoregulation assessment in humans using longitudinal high-resolution peripheral quantitative computed tomography in vivo. *Bone*. 2023;172:116780. doi: [10.1016/j.bone.2023.116780](https://doi.org/10.1016/j.bone.2023.116780).

For multistack registration, cite:

Whittier DE, Walle M, Schenk D, Atkins PR, Collins CJ, Zysset P, Lippuner K, Müller R. A multi-stack registration technique to improve measurement accuracy and precision across longitudinal HR-pQCT scans. *Bone*. 2023;176:116893. doi: [10.1016/j.bone.2023.116893](https://doi.org/10.1016/j.bone.2023.116893).
