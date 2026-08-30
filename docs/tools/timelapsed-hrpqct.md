# Timelapsed HR-pQCT

`Timelapsed HR-pQCT` is the Slicer front end for longitudinal HR-pQCT workflows. It organizes scans into subject/site/session structure, runs timelapsed processing, and loads remodelling outputs for review.

## When To Use

Use this tool when you have longitudinal Scanco HR-pQCT AIM datasets and want to:

- parse raw AIM files into a cohort layout,
- generate masks when needed,
- register longitudinal scans,
- run remodelling analysis,
- review raw, transformed, and remodelling-image outputs in Slicer,
- export cohort-level remodelling summary tables.

## Setup

1. Open `Bone Imaging > HR-pQCT > Timelapsed HR-pQCT`.
2. Click `Install / Update timelapsed-hrpqct`.
3. Restart Slicer if the install process asks for it.

The Slicer module is the GUI layer. Core processing lives in the `timelapsed-hrpqct` Python package and repository:

https://github.com/wallematthias/TimelapsedHRpQCT

## Basic Workflow

The module has two modes:

- `Scene` uses volumes and masks already loaded in Slicer for a small explicit longitudinal series.
- `Batch` discovers scans from a dataset folder and writes reproducible cohort outputs.

## Scene Mode

Use `Scene` when the timepoints are already loaded in Slicer. Add one row per session, select the image node and any available masks, choose a results folder, then run the scene pipeline. The module exports those selected nodes into a scoped scene-run folder and launches the same Timelapsed workflow in the background.

## Batch Mode

1. Select the AIM dataset root.
2. Click `Parse input`.
3. Review the parse table and correct site/session values if needed.
4. Set `Results folder` if you do not want the default `<dataset_root>/TimelapsedHRpQCT`.
5. Select a study `Profile`.
6. Click `Apply profile`.
7. Leave mask generation enabled unless you already have valid masks/contours.
8. Click `Run pipeline`.
9. Use `Load Processed Data` to load `raw`, `transformed`, or `remodelling image` outputs.
10. Inspect loaded volumes in 2D/3D.

Most controls include hover tooltips. These are the first-line reference for profile effects, threshold units, and review-state behavior.

## Interactive Remodelling Review

After loading a saved remodelling image, the module exposes the key review parameters used by the pipeline:

- threshold,
- cluster size,
- analysis method,
- pair mode,
- full-mask dilation,
- marrow-mask erosion,
- Gaussian filtering,
- Gaussian sigma.

By default, loaded remodelling images update only when `Update remodelling image` or `Apply profile` is clicked. When auto update is enabled, changing these controls recomputes the loaded preview from the transformed image pair already on disk. This allows exploratory threshold/filter changes without rerunning the full command-line analysis.

Use `Rerun cohort analysis` when you want to recompute saved outputs for the whole processed cohort.

## Results Layout

Pipeline outputs are saved in a structured folder layout under the results root.

- Default results root: `<dataset_root>/TimelapsedHRpQCT`
- Optional override: `Results folder`
- Typical organization: subject/site/session folders with derivative outputs grouped by processing stage

This layout is intended to make results easy to browse, reload in Slicer, and reuse in downstream analysis.

## Input Filename Format

The parser expects AIM filenames that include:

- subject identifier,
- site token such as `DR`, `DT`, or `KN`,
- session token such as `T1`, `T2`, `C1`, `BL`, `FL`, or `FL1`,
- optional stack token for multistack data, such as `STACK01`, `STACK_01`, or `STACK-01`,
- optional mask roles such as `TRAB_MASK`, `CORT_MASK`, `FULL_MASK`, `REGMASK`, `ROI1`, or `ROI2`.

Examples:

```text
SUBJ001_DR_T1.AIM
SUBJ001_DR_T1_TRAB_MASK.AIM
SUBJ001_DR_T2.AIM

SUBJ010_DT_STACK01_T1.AIM
SUBJ010_DT_STACK02_T1_CORT_MASK.AIM

SAMPLE355_KN_BL.AIM
SAMPLE355_KN_FL1_REGMASK.AIM
```

If filename parsing is incomplete or ambiguous, the parser can fall back to AIM header metadata when available.

## Attribution

For the main `eth-uofc` timelapsed HR-pQCT method, cite:

Walle M, Whittier DE, Schenk D, Atkins PR, Blauth M, Zysset P, Lippuner K, Müller R, Collins CJ. Precision of bone mechanoregulation assessment in humans using longitudinal high-resolution peripheral quantitative computed tomography in vivo. *Bone*. 2023;172:116780. doi: 10.1016/j.bone.2023.116780.

For multistack registration, cite:

Whittier DE, Walle M, Schenk D, Atkins PR, Collins CJ, Zysset P, Lippuner K, Müller R. A multi-stack registration technique to improve measurement accuracy and precision across longitudinal HR-pQCT scans. *Bone*. 2023;176:116893. doi: 10.1016/j.bone.2023.116893.

Related applications using the method include:

- Walle M, Duseja A, Whittier DE, Vilaca T, Paggiosi M, Eastell R, Müller R, Collins CJ. Bone remodeling and responsiveness to mechanical stimuli in individuals with type 1 diabetes mellitus. *Journal of Bone and Mineral Research*. 2024;39(2):85-94.
- Walle M, Gabel L, Whittier DE, Liphardt AM, Hulme PA, Heer M, Zwart SR, Smith SM, Sibonga JD, Boyd SK. Tracking of spaceflight-induced bone remodeling reveals a limited time frame for recovery of resorption sites in humans.
- Matheson BE, Walle M, Bugbird AR, Rosenberg M, Mateus J, Boyd SK. Early skeletal deteriorations following short-duration spaceflight.
