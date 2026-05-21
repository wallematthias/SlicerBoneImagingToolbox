<p align="center">
  <img src="resources/TimelapsedHRpQCTSlicer.png" alt="TimelapsedHRpQCTSlicer logo" width="320">
</p>

# HR-pQCT Toolbox for 3D Slicer

3D Slicer scripted extension for HR-pQCT workflows: longitudinal timelapsed analysis, motion scoring, Scanco AIM import/export, and contouring/segmentation helpers.

`timelapsed-hrpqct` is a longitudinal high-resolution peripheral quantitative computed tomography (HR-pQCT) analysis workflow that aligns longitudinal scans of the same subject across timepoints and computes remodelling-related outputs from those registered volumes.
It is designed for high-throughput, pipeline-style processing of multi-subject datasets.
It is intended for HR-pQCT datasets acquired on Scanco XtremeCT systems. The current workflow was developed primarily for second-generation XtremeCT data, but can be adapted to other compatible acquisition setups.

The extension appears in Slicer as one toolbox category:

```text
HR-pQCT
  Timelapsed HR-pQCT
  Motion Scoring
  Scanco I/O
  Contours and Segmentation
```

## Core Repositories

The Slicer modules remain workflow wrappers around core Python packages:

- `TimelapsedHRpQCT`: https://github.com/wallematthias/TimelapsedHRpQCT
- `MotionScoreHRpQCT`: https://github.com/wallematthias/MotionScoreHRpQCT

<p align="center">
  <img src="resources/screenshot-slicer-TimelapsedHRpQCT.png" alt="TimelapsedHRpQCT module screenshot" width="1000">
</p>
<p align="center">
  <em>Example output: HR-pQCT scan of a knee (2 stacks), registered longitudinally, with formation sites in orange and resorption sites in purple. Data kindly provided by Dr. Sarah Manske.</em>
</p>

## Modules

- **Timelapsed HR-pQCT**: End-to-end longitudinal HR-pQCT workflow in Slicer.
  - Parses AIM datasets into subject/site/session structure.
  - Generates masks (if needed), runs timelapse registration, and computes remodelling outputs.
  - Loads processed outputs (`raw`, `transformed`, `remodelling image`) for review and 3D visualization.
- **Motion Scoring**: Interactive HR-pQCT scan motion grading workflow.
  - Runs MotionScore predictions.
  - Supports rapid reviewer grading and review-table export.
  - Keeps model inference and retraining logic in the MotionScore core package.
  - Supports local or downloaded model bundles without requiring a hosted license API.
- **Scanco I/O**: Focused AIM import/export utility.
  - Imports Scanco `.AIM` images as density/BMD, native Scanco values, mu, or HU.
  - Preserves AIM metadata on imported Slicer volumes so edited data can be exported without a reference AIM.
  - Exports edited grayscale volumes or binary masks back to `.AIM`.
  - Uses a lightweight local wrapper around `aimio-py` / `py_aimio`; it does not install the full `timelapsed-hrpqct` pipeline.
- **Contours and Segmentation**: Lightweight Slicer workflow helper for masks and contours.
  - Creates a threshold-based segmentation from an HR-pQCT volume.
  - Opens Slicer's Segment Editor for manual cleanup.
  - Leaves algorithmic segmentation methods in the core packages.

## Installation

### Option A: Extension Manager (recommended when listed)

1. Open 3D Slicer.
2. Install `HR-pQCT Toolbox` from Extension Manager.
3. Restart 3D Slicer.

### Option B: Developer mode (current fallback)

1. Open 3D Slicer.
2. Go to `View -> Python Interactor`.
3. Run:
   ```python
   script = "<repo>/TimelapsedHRpQCTSlicer/scripts/link_local_toolbox_modules.py"
   exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
   ```
4. Restart Slicer.
5. Open modules from the `HR-pQCT` category.

Manual alternative: go to `Edit -> Application Settings -> Modules` and add all module paths:

   - `<repo>/TimelapsedHRpQCTSlicer/TimelapsedHRpQCT`
   - `<repo>/TimelapsedHRpQCTSlicer/MotionScoreHRpQCT`
   - `<repo>/TimelapsedHRpQCTSlicer/ScancoIO`
   - `<repo>/TimelapsedHRpQCTSlicer/HRpQCTSegmentation`

## Tutorial

1. Open module `Timelapsed HR-pQCT`.
2. Click `Install / Update timelapsed-hrpqct` to install runtime dependencies in Slicer Python.
3. Select your AIM dataset root.
4. Click `Parse input`.
   - `Parse summary` reports how many sessions were discovered.
   - In `Parse details`, you can correct `site` and `session` values if needed before running.
5. Choose where results should be written:
   - default: `<dataset_root>/TimelapsedHRpQCT`
   - optional: set `Results folder` to override.
6. If you do not already have valid masks/contours, click `1. Generate Masks`.
7. Click `2. Timelapse Pipeline` to run the timelapse processing and create remodelling outputs.
8. Load `remodelling image` from `Load Processed Data` and inspect in 2D/3D.
9. If you change analysis settings (for example density threshold or cluster size), click `3. Re-run Analysis`.
10. Use `Load Processed Data` to load different processing stages (`raw`, `transformed`, `remodelling image`) for quick comparison.

## Interactive Remodelling Review

The Timelapsed module includes study profiles for ETH/UofC, UCSF, Shriners, and the core standard defaults.
Applying a profile updates the visible analysis controls and passes the selected core profile to new runs.
After loading a saved remodelling image, the module exposes the key review parameters used by the pipeline: threshold, cluster size, analysis method, pair mode, full-mask dilation, marrow-mask erosion, Gaussian filtering, and Gaussian sigma.
When auto update is enabled, changing these controls recomputes the loaded preview from the transformed image pair already on disk, so exploratory threshold/filter changes do not require rerunning the full command-line analysis.
The series summary panel can load saved cohort-level pairwise outputs and export the available cohort rows with subject/site, timepoint pair, compartment, formation fraction, and resorption fraction.
Mask segmentation can be generated with adaptive, seg_gauss, or Laplace-Hamming binarization.

## MotionScore Model Access

Motion Scoring no longer depends on a hosted license request service. The recommended model distribution path is a local or lab-hosted `.tar.gz` model bundle, optionally linked from a registration page or GitHub release. This keeps the workflow usable without paid infrastructure while still allowing aggregate download counts and voluntary user registration.

## Scanco AIM I/O

The `Scanco I/O` module is intended for simple Slicer round trips:

1. Import an AIM image using density/BMD, native, mu, or HU scaling.
2. Use standard Slicer tools to inspect, segment, crop, smooth, or edit the loaded volume.
3. Export the edited scalar volume or labelmap back to AIM.

The Slicer module is only the GUI layer. AIM parsing and writing are handled by a lightweight local wrapper backed by the `aimio-py` package (`py_aimio` import name). The `Install / Update AIM I/O` button installs only that package, not the full `timelapsed-hrpqct` pipeline.

Imported AIM metadata is stored on the loaded Slicer volume and shown as editable JSON in the export panel. For exports from volumes that did not originate from `Scanco I/O`, provide an imported-stack metadata JSON from the timelapsed pipeline, paste/edit header JSON manually, or explicitly enable minimal metadata export. Geometry fields that can be read from the selected Slicer volume, such as dimensions, spacing, origin, and direction, are refreshed at export time.

## Contours And Segmentation

The `Contours and Segmentation` module creates an initial threshold segmentation from a selected HR-pQCT volume and then opens Slicer's Segment Editor for manual contour cleanup. This module is intentionally small; it provides Slicer workflow glue and does not move core segmentation algorithms out of their Python packages.

## Results Layout

Pipeline outputs are saved in a structured MIDS/BIDS-style folder layout under the results root.

- Default results root: `<dataset_root>/TimelapsedHRpQCT`
- Optional override: `Results folder` field in the module
- Typical organization: subject/site/session-based folders with derivative outputs grouped by processing stage

This structure is intended to make results easy to browse, reload in Slicer, and reuse in downstream analysis.

## Input Filename Format

The parser expects AIM filenames that include:

- subject identifier
- site token (`DR`, `DT`, or `KN`)
- session token (for example `T1`, `T2`, `C1`, `BL`, `FL`, `FL1`)
- optional stack token for multistack data (`STACK01`, `STACK_01`, `STACK-01`)
- optional mask roles (`TRAB_MASK`, `CORT_MASK`, `FULL_MASK`, `REGMASK`, `ROI1`, `ROI2`, ...)

Examples:

```text
SUBJ001_DR_T1.AIM
SUBJ001_DR_T1_TRAB_MASK.AIM
SUBJ001_DR_T1_CORT_MASK.AIM
SUBJ001_DR_T2.AIM

SUBJ010_DT_STACK01_T1.AIM
SUBJ010_DT_STACK01_T1_TRAB_MASK.AIM
SUBJ010_DT_STACK02_T1.AIM
SUBJ010_DT_STACK02_T1_CORT_MASK.AIM

SAMPLE355_KN_BL.AIM
SAMPLE355_KN_FL1.AIM
SAMPLE355_KN_FL1_REGMASK.AIM
SAMPLE355_KN_FL1_ROI1.AIM
```

If filename parsing is incomplete or ambiguous, the parser can fall back to AIM header metadata (such as `Index Patient`, `Index Measurement`, and `Original Creation-Date`) when available.

Notes:

- Input discovery is recursive, so flat folders and nested BIDS/MIDS-style trees are both supported.
- Parse supports generic and sided sites (`radius/tibia/knee` and `*_left/*_right` variants).
- `Restructure raw inputs` is disabled when parse-table label overrides are active, because overrides run through a virtual input root.

## Publication

If you use this extension in your research, please cite:

Walle M, Whittier DE, Schenk D, Atkins PR, Blauth M, Zysset P, Lippuner K, Müller R, Collins CJ. Precision of bone mechanoregulation assessment in humans using longitudinal high-resolution peripheral quantitative computed tomography in vivo. *Bone*. 2023 Jul;172:116780. doi: 10.1016/j.bone.2023.116780. Epub 2023 May 1. PMID: 37137459.

## License

This extension is distributed under the **MIT License** (see [LICENSE](LICENSE)).

The core pipeline dependency `timelapsed-hrpqct` is installed separately from PyPI and is governed by its own license terms in that repository.
