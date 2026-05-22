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
  - Generates HR-pQCT full, trabecular, cortical, and binary segmentation outputs from an input volume.
  - Provides radius, tibia, and knee contour presets plus standard Gaussian, Laplace-Hamming, and adaptive segmentation methods.
  - Keeps expert threshold and morphology settings collapsed by default, then optionally opens Slicer's Segment Editor for cleanup.

## Installation

The toolbox is not yet available in the full stable Slicer Extensions Index for all users. Until it is listed for your Slicer version, install it manually by downloading this repository and adding the module folders to Slicer.

### Option A: Extension Manager (when listed)

1. Open 3D Slicer.
2. Install `HR-pQCT Toolbox` from Extension Manager.
3. Restart 3D Slicer.

### Option B: Manual install from download or clone

1. Download or clone this repository:
   - Git clone: `git clone https://github.com/wallematthias/SlicerTimelapsedHRpQCT.git`
   - Or download the repository ZIP from GitHub and extract it somewhere permanent.
2. Open 3D Slicer.
3. Go to `View -> Python Interactor`.
4. Run the helper script below, replacing `<repo>` with the folder that contains this README:
   ```python
   script = "<repo>/scripts/link_local_toolbox_modules.py"
   exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
   ```
5. Restart Slicer.
6. Open modules from the `HR-pQCT` category:
   - `Timelapsed HR-pQCT`
   - `Motion Scoring`
   - `Scanco I/O`
   - `Contours and Segmentation`

Example script paths:

- macOS/Linux: `script = "/Users/<you>/Downloads/SlicerTimelapsedHRpQCT/scripts/link_local_toolbox_modules.py"`
- Windows: `script = r"C:\Users\<you>\Downloads\SlicerTimelapsedHRpQCT\scripts\link_local_toolbox_modules.py"`

### Manual Slicer UI alternative

If you prefer not to run the helper script, add the module folders through Slicer's settings:

1. In Slicer, open `Edit -> Application Settings -> Modules`.
2. Under `Additional module paths`, add each folder below.
3. Click `OK` or `Apply`, then restart Slicer.

Add these folders, replacing `<repo>` with the downloaded/cloned repository folder:

- `<repo>/TimelapsedHRpQCT`
- `<repo>/MotionScoreHRpQCT`
- `<repo>/ScancoIO`
- `<repo>/HRpQCTSegmentation`

Do not add only the top-level repository folder. Slicer needs the four module folders above to load the whole toolbox.

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

The Timelapsed module lists the profiles bundled with the installed `timelapsed-hrpqct` package, including standard, XCT1 standard, ETH/UofC, UCSF, Shriners, and workflow profiles such as multistack.
Applying a profile updates the visible analysis controls and passes the selected core profile to new runs.
After loading a saved remodelling image, the module exposes the key review parameters used by the pipeline: threshold, cluster size, analysis method, pair mode, full-mask dilation, marrow-mask erosion, Gaussian filtering, and Gaussian sigma.
When auto update is enabled, changing these controls recomputes the loaded preview from the transformed image pair already on disk, so exploratory threshold/filter changes do not require rerunning the full command-line analysis.
The series summary panel can load saved cohort-level pairwise outputs and export the available cohort rows with subject/site, timepoint pair, compartment, formation volume fraction (`FV/BV`), resorption volume fraction (`RV/BV`), net change volume fraction (`NV/BV = FV/BV - RV/BV`), and active volume fraction (`AV/BV = FV/BV + RV/BV`).
When scan interval metadata is already present in pairwise outputs, the export also carries `scan_period_days` and `scan_period_years`; these interval fields are reported for context and are not used to normalize remodelling metrics.
Mask segmentation can be generated with adaptive, seg_gauss, or Laplace-Hamming binarization.

## MotionScore Model Access

Motion Scoring no longer depends on a hosted license request service. The recommended model distribution path is a downloadable `.tar.gz` model bundle, for example a GitHub release artifact or local/lab-hosted model bundle. This keeps the workflow usable without paid infrastructure while still allowing aggregate download counts and voluntary user registration when useful.

The Slicer module has a `Local models folder` field in the Motion Scoring setup panel. The automatic `Install / Download Models` button downloads the default model from the MotionScoreHRpQCT release catalog. If that automatic download is unavailable, install the same model weights manually:

1. Download the default MotionScore base model bundle:
   - Windows-friendly zip: https://github.com/wallematthias/MotionScoreHRpQCT/releases/download/v2.5.4/motionscore-base-v1.zip
   - macOS/Linux tarball: https://github.com/wallematthias/MotionScoreHRpQCT/releases/download/v2.5.4/motionscore-base-v1.tar.gz
2. In Slicer, open `Motion Scoring` and note the path shown in `Local models folder`.
3. Extract the downloaded bundle into that folder.
   - The `.zip` bundle already contains the `base-v1` folder.
   - If using the `.tar.gz` bundle manually, create a `base-v1` folder inside `Local models folder` and extract the tarball contents into that `base-v1` folder.
4. Confirm the extracted files include model weights named like `DNN_*.pt` or `DNN_*.h5`, either directly in the local models folder or inside a model subfolder such as `base-v1`.
5. Restart Slicer or reopen the `Motion Scoring` module, then select `Base v1 (base-v1)` from `Model Profile`.

Recommended folder structure:

```text
<Local models folder>/
  base-v1/
    DNN_*.pt
    ...other files from the extracted model bundle...
```

For example, if the Slicer field shows:

```text
C:\Users\<you>\AppData\Roaming\NA-MIC\Slicer 5.10\MotionScore\models
```

then the model files should be placed like:

```text
C:\Users\<you>\AppData\Roaming\NA-MIC\Slicer 5.10\MotionScore\models\base-v1\DNN_*.pt
```

On macOS/Linux, the same structure applies:

```text
<Local models folder>/base-v1/DNN_*.pt
```

The module also accepts a flat fallback layout:

```text
<Local models folder>/
  DNN_*.pt
```

However, the `base-v1` subfolder layout is preferred because it keeps model versions separated.

When valid model weights are already present in `Local models folder`, the Slicer module uses those local files and skips the GitHub download. This is the preferred setup for managed workstations or offline installations.

The current default model catalog is:

- https://github.com/wallematthias/MotionScoreHRpQCT/releases/latest/download/model_catalog.json

## Scanco AIM I/O

The `Scanco I/O` module is intended for simple Slicer round trips:

1. Import an AIM image using density/BMD, native, mu, or HU scaling.
2. Use standard Slicer tools to inspect, segment, crop, smooth, or edit the loaded volume.
3. Export the edited scalar volume or labelmap back to AIM.

The Slicer module is only the GUI layer. AIM parsing and writing are handled by a lightweight local wrapper backed by the `aimio-py` package (`py_aimio` import name). The `Install / Update AIM I/O` button installs only that package, not the full `timelapsed-hrpqct` pipeline.

Imported AIM metadata is stored on the loaded Slicer volume. The processing log is shown as an editable field table using `aimio-py`'s `log_to_dict` / `dict_to_log` helpers, while the remaining AIM header metadata is shown as editable JSON. For exports from volumes that did not originate from `Scanco I/O`, provide an imported-stack metadata JSON from the timelapsed pipeline, paste/edit header JSON manually, or explicitly enable minimal metadata export. Geometry fields that can be read from the selected Slicer volume, such as dimensions, spacing, origin, and direction, are refreshed at export time.

## Contours And Segmentation

The `Contours and Segmentation` module wraps the core `timelapsed-hrpqct` contour-generation code. It can generate full, trabecular, cortical, and binary segmentation outputs from a selected HR-pQCT volume using radius, tibia, or knee presets. Segmentation methods include standard Gaussian thresholding, Laplace-Hamming, and adaptive thresholding. Expert thresholds and morphology settings are available in a collapsed panel for method validation and scanner-specific tuning.

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

## Method Citations

For the main `eth-uofc` timelapsed HR-pQCT method, cite:

Walle M, Whittier DE, Schenk D, Atkins PR, Blauth M, Zysset P, Lippuner K, Müller R, Collins CJ. Precision of bone mechanoregulation assessment in humans using longitudinal high-resolution peripheral quantitative computed tomography in vivo. *Bone*. 2023 Jul;172:116780. doi: 10.1016/j.bone.2023.116780. Epub 2023 May 1. PMID: 37137459.

Related applications using the method include:

- Walle M, Duseja A, Whittier DE, Vilaca T, Paggiosi M, Eastell R, Müller R, Collins CJ. Bone remodeling and responsiveness to mechanical stimuli in individuals with type 1 diabetes mellitus. *Journal of Bone and Mineral Research*. 2024;39(2):85-94.
- Walle M, Gabel L, Whittier DE, Liphardt AM, Hulme PA, Heer M, Zwart SR, Smith SM, Sibonga JD, Boyd SK. Tracking of spaceflight-induced bone remodeling reveals a limited time frame for recovery of resorption sites in humans.
- Matheson BE, Walle M, Bugbird AR, Rosenberg M, Mateus J, Boyd SK. Early skeletal deteriorations following short-duration spaceflight.

For multistack registration, cite:

Whittier DE, Walle M, Schenk D, Atkins PR, Collins CJ, Zysset P, Lippuner K, Müller R. A multi-stack registration technique to improve measurement accuracy and precision across longitudinal HR-pQCT scans. *Bone*. 2023;176:116893. doi: 10.1016/j.bone.2023.116893.

For adapted profile variants, cite the profile-specific paper when relevant:

- `shriners`: Hosseinitabatabaei S, Vitienes I, Rummler M, Birkhold A, Rauch F, Willie BM. Non-invasive quantification of bone (re)modeling dynamics in adults with osteogenesis imperfecta treated with setrusumab using timelapse high-resolution peripheral-quantitative computed tomography. *Journal of Bone and Mineral Research*. 2025;40(3):348. https://academic.oup.com/jbmr/article/40/3/348/7978263
- `ucsf`: Zhou M, Sadoughi S, Go L, Ramil G, Yu I, Saeed I, Fan B, Wu PH, Salusky IB, Nickolas TL, Ix JH, Kazakia GJ. Time-lapse HR-pQCT reliably assesses and monitors local bone turnover in patients with chronic kidney disease. *Journal of Bone and Mineral Research*. 2025;40(6):738-752. doi: 10.1093/jbmr/zjaf006.

For Laplace-Hamming segmentation, refer to the Galateia Kazakia lab implementation and related work: https://github.com/gkazakia

For Motion Scoring, cite:

Walle M, Eggemann D, Atkins PR, Kendall JJ, Stock K, Müller R, Collins CJ. Motion grading of high-resolution quantitative computed tomography supported by deep convolutional neural networks. *Bone*. 2023 Jan;166:116607. doi: 10.1016/j.bone.2022.116607. Epub 2022 Nov 8. PMID: 36368464.

## License

This extension is distributed under the **MIT License** (see [LICENSE](LICENSE)).

The core pipeline dependency `timelapsed-hrpqct` is installed separately from PyPI and is governed by its own license terms in that repository.
