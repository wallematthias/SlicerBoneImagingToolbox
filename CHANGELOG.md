# Changelog

All notable changes to this extension are documented in this file.

## Unreleased

### Changed

- Renamed the extension container to SlicerBoneImagingToolbox and moved built-in modules under the `Bone Imaging` Slicer category.
- Added a toolbox module manifest and external module discovery for vendored scripted modules under `ExternalModules/`.
- Renamed updater/linker metadata to the Bone Imaging Toolbox while keeping legacy import shims for existing internal imports.
- Added `Bone Imaging.HR-pQCT` and `Bone Imaging.I/O` Slicer subcategories for a clearer toolbox menu.
- Renamed the internal segmentation module ID from `HRpQCTSegmentation` to `SegmentationHRpQCT` and its display title to `Segmentation and Contours`.
- Grouped built-in module folders under `HRpQCTTools/` and `IOTools/` while keeping explicit top-level CMake entries for Slicer ExtensionIndex builds.
- Reworked the README into a compact toolbox overview and moved detailed tool instructions and attribution into focused per-tool documentation.
- Updated the toolbox logo image for the Bone Imaging Toolbox name.

## [0.2.0] - 2026-05-21

### Added

- Evolved the extension into an HR-pQCT Toolbox with Timelapsed, Motion Scoring, Scanco I/O, and Contours and Segmentation modules under one Slicer category.
- Added a Scanco I/O module for AIM import with density/native/mu/HU scaling and AIM export for grayscale or binary mask volumes.
- Added a Contours and Segmentation module for threshold-based segmentation creation followed by cleanup in Slicer's Segment Editor.
- Added Contours and Segmentation mask utility tools for missing-mask derivation, boolean operations, relabelling, validation, voxel counts, and HOM/material labelmap creation.
- Added broad tooltip coverage across the toolbox modules for install, parsing, profile, review, export, segmentation, and mask utility controls.
- Added MotionScoreHRpQCT as a sibling toolbox module.
- Added Timelapsed study profile selection for ETH/UofC, UCSF, Shriners, and standard core defaults.
- Added MotionScore local/downloaded model bundle setup as a no-hosted-license-service alternative.
- Added editable AIM header metadata display in the Scanco I/O export panel, including a processing-log field table backed by `aimio-py` log/dict helpers.
- Added radius/tibia/knee presets and standard Gaussian, Laplace-Hamming, and adaptive segmentation methods to the Contours and Segmentation module.

### Changed

- Updated extension metadata and README language from a single Timelapsed wrapper toward the HR-pQCT Toolbox identity while keeping the existing repository URL.
- Simplified Timelapsed remodelling review controls by removing the 3D preview rendering controls.
- Removed separate Timelapsed quick presets so study profile is the single preset/profile control.
- Scanco I/O now installs only `aimio-py` and no longer requires installing the full `timelapsed-hrpqct` pipeline.

## [0.1.6] - 2026-05-20

### Added

- Added `laplace_hamming` as a mask segmentation method option, with LH threshold and minimum component size controls mapped into the core pipeline config.

## [0.1.5] - 2026-05-20

### Changed

- Cohort row export now keeps the compact row format: subject/site, compartment, timepoint pair, formation fraction, and resorption fraction.

## [0.1.4] - 2026-05-20

### Changed

- The series summary export now writes only the cohort row table, preserving all available saved pairwise remodelling rows instead of writing a separate aggregate summary CSV.

## [0.1.3] - 2026-05-20

### Fixed

- Dataset/results root changes now rehydrate pipeline progress from existing artifact indexes and output files, so reopening a processed data root shows completed parse, mask, registration, and analysis stages.

## [0.1.2] - 2026-05-20

### Added

- Study-level summary export from the series summary panel, with CSV output and optional XLSX output when workbook support is available.
- README documentation for interactive remodelling review controls and live preview behavior.

### Fixed

- Progress state now refreshes immediately when the dataset or results root changes.
- Parse failures now stay in the module UI/log and provide editable manual fallback rows for anonymized or unusual AIM filenames.

## [0.1.1] - 2026-04-08

### Added

- Expanded workflow controls and analysis tooling in the scripted module UI.
- Additional process/runtime guards for cleaner subprocess handling and temporary config cleanup.

### Changed

- Default results root in documentation now matches pipeline output path (`<dataset_root>/TimelapsedHRpQCT`).
- Refined registration/analysis panel organization for clearer stage-level tuning.

### Fixed

- Suppressed recurring SimpleITK/ITK warning noise in Slicer logs.
- Improved compatibility handling for process output decoding and pipeline runtime setup.

## [0.1.0] - 2026-03-31

### Added

- First release-ready Slicer module scaffold for TimelapsedHRpQCT workflows.
- Pipeline controls for full run, masks, timelapse, multistack, and analysis rerun.
- Remodelling segmentation loading and 3D preview tools.
- Runtime install/update flow for `timelapsed-hrpqct`.
- Module icon wiring and extension metadata setup.
- Scripted smoke tests (`TimelapsedHRpQCTTest`).

### Changed

- UI compacted for better 2D/3D viewer space.
- Improved process logging/cancellation behavior and stage status handling.
- Robust config resolution with fallback when packaged defaults are missing.
- Improved artifact lookup fallback for remodelling load paths.

### Fixed

- QProcess output decode handling across Qt/Python bindings.
- Process-finished callback signature compatibility.
- Analysis stage status completion behavior in full pipeline runs.
- Forced reinstall behavior for dependency update button.
