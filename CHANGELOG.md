# Changelog

All notable changes to this extension are documented in this file.

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
