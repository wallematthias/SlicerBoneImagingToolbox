# Changelog

All notable changes to this extension are documented in this file.

## [0.3.6] - 2026-10-05

### Fixed

- Require voidspace 0.1.5 so registered batch analysis accepts sub-nanometre AIM/NIfTI header rounding while retaining physical mask alignment and rejecting wrong-resolution common regions.
- Load longitudinal scene Voidspace results using the core package's expanded/contracted mask fields, avoiding the post-analysis AttributeError.
- Preserve theme-default table colours after Dataset Naming Helper cell edits and use contrasting palette colours for warnings in light and dark themes.

### Changed

- Explain in the longitudinal scene UI and Voidspace documentation that inputs must already be anatomically registered; grid resampling does not perform registration.

## [0.3.5] - 2026-10-02

### Changed

- Set standard XCTI periosteal defaults to 250 for radius/tibia and retain knee at 150, matching the updated core/batch presets. Keep XCTII settings and explicit custom/saved overrides unchanged; tissue segmentation, inner contours, morphology and smoothing are unchanged.
- Require bone-contouring 0.3.5 in Toolbox Setup so installed batch profiles use the same updated XCTI thresholds as the scene module.

## [0.3.4] - 2026-10-02

### Fixed

- Selecting XCTI initializes Laplace–Hamming tissue segmentation and XCTII initializes Gaussian; custom and saved profiles preserve their configured method.
- Require bone-contouring 0.3.4 in Setup for repaired standard outer filling and endosteal marrow-seed restriction. Document the shared contour path; existing saved masks must be explicitly regenerated to use the repair.

## [0.3.3] - 2026-10-02

### Fixed

- Restore original sidecars, preflight undo collisions, and archive completed rename manifests so rename/undo/rename works; use AIM patient headers for native Scanco measurement filenames without overriding explicit anonymized IDs.
- Preserve imported cropped-mask geometry in batch loading and align Microarchitecture/Functional bone inputs on the grayscale physical grid with nearest-neighbor resampling.
- Hide previous batch case segmentations without affecting manual nodes, preserve grouped registered timepoints, and avoid duplicate contour nodes on reload.
- Use available full bone masks for native Voidspace; compute registered/dynamic morphology before common-region clipping and recompute native maps when forcing registered runs.
- Accept native large-void maps for Functional bone and explain the missing Voidspace prerequisite. Core Voidspace also preserves scan-end-connected cavities and uses endpoint continuation during morphology.

### Added

- U-Net batch contouring now also generates XCTII Laplace–Hamming tissue SEG and material labels, reuses completed compartments without inference, and loads SEG alongside full/trab/cort in one node. Scene choices are unchanged.
- Add a Functional Bone batch tutorial covering required contour/voidspace inputs, native versus registered profiles, result inspection, and safe reruns.
- Add Native Functional Bone and Registered Functional Bone batch profiles under Microarchitecture, with separate measurement directories and analysis-region overlays, shared native maps, and preserved registered semantics for saved `functional-bone` jobs.
- Added an `HR-pQCT Density` volume-rendering preset with transparent sub-trabecular densities and progressively stronger trabecular and cortical opacity.
- Added a Voidspace scene module for single-case and baseline/follow-up interactive analysis backed by the `voidspace` core package.
- Added Batch Processor support for Voidspace, Registered voidspace, and Dynamic voidspace profiles.
- Added Voidspace runtime package status, setup metadata, toolbox registry entries, and user documentation.

### Changed

- Require bone-contouring 0.3.3, bone-imaging-derivatives 0.1.7, bone-microarchitecture 0.2.5, and voidspace 0.1.4 in Setup.
- Registered voidspace now stays in native segmentation space and applies the matching native common region for restricted reporting.
- Dynamic voidspace writes baseline/follow-up intermediate maps inside the dynamic pair output folder so Registered voidspace remains an independent profile.
- Voidspace load-back imports source segmentations, analysis masks, and change maps into one Slicer segmentation node with distinct display colors.
- Remote batch processing checks server availability before starting server-side work so missing VPN/server access fails quickly instead of hanging.

## [0.3.1] - 2026-10-02

### Changed

- Require `bone-contouring[unet]>=0.3.1` for matching custom-profile support, SEG behavior, and faster voxel-equivalent U-Net post-processing.
- Use Gaussian tissue-segmentation sigma 1.2 voxels with trabecular/cortical thresholds 320/450 mg HA/cm³; label sigma units explicitly and preserve contour smoothing and saved custom values.
- Remove tissue SEG's largest-component filtering/control, including legacy profile overrides; retain minimum-size noise cleanup and leave FEA's independent connectivity preprocessing intact.
- Fold the published U-Net into Contouring as a fixed-default backend with device selection, progress/cancel, and independent optional tissue SEG; add its profile under the existing Bone Contouring batch tool and retain upstream author/citation guidance.
- Put XCTI, XCTII, Custom and saved profiles in the first scene Profile selector, without a separate scanner selector or resolution-based scanner detection. Keep AIM-header site detection and prompt on unresolved Site Auto. Split Contouring and Tissue segmentation advanced settings beneath their respective selectors; U-Net exposes Device only. Preserve legacy scene recipes, tissue-only processing, and existing standard/custom batch profiles.
- Remove the standalone Deep Learning Segmentation module registration; retain shared scene/batch import and cleanup of legacy module paths.
- Honor wheel-only package installation settings before considering sibling editable checkouts, preventing Plate/Rod Morphometry updates from attempting local compilation.
- Require `bone-microarchitecture>=0.2.4` for canonical contour-role discovery; Plate/Rod Morphometry's binary-only upgrade can now install the macOS `0.1.9` wheels with shared contour discovery.

## [0.3.0] - 2026-10-02

### Added

- Add Deep Learning Segmentation scene mode and a U-Net Bone Contours batch tool using Neeteson et al.'s published model and fixed morphology through `bone-contouring[unet]`, without Bonelab or vtkbone.
- Import full, trabecular and cortical compartment masks into one segmentation node, with CPU/CUDA/MPS device selection and a shared, checksum-verified model cache downloaded by the runtime modules.
- Document setup, scene/batch/SSH use, weight provisioning, applicability limits, and attribution to Nathan J. Neeteson, Bryce A. Besler, Danielle E. Whittier and Steven K. Boyd.

### Changed

- Require `bone-contouring>=0.3.0` with its U-Net extra and `bone-imaging-derivatives>=0.1.6` in Setup. The scientific backend is GPL-3.0-only; the Slicer wrapper remains MIT.
- Publish U-Net masks under the standard BoneContours naming and manifest contract. Preserve existing compartments when generating missing tissue SEG/material labels.

### Fixed

- Withhold incomplete U-Net outputs from discovery; offer manifest-only publication retries after failed publication and block conflicting or partial existing contours.
- Ensure local-development Slicer workers import the selected core checkout instead of an older installed package.

## [0.2.3] - 2026-10-01

### Fixed

- Restore the standard contouring Peel control with a default 3-voxel XY minimum cortical compartment rim for XCTI/XCTII radius, tibia, and knee; retain custom peel overrides and cross-slice smoothing. The core applies the constraint after smoothing and hole filling, without peeling Z end slices.

### Changed

- Require `bone-contouring>=0.2.1` so Setup installs the implementation that honors the Peel control.

## [0.2.1] - 2026-09-08

### Changed

- Renamed the extension container to SlicerBoneImagingToolbox and moved built-in modules under the `Bone Imaging` Slicer category.
- Added a toolbox module manifest and external module discovery for vendored scripted modules under `ExternalModules/`.
- Renamed updater/linker metadata to the Bone Imaging Toolbox while keeping legacy import shims for existing internal imports.
- Added `Bone Imaging.HR-pQCT` and `Bone Imaging.I/O` Slicer subcategories for a clearer toolbox menu.
- Renamed the internal segmentation module ID from `HRpQCTSegmentation` to `SegmentationHRpQCT` and its display title to `Segmentation and Contours`.
- Grouped built-in module folders under `HRpQCTTools/` and `IOTools/` while keeping explicit top-level CMake entries for Slicer ExtensionIndex builds.
- Reworked the README into a compact toolbox overview and moved detailed tool instructions and attribution into focused per-tool documentation.
- Updated the toolbox logo image for the Bone Imaging Toolbox name, including a transparent PNG variant for README and extension metadata previews.
- Extended Scanco I/O import to use `aimio-py` image dispatch for AIM, ISQ, SCV, and GOBJ files, with explicit Slicer drag/drop readers for native volume, HU volume, density volume, and native segmentation import.
- Added a `Bone Imaging.CT` Spine Segmentation module that installs `spine-segment`, runs vertebral CT segmentation in a Slicer subprocess, and loads vertebral-level, process/body, and cortical/trabecular outputs.
- Added the central Batch Processor with shared discovery, queued row execution, load-back support, skip-existing behavior, and profile-aware row grouping for HR-pQCT workflows.
- Added optional private batch backend hooks so remote ARC/SLURM execution can be enabled from the private toolbox without making public modules depend on private code.
- Improved remote batch robustness by freezing each queued job's tool/profile/row context, keeping completion state tied to stable row identity, and surfacing remote FEA solver log tails.
- Kept remote dataset path configuration in the private setup while preserving public Batch Processor wall-time, memory, and CPU overrides for individual batches.

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
