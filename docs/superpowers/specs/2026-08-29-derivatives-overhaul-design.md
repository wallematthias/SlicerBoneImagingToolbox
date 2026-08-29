# Derivatives Overhaul Design

## Goal

Refactor SlicerBoneImagingToolbox around reusable derivative products so registration, common-region masks, microarchitecture, plate/rod morphometry, timelapsed analysis, FEA, mechanoregulation, trabecular segmentation, and void-space analysis can share inputs and outputs without duplicating workflow logic.

## Core Model

Major workflows produce named derivatives under a shared derivatives root. Each derivative writes a manifest that records inputs, outputs, coordinate space, settings, software versions, and dependency relationships. Modules discover existing derivatives through manifests first and fall back to legacy filename discovery when manifests are absent.

Scene mode and batch mode are both supported. Scene mode consumes loaded Slicer MRML nodes and creates MRML nodes. Batch mode consumes folders/manifests and writes derivative folders. Both modes call the same backend services.

## Derivative Families

- `Registration`: pairwise and composed transforms for longitudinal series. Supports single-stack and multi-stack series.
- `CommonRegion`: scan/FOV common-region masks built from registration outputs. It does not intersect biological masks.
- `Segmentation`: bone, periosteal/full, endosteal/trabecular, cortical, and future role-specific masks.
- `Microarchitecture`: measurement tables and scalar/map outputs from grayscale images and masks.
- `PlateRodMorphometry`: plate/rod maps and tables from bone/trabecular masks.
- `Timelapsed`: longitudinal remodelling/change outputs using registration, common region, and compartment masks.
- `FEA`: meshes, material maps, boundary conditions, solver outputs, and mechanical fields.
- `Mechanoregulation`: combined biological change and mechanical-field outputs.
- `VoidSpace`: future void-space masks, maps, and measurements.

## Shared Services

Shared logic belongs in `SlicerBoneImagingToolboxLib`, not individual Slicer modules:

- derivative manifest read/write/discovery
- dataset and scene discovery
- mask-role normalization
- image IO and calibration metadata helpers
- registration wrappers around `timelapsedhrpqct`
- common-region construction
- dependency planning
- background job execution
- Slicer node/file adapters
- result table/export helpers

## Dependency Resolution

Each workflow declares required and produced derivatives. When a user runs a workflow, the module builds an execution plan:

1. discover available scene nodes or derivative manifests
2. check whether required derivatives exist and are compatible
3. generate missing prerequisites when inputs are sufficient and the user has enabled prerequisite generation
4. run the requested workflow
5. write or load outputs

Example: Timelapsed requires registration and common region. If they do not exist, Timelapsed triggers the shared registration and common-region services before running Timelapsed analysis.

## Common-Region Semantics

Common region means overlapping scan/FOV support across all selected timepoints. It is computed from scan-region masks transformed into a reference space, intersected there, and returned to each native image space. It is not a bone, trabecular, cortical, marrow, or void mask.

Analysis modules apply the common region locally:

```python
analysis_mask = biological_mask & native_common_scan_region
```

This preserves biological change while constraining measurement support to the comparable scan region.

## Public Module Pattern

Each major public module uses the same structure:

- `Scene` tab: loaded Slicer nodes in, MRML nodes/tables/transforms out
- `Batch` tab: dataset/derivatives folder in, derivative files/manifests out
- status panel: discovered inputs, prerequisite status, and generated outputs
- background execution for long-running operations

## Migration Rules

- Do not duplicate Timelapsed registration algorithms inside Slicer modules.
- Move registration/common-region code out of `BoneMicroarchitecture.py`.
- Keep batch cohort orchestration out of public analysis modules unless it is generally reusable.
- Maintain compatibility with existing `RegisteredMicroarchitecture` outputs during migration.
- Prefer manifests over implicit folder assumptions for new code.
- Keep rollback tag `pre-derivatives-overhaul-2026-08-29` as the pre-overhaul checkpoint.
