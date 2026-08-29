# Cross-Repository Derivatives Architecture

## Goal

Restructure the Timelapsed HR-pQCT, bone microarchitecture, plate/rod, and Slicer toolbox ecosystem around shared derivative products so command-line workflows and Slicer workflows use the same core logic, the same folder contracts, and the same provenance model.

## Current Problem

The current Slicer overhaul branch adds useful derivative concepts, but too much orchestration still lives in `SlicerBoneImagingToolbox`. That is backwards for long-term maintenance. Registration, longitudinal transform chaining, common scan-region generation, dataset discovery, and derivative manifest writing are package-level concerns. Slicer should expose them, launch them in background workers, and load outputs for review.

The existing package boundary should be tightened:

- `timelapsed-hrpqct` should own longitudinal discovery, registration, transform chains, common scan regions, and timelapsed remodelling.
- `bone-microarchitecture` should own microarchitecture measurements from arrays and masks.
- `plate-rod-thinning` should own plate/rod computation from arrays and masks.
- `SlicerBoneImagingToolbox` should own MRML node selection, scene loading, package installation, background process management, and visualization.

## Design Principles

1. Core algorithms live in Python packages, not Slicer modules.
2. Slicer modules are adapters around package APIs.
3. Every reusable workflow writes a derivative manifest.
4. Batch mode and Slicer scene mode call the same package-level workflow functions.
5. Common scan region means scan/FOV overlap only, not biological mask intersection.
6. Biological masks are clipped by common scan region at analysis time.
7. Missing prerequisites may be generated automatically if enough inputs and settings are available.
8. Command-line and Slicer behavior must remain equivalent.
9. New code should prefer explicit manifest records over filename inference.
10. Existing Timelapsed outputs remain readable during migration.

## Repository Ownership

### `timelapsed-hrpqct`

This becomes the source of truth for reusable longitudinal infrastructure.

Responsibilities:

- structured dataset discovery
- subject/site/session/stack grouping
- registration inputs and mask selection
- sequential pairwise registration
- transform composition to reference space
- transform serialization and discovery
- scan-region mask generation
- common scan-region construction
- native-space common-region export
- derivative manifest schema and writer
- workflow dependency planning for longitudinal jobs
- command-line entry points for registration and common region
- Timelapsed remodelling analysis

This package should not import Slicer.

### `bone-microarchitecture`

This remains a measurement package.

Responsibilities:

- scalar and map-based microarchitecture metrics
- backend selection for CPU, Metal/MPS-style local GPU code, and OpenCL where supported
- mask intersection rules inside the measurement function
- table/report formatting for measurement outputs
- optional CLI for one case or a folder of already-prepared inputs

This package should not own longitudinal registration or derivative discovery. It may accept an optional common-region mask as an input.

### `plate-rod-thinning`

This remains a morphology package.

Responsibilities:

- plate/rod skeletonization and morphometry
- optional GPU backend internals when available
- one-case API and optional CLI
- measurement outputs and maps

This package should not own registration, common-region generation, or Slicer loading.

### `SlicerBoneImagingToolbox`

This becomes the user-facing Slicer shell around package workflows.

Responsibilities:

- install/status UI for package dependencies
- Slicer node-to-file and file-to-node adapters
- segmentation node label/segment selection
- scene-mode execution for loaded nodes
- batch-mode job configuration
- background process launch and progress display
- output loading into Slicer tables, volumes, labelmaps, transforms, and models
- derivative manifest browsing and status display

It should not duplicate Timelapsed registration/common-region algorithms.

### Future Repositories

Future packages should consume the same derivative contract:

- `void-space`
- FEA package or adapter
- `BoneMechanoregulation`
- future individual trabecular segmentation package
- future density/calibration tools

Each package should either produce or consume manifest records with declared derivative family, role, space, and provenance.

## Shared Derivative Contract

The derivative contract should move from `SlicerBoneImagingToolboxLib` into `timelapsedhrpqct.derivatives` or, if reuse grows beyond Timelapsed, a small standalone package later. The first implementation should stay inside `timelapsed-hrpqct` to avoid creating another package too early.

### Manifest Location

Each derivative family writes:

```text
<dataset_root>/derivatives/<DerivativeFamily>/manifest.json
```

Examples:

```text
derivatives/Registration/manifest.json
derivatives/CommonRegion/manifest.json
derivatives/Microarchitecture/manifest.json
derivatives/PlateRodMorphometry/manifest.json
derivatives/Timelapsed/manifest.json
derivatives/FEA/manifest.json
derivatives/Mechanoregulation/manifest.json
derivatives/VoidSpace/manifest.json
```

Legacy `derivatives/TimelapsedHRpQCT/...` outputs remain discoverable through compatibility readers.

### Manifest Schema

Schema version: `1`

Required top-level fields:

```json
{
  "schema_version": 1,
  "derivative_family": "CommonRegion",
  "software": {
    "name": "timelapsed-hrpqct",
    "version": "2.x"
  },
  "dataset_root": "/absolute/or/relative/root",
  "created_at": "2026-08-29T00:00:00Z",
  "records": []
}
```

Each record contains:

```json
{
  "derivative": "CommonRegion",
  "role": "scan_region_native_common",
  "subject_id": "SAMPLE001",
  "site": "tibia",
  "session_id": "1",
  "stack_index": 1,
  "space": "native",
  "path": "derivatives/CommonRegion/sub-SAMPLE001/site-tibia/native_space/ses-1/masks/sub-SAMPLE001_ses-1_site-tibia_mask-scan-region_native_common.nii.gz",
  "source": "generated",
  "inputs": [],
  "metadata": {}
}
```

Required record fields:

- `derivative`: derivative family name
- `role`: semantic role
- `subject_id`: subject identifier without `sub-`
- `site`: normalized site
- `session_id`: session identifier without `ses-`
- `stack_index`: integer or null
- `space`: `native`, `reference`, `moving`, `fixed`, `model`, or `table`
- `path`: path relative to dataset root when possible
- `source`: `generated`, `provided`, `derived`, or `legacy`
- `inputs`: list of record IDs or source paths
- `metadata`: JSON object

Optional record fields:

- `record_id`: stable string generated from family, role, subject, site, session, stack, space, and path
- `content_type`: `image`, `mask`, `transform`, `table`, `mesh`, `metadata`, or `report`
- `coordinate_reference`: reference session/stack/space metadata
- `settings_hash`: hash of the settings used to generate the record
- `software`: record-specific software override

## Standard Derivative Families and Roles

### Registration

Produced by `timelapsed-hrpqct`.

Roles:

- `transform_pairwise`: adjacent sequential registration transform
- `transform_to_reference`: composed transform from native session to reference session
- `transform_from_reference`: inverse transform when materialized
- `registration_mask`: mask used for registration
- `registration_qc`: optional table or figure output

Expected layout:

```text
derivatives/Registration/sub-<subject>/site-<site>/transforms/
derivatives/Registration/sub-<subject>/site-<site>/qc/
```

### CommonRegion

Produced by `timelapsed-hrpqct`.

Roles:

- `scan_region_native`: scan/FOV support mask for one native image
- `scan_region_reference`: native scan support transformed into reference space
- `scan_region_common_reference`: intersection of scan support in reference space
- `scan_region_native_common`: common scan support transformed back to native image space

Expected layout:

```text
derivatives/CommonRegion/sub-<subject>/site-<site>/reference_space/
derivatives/CommonRegion/sub-<subject>/site-<site>/native_space/ses-<session>/masks/
```

Important rule:

`CommonRegion` must not intersect trabecular, cortical, full, bone, marrow, void, or any other biological masks. It only defines comparable scan support.

### Segmentation

Produced by Slicer segmentation tools or package-level segmentation workflows.

Roles:

- `bone_segmentation`
- `periosteal_mask`
- `endosteal_mask`
- `trabecular_mask`
- `cortical_mask`
- `scan_region_mask`
- future roles such as `void_mask`

The terms should be normalized at the manifest level. Slicer labels may be user-facing, but records should use stable role strings.

### Microarchitecture

Produced by `bone-microarchitecture` directly or through Slicer.

Roles:

- `measurements_table`
- `trabecular_thickness_map`
- `trabecular_spacing_map`
- `trabecular_number_map`
- `cortical_thickness_map`
- `cortical_porosity_map`
- optional density/BMD tables and maps

Inputs:

- grayscale image
- bone segmentation
- periosteal/full mask
- trabecular mask, cortical mask, or both
- optional common scan-region mask

Analysis rule:

```python
effective_mask = biological_mask & common_scan_region
```

### PlateRodMorphometry

Produced by `plate-rod-thinning` directly or through Slicer.

Roles:

- `plate_rod_label_map`
- `plate_rod_measurements_table`
- `plate_local_thickness_map`
- `rod_local_thickness_map`
- `skeleton_map`

Inputs:

- trabecular or bone mask
- optional common scan-region mask

### Timelapsed

Produced by `timelapsed-hrpqct`.

Roles:

- `remodelling_pairwise_table`
- `remodelling_trajectory_table`
- `formation_mask`
- `resorption_mask`
- `stable_mask`
- `transformed_image`
- `filled_image`
- `qc_report`

Inputs:

- registration transforms
- common scan-region masks
- compartment masks
- native grayscale images or transformed images depending on workflow mode

### FEA

Future derivative family.

Roles:

- `mesh`
- `material_map`
- `boundary_conditions`
- `solver_input`
- `solver_output`
- `strain_map`
- `stress_map`
- `load_transfer_table`

Inputs:

- calibrated grayscale image
- segmentation masks
- optional common scan-region mask
- optional registration transforms for longitudinal comparisons

### Mechanoregulation

Future derivative family.

Roles:

- `mechanoregulation_table`
- `mechanical_signal_map`
- `formation_mechanics_map`
- `resorption_mechanics_map`
- `adaptation_classification_map`

Inputs:

- Timelapsed remodelling outputs
- FEA fields
- common scan-region masks
- registration metadata

### VoidSpace

Future derivative family.

Roles:

- `void_mask`
- `void_measurements_table`
- `void_distance_map`
- `void_connectivity_map`

Inputs:

- grayscale image
- bone or periosteal mask
- optional common scan-region mask

## Package API Design

### `timelapsedhrpqct.derivatives`

New package area:

```text
src/timelapsedhrpqct/derivatives/
  __init__.py
  manifest.py
  records.py
  discovery.py
  roles.py
  compatibility.py
```

Key interfaces:

```python
@dataclass(frozen=True)
class DerivativeRecord:
    derivative: str
    role: str
    subject_id: str
    site: str
    session_id: str | None
    stack_index: int | None
    space: str
    path: Path
    source: str
    inputs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DerivativeManifest:
    schema_version: int
    derivative_family: str
    dataset_root: Path
    records: tuple[DerivativeRecord, ...]
    software: Mapping[str, str]
    created_at: str
```

Functions:

```python
def read_manifest(path: Path) -> DerivativeManifest: ...
def write_manifest(manifest: DerivativeManifest, path: Path) -> None: ...
def discover_manifests(dataset_root: Path) -> list[DerivativeManifest]: ...
def find_records(
    manifests: Sequence[DerivativeManifest],
    *,
    derivative: str | None = None,
    role: str | None = None,
    subject_id: str | None = None,
    site: str | None = None,
    session_id: str | None = None,
    stack_index: int | None = None,
    space: str | None = None,
) -> list[DerivativeRecord]: ...
```

Compatibility:

```python
def discover_legacy_timelapsed_records(dataset_root: Path) -> list[DerivativeRecord]: ...
def write_compatibility_manifest(dataset_root: Path) -> Path: ...
```

### `timelapsedhrpqct.registration`

New public package area or cleaned wrapper over existing `processing.registration` and `processing.transform_chain`.

```text
src/timelapsedhrpqct/registration/
  __init__.py
  models.py
  workflow.py
  io.py
```

Key interfaces:

```python
@dataclass(frozen=True)
class RegistrationInput:
    subject_id: str
    site: str
    session_id: str
    stack_index: int | None
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class RegistrationWorkflowResult:
    reference_session_id: str
    manifests: tuple[DerivativeManifest, ...]
    records: tuple[DerivativeRecord, ...]
```

Functions:

```python
def run_registration_workflow(
    inputs: Sequence[RegistrationInput],
    *,
    output_root: Path,
    reference_session_id: str | None = None,
    settings: RegistrationSettings | None = None,
    progress: Callable[[str], None] | None = None,
) -> RegistrationWorkflowResult: ...
```

The implementation should call existing registration and transform-chain code rather than rewriting algorithms.

### `timelapsedhrpqct.common_region`

New public package area:

```text
src/timelapsedhrpqct/common_region/
  __init__.py
  masks.py
  workflow.py
  io.py
```

Key interfaces:

```python
@dataclass(frozen=True)
class CommonRegionInput:
    subject_id: str
    site: str
    session_id: str
    stack_index: int | None
    image_path: Path
    transform_to_reference_path: Path


@dataclass(frozen=True)
class CommonRegionWorkflowResult:
    reference_session_id: str
    reference_common_mask_path: Path
    native_common_mask_paths: Mapping[str, Path]
    manifest: DerivativeManifest
```

Functions:

```python
def scan_region_from_image(image: sitk.Image) -> sitk.Image: ...

def build_common_scan_region(
    inputs: Sequence[CommonRegionInput],
    *,
    output_root: Path,
    reference_session_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> CommonRegionWorkflowResult: ...
```

Semantics:

1. Create a full scan-region mask for each native image.
2. Transform each scan-region mask into reference space.
3. Intersect transformed scan-region masks in reference space.
4. Transform the common scan-region mask back to each native session.
5. Save both reference and native-space common-region masks.
6. Write `CommonRegion/manifest.json`.

### `timelapsedhrpqct.workflows.dependencies`

New workflow dependency planner:

```python
@dataclass(frozen=True)
class WorkflowRequirement:
    derivative: str
    roles: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class WorkflowPlan:
    workflow: str
    available: tuple[DerivativeRecord, ...]
    missing: tuple[WorkflowRequirement, ...]
    steps: tuple[str, ...]
    blocked: bool
```

Functions:

```python
def resolve_workflow_plan(
    workflow: str,
    *,
    manifests: Sequence[DerivativeManifest],
    subject_id: str,
    site: str,
    sessions: Sequence[str],
    generate_missing: bool,
) -> WorkflowPlan: ...
```

Initial workflow dependencies:

- `CommonRegion` requires `Registration.transform_to_reference`.
- `Microarchitecture` may consume `CommonRegion.scan_region_native_common`.
- `PlateRodMorphometry` may consume `CommonRegion.scan_region_native_common`.
- `Timelapsed` requires `Registration.transform_to_reference` and should consume `CommonRegion.scan_region_native_common`.
- `FEA` may consume `CommonRegion.scan_region_native_common`.
- `Mechanoregulation` requires `Timelapsed` and `FEA`.

## Command-Line Design

The command-line interface should expose derivative-producing workflows directly.

Existing command:

```text
timelapse <command>
```

New or revised commands:

```text
timelapse derivatives inspect <dataset_root>
timelapse registration run <dataset_root> --subject SAMPLE001 --site tibia
timelapse common-region run <dataset_root> --subject SAMPLE001 --site tibia
timelapse prerequisites ensure <dataset_root> --workflow timelapsed --subject SAMPLE001 --site tibia
timelapse analyse <dataset_root> --subject SAMPLE001 --site tibia --use-common-region
```

Rules:

- Existing commands remain available.
- New commands write manifests.
- Commands accept `--subject` and `--site`.
- Commands support `--dry-run`.
- Commands support `--force` to recompute outputs.
- Commands print progress lines suitable for Slicer background parsing.
- Commands exit non-zero when prerequisites are missing and `--generate-missing` is false.

Example:

```bash
timelapse prerequisites ensure /data/UCSF_single \
  --workflow microarchitecture \
  --subject SAMPLE355 \
  --site tibia \
  --generate-missing
```

This should discover or generate registration and common-region outputs, then stop. The analysis package can then consume those derivatives.

## Slicer Architecture

### Shared Slicer Adapters

`SlicerBoneImagingToolboxLib` should keep only Slicer-specific adapters:

```text
SlicerBoneImagingToolboxLib/
  slicer_nodes.py
  slicer_segments.py
  slicer_jobs.py
  slicer_tables.py
  package_status.py
  registry.py
  updater.py
```

Code that should move out of Slicer into package repos:

- derivative manifest model
- derivative discovery
- registration workflow wrappers
- common-region construction
- non-Slicer image/mask IO
- non-Slicer workflow dependency planning

Code that should remain in Slicer:

- MRML node selectors
- segmentation segment selectors
- temporary export/import adapters
- background job wrappers around package CLIs
- UI status rendering
- output node loading

### Public Slicer Modules

Each public module should have:

- `Scene` tab for loaded nodes
- `Batch` tab for dataset folders/manifests
- package status block
- prerequisite status block
- one main `Run` button
- background process execution for long jobs
- output loading after completion

Modules:

- `SegmentationHRpQCT`
- `RegisteredCommonRegion`
- `BoneMicroarchitecture`
- `PlateRodMorphometryHRpQCT`
- `TimelapsedHRpQCT`
- future `VoidSpace`
- future `FEA`
- future `Mechanoregulation`

### Registered Common Region Module

This module should become a thin Slicer UI over:

```python
timelapsedhrpqct.registration.run_registration_workflow()
timelapsedhrpqct.common_region.build_common_scan_region()
```

Scene mode:

- user selects loaded timepoint images
- user selects or derives scan masks
- module generates transforms and common-region masks
- outputs are loaded as Slicer transform nodes and labelmaps

Batch mode:

- user selects dataset root
- module discovers subject/site/session series
- user selects all series or one series
- module runs package CLI in background
- module parses progress lines
- module loads selected outputs optionally

### Timelapsed Module

The Timelapsed Slicer module should no longer own registration/common-region duplication.

Run behavior:

1. Discover selected subject/site/session series.
2. Inspect derivative manifests.
3. Resolve prerequisites.
4. If registration/common region are missing and generation is enabled, call package CLI/API to generate them.
5. Run Timelapsed analysis.
6. Load tables, masks, and QC outputs.

The UI should report whether prerequisites are:

- found from manifests
- found from legacy Timelapsed folders
- generated during this run
- missing and blocking execution

### Microarchitecture Module

The module should call `bone-microarchitecture`.

Scene mode:

- accepts grayscale image
- accepts bone/periosteal/endosteal/trab/cort masks from labelmap or segmentation node segment selection
- optional common scan-region mask
- displays measurements table
- loads generated maps

Batch mode:

- discovers grayscale images and masks from manifests or filenames
- discovers common-region masks from manifest
- runs per session in background
- writes `derivatives/Microarchitecture/manifest.json`

Common-region use:

```python
effective_trab = trab_mask & bone_segmentation & common_scan_region
effective_cort = cort_mask & bone_segmentation & common_scan_region
```

### Plate/Rod Module

Same pattern as microarchitecture.

Common-region use:

```python
effective_trab = trab_or_bone_mask & common_scan_region
```

## Dataset Layout

Preferred new layout:

```text
dataset/
  sourcedata/
  derivatives/
    Registration/
      manifest.json
      sub-SAMPLE001/
        site-tibia/
          transforms/
          qc/
    CommonRegion/
      manifest.json
      sub-SAMPLE001/
        site-tibia/
          reference_space/
          native_space/
            ses-1/
              masks/
            ses-2/
              masks/
    Microarchitecture/
      manifest.json
      sub-SAMPLE001/
        site-tibia/
          native_space/
            ses-1/
              measurements/
              maps/
    PlateRodMorphometry/
      manifest.json
    Timelapsed/
      manifest.json
    FEA/
      manifest.json
    Mechanoregulation/
      manifest.json
    VoidSpace/
      manifest.json
```

Compatibility:

- Existing `derivatives/TimelapsedHRpQCT` remains supported.
- Existing `RegisteredMicroarchitecture` remains readable for now.
- New writes should use family-specific derivative folders.
- Compatibility readers may produce in-memory records from legacy folders without rewriting data.

## Background Execution

Slicer should keep the UI responsive by launching package CLIs through `PythonSlicer` or the configured Python executable.

Requirements:

- CLI commands must print parseable progress lines.
- Slicer should not import long-running package workflows on the UI thread.
- Job JSON files may be used as argument payloads for complex runs.
- Slicer should support cancel by terminating the process.
- Partial outputs should be manifest-marked only after successful completion.
- Failed jobs should leave logs, not half-valid manifests.

Recommended progress line format:

```text
[derivative] family=CommonRegion subject=SAMPLE001 site=tibia step=register session=2 status=running
[derivative] family=CommonRegion subject=SAMPLE001 site=tibia step=write-manifest status=done path=...
```

## Migration Plan

### Phase 1: Move Contract to `timelapsed-hrpqct`

Move or reimplement these from `SlicerBoneImagingToolboxLib` into `timelapsedhrpqct`:

- derivative manifest model
- manifest read/write/discovery
- workflow dependency planning
- common mask helpers that do not depend on Slicer
- registration/common-region wrappers

Add tests in `TimelapsedHRpQCT`.

### Phase 2: Add Package-Level Registration and CommonRegion Workflows

Add public API and CLI commands:

- `timelapse registration run`
- `timelapse common-region run`
- `timelapse derivatives inspect`
- `timelapse prerequisites ensure`

Use existing registration, transform-chain, and dataset discovery internals.

### Phase 3: Refactor Slicer RegisteredCommonRegion

Replace Slicer-local registration/common-region services with calls to package APIs/CLI.

Keep:

- UI
- node adapters
- background process control
- output loading

Remove:

- duplicate manifest model
- duplicate common-region math
- duplicate registration wrappers

### Phase 4: Refactor Timelapsed Slicer Module

Make Timelapsed consume/generate prerequisites through the package-level planner.

Expected result:

- running Timelapsed from Slicer automatically generates missing registration and common-region derivatives when allowed
- existing derivative outputs are reused
- status panel shows provenance clearly

### Phase 5: Refactor Microarchitecture and Plate/Rod Batch Modes

Keep measurement packages focused.

Slicer batch mode should:

- discover inputs from manifests
- pass common-region masks to package functions
- write family-specific manifests
- load tables/maps after completion

### Phase 6: Future Derivative Consumers

Add FEA, mechanoregulation, and void-space modules against the same contract.

No future module should need to invent registration/common-region discovery.

## Backward Compatibility

Must preserve:

- existing `timelapse` CLI entry point
- existing core Timelapsed workflow commands
- existing config profiles
- existing `derivatives/TimelapsedHRpQCT` discovery
- existing subject/site/session naming
- existing Slicer module names where possible, unless UI name changes are explicitly migrated

Allowed changes:

- new derivative folders
- new manifests
- new CLI subcommands
- Slicer UI reorganization
- moving non-Slicer logic from Slicer extension into package repos

Deprecation rule:

- Legacy folders should be read for at least one release cycle after the new manifest writers exist.
- Warnings should say what was discovered and what the preferred new derivative family is.

## Testing Requirements

### `timelapsed-hrpqct`

Required tests:

- manifest round-trip
- manifest record filtering
- legacy Timelapsed discovery adapter
- registration workflow with mocked registration function
- transform composition record writing
- common scan-region semantics with differently sized but overlapping images
- native common-region mask output geometry
- CLI dry-run for registration/common-region
- CLI progress output
- prerequisite planner

### `bone-microarchitecture`

Required tests:

- common-region mask clips all biological masks
- trab/cort masks are intersected with bone segmentation
- GPU backend selection remains independent from Slicer
- output table schema remains stable

### `plate-rod-thinning`

Required tests:

- common-region clipping before plate/rod computation
- output map/table writing if CLI is added

### `SlicerBoneImagingToolbox`

Required tests:

- package status detects required package versions
- modules can import under Slicer Python
- Slicer adapters export selected segments correctly
- RegisteredCommonRegion launches background package CLI
- Timelapsed prerequisite status uses package planner
- Microarchitecture and Plate/Rod pass common-region paths/nodes correctly
- extension packaging includes only Slicer modules and adapters

## Release Strategy

1. Keep `feature/derivatives-overhaul` as the current Slicer-side foundation branch.
2. Create a new branch in `TimelapsedHRpQCT`: `feature/derivative-contract`.
3. Implement and release `timelapsed-hrpqct` with package-level derivative APIs and CLI commands.
4. Update `SlicerBoneImagingToolbox` to depend on that version.
5. Remove duplicated Slicer-local shared services after the package APIs are available.
6. Release Slicer extension update.
7. Then update `bone-microarchitecture` and `plate-rod-thinning` only where their package-level APIs need common-region/file-output improvements.

## Acceptance Criteria

The overhaul is complete when:

- `timelapse common-region run` can generate registration and common-region derivatives without Slicer.
- Slicer `Registered Common Region` calls the same package workflow.
- Slicer `TimelapsedHRpQCT` can generate or reuse registration/common-region derivatives automatically.
- Slicer `BoneMicroarchitecture` and `PlateRodMorphometryHRpQCT` can consume common-region derivatives from manifests.
- Existing command-line Timelapsed workflows still run.
- Existing legacy derivative folders can still be discovered.
- All derivative-producing workflows write manifests.
- No algorithmic registration/common-region logic remains duplicated in Slicer modules.

## Open Decisions

1. Whether the manifest contract should stay inside `timelapsed-hrpqct` long term or become a tiny standalone `bone-imaging-derivatives` package.
2. Whether `bone-microarchitecture` should gain its own CLI now or remain API-first for the next release.
3. Whether `RegisteredMicroarchitecture` legacy output folders should be migrated automatically or only read as compatibility inputs.
4. Whether Slicer should use `PythonSlicer` exclusively or allow a configured external Python environment for heavier batch runs.
5. Whether FEA and mechanoregulation should write into family-specific derivative folders immediately or first consume manifests read-only.

## Recommendation

Implement the next phase in `timelapsed-hrpqct` first. The current Slicer foundation branch is useful, but it should become a temporary consumer of the derivative architecture, not the owner of it. Once `timelapsed-hrpqct` exposes registration, common-region, manifest, and prerequisite APIs, the Slicer toolbox can be simplified substantially and command-line users will get the same functionality without Slicer.
