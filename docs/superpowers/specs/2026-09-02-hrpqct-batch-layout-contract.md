# Batch Processor Layout Contract

## Goal

Define a stable dataset, derivative, and batch-job contract for HR-pQCT workflows so Slicer modules, command-line tools, and future remote execution backends can share the same inputs and outputs without another layout overhaul.

## Design Position

Batch Processor should require a normalized dataset. Loose historical filenames remain acceptable for discovery and normalization, but not as the stable execution contract. The normalization helper may rename data into the contract, write a manifest of every rename, and undo the rename after processing if the user wants the original filenames back.

The layout contract lives in `bone-imaging-derivatives`. Processing packages and Slicer modules must not invent output folders. They ask the shared package to discover artifacts, plan jobs, write derivative paths, write manifests, and decide whether compatible outputs already exist.

## Dataset Layout

Raw XCT files live under subject, session, and modality folders:

```text
dataset/
  sub-001/
    ses-001/
      xct/
        sub-001_ses-001_voi-radiusleft_xct.AIM
        sub-001_ses-001_voi-radiusleft_xct.json
```

The `voi` token identifies the scanned volume of interest. It must preserve side and anatomical specificity when known:

- `radiusleft`
- `radiusright`
- `tibialeft`
- `tibiaright`
- `kneeleft`
- `kneeright`
- `tibiaproxleft`
- `tibiaproxright`

Generic VOIs such as `radius`, `tibia`, and `knee` are allowed only when the input truly does not distinguish left and right. Discovery must not collapse `RL` and `RR` into one generic radius series.

## Session Rules

The normalized dataset always uses `ses-*`.

If a timepoint is known, preserve the meaningful label after normalization:

- `Y00` or `00` may normalize to `ses-001` for anonymized/shareable exports, with the original label stored in sidecar metadata.
- The sidecar should store `original_session_label`, for example `Y00`, `Y04`, or `Y08`.
- Batch tables may display both the normalized session and original label when useful.

If only one timepoint is present and no session can be inferred, the normalization helper assigns `ses-001`. This keeps single-timepoint tools such as contouring, microarchitecture, plate/rod morphometry, FEA, and motion scoring usable without forcing the user to invent a longitudinal label.

If multiple timepoints are present and no stable order can be inferred, Batch Processor must stop at the normalization/preflight table and ask the user to assign sessions before running longitudinal tools. Single-session tools may still run row-by-row after the user confirms the mapping.

When AIM metadata are available, discovery should use `py-aimio`/`aim_info` to validate and enrich session assignment. Filename parsing remains the first pass because it also supports NIfTI, NRRD, MHA, and other non-AIM files, but AIM metadata should be used to check:

- patient or measurement identifier
- original scan date/time
- scanner site code or VOI
- scanner type and voxel spacing
- measurement index
- dimensions and slice count

For longitudinal series, chronological scan date/time is the preferred session ordering signal. If filename-derived order and AIM metadata-derived order disagree, the naming helper must flag the row for review instead of silently choosing one.

## Multistack Rules

A physical multistack acquisition is represented by the same subject, session, VOI, and a stack token:

```text
sub-001/ses-001/xct/sub-001_ses-001_voi-tibiaright_stack-01_xct.AIM
sub-001/ses-001/xct/sub-001_ses-001_voi-tibiaright_stack-02_xct.AIM
```

Rules:

- `stack_index` is a manifest field, not just a filename string.
- Missing stack tokens mean single-stack input.
- A one-stack multistack-style input is still valid and behaves as stack 1.
- Registration may operate per stack when physical stacks are separate.
- Tools may emit fused images or fused masks when useful, but should not duplicate raw AIM data unless the workflow genuinely needs an explicit derived image.
- Downstream analysis may consume per-stack artifacts, fused artifacts, or both, depending on the tool profile.

Some multistack acquisitions are stored inside one AIM file rather than as separate files. In that case discovery should create virtual stack records instead of copying or splitting the source image by default. Profile-level defaults define stack depth, for example XtremeCT II commonly uses 168 slices and XtremeCT I commonly uses 110 slices. Timelapse and other stack-aware tools can materialize temporary stack views during processing only when required by an external backend, but stable derivatives should avoid duplicate raw image copies.

## Derivative Layout

All generated outputs live under:

```text
dataset/derivatives/<DerivativeFamily>/
```

Session-level outputs:

```text
derivatives/<DerivativeFamily>/sub-001/ses-001/xct/
```

Subject-level or series-level outputs:

```text
derivatives/<DerivativeFamily>/sub-001/xct/
```

The layout must not emit:

- `site-*` directories
- `native_space` directories
- `reference_space` directories
- `common_space` directories
- derivative-family-specific nested layouts such as `RegisteredMicroarchitecture`

Space, VOI, stack, and coordinate-reference information belong in filenames and manifest records.

## Derivative Families

The expected public families are:

- `ImportedContours`: imported scanner, Scanco, IPL, or manually curated masks such as `TRAB_MASK`, `CORT_MASK`, `CRTX_MASK`, and `BLCK_MASK`
- `BoneContours`: generated bone segmentation and ROI masks from the contouring package
- `Registration`: transforms and registration diagnostics
- `CommonRegion`: common scan/FOV region masks derived from registration
- `Timelapse`: remodelling maps, tables, and trajectory outputs
- `Microarchitecture`: measurement tables and parameter maps
- `PlateRodMorphometry`: plate/rod maps, skeletons, and tables
- `FEA`: material/model images, solver inputs, solver outputs, and mechanical fields
- `Mechanoregulation`: combined remodelling and mechanical stimulus outputs
- `MotionScoring`: motion scores, diagnostic images, and review tables

Motion scoring is the least perfect fit because it is a quality-control tool rather than a derivative consumer. It should still write outputs under `derivatives/MotionScoring/sub-*/ses-*/xct/` and participate in Batch Processor as a single-session tool.

## Filename Pattern

Generated derivative filenames use:

```text
sub-<subject>_ses-<session>_voi-<voi>[_stack-##]_desc-<description>_<suffix>.<ext>
```

Examples:

```text
derivatives/BoneContours/sub-001/ses-001/xct/sub-001_ses-001_voi-radiusleft_desc-seg_mask.AIM
derivatives/BoneContours/sub-001/ses-001/xct/sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM
derivatives/ImportedContours/sub-001/ses-001/xct/sub-001_ses-001_voi-radiusleft_desc-trab_mask.AIM
derivatives/Microarchitecture/sub-001/ses-001/xct/measurements/sub-001_ses-001_voi-radiusleft_measurements.csv
derivatives/Microarchitecture/sub-001/ses-001/xct/maps/sub-001_ses-001_voi-radiusleft_map-tb-th.nii.gz
```

Transforms use:

```text
derivatives/Registration/sub-001/ses-002/xct/pairwise/sub-001_ses-002_voi-radiusleft_stack-01_from-ses-002_to-ses-001_pairwise.tfm
derivatives/Registration/sub-001/ses-003/xct/baseline/sub-001_ses-003_voi-radiusleft_stack-01_from-ses-003_to-ses-001_baseline.tfm
```

Common scan region uses:

```text
derivatives/CommonRegion/sub-001/xct/masks/sub-001_voi-radiusleft_stack-01_mask-scan-region_common.nii.gz
derivatives/CommonRegion/sub-001/ses-001/xct/masks/sub-001_ses-001_voi-radiusleft_stack-01_mask-scan-region_native_common.nii.gz
```

## Manifest Contract

Every derivative family writes:

```text
derivatives/<DerivativeFamily>/manifest.json
```

Each record must include:

- `derivative`
- `role`
- `subject_id`
- `session_id`
- `site` or `voi`
- `stack_index`
- `modality`
- `space`
- `path`
- `source`
- `content_type`
- `inputs`
- `settings_hash`
- `metadata`
- `coordinate_reference` when the artifact is not in native image space

The manifest is the source of truth. Filenames are a recovery and convenience layer.

Manifest paths must be portable. Records should store paths relative to the dataset root whenever the artifact lives under the dataset root. Absolute paths may be retained only in optional provenance/debug metadata. Copying the dataset root to a new machine must not break artifact discovery.

## Discovery Policy

There are two discovery modes:

1. Normalization discovery
   - scans loose historical filenames
   - reads sidecars and AIM metadata when available
   - recognizes STRAMBO, Calgary, Nina, Scanco, IPL, and MIDS-like patterns
   - lets the user correct subject, session, VOI, stack, and role
   - writes a rename manifest and optional private identity sidecar

2. Batch execution discovery
   - requires the normalized dataset layout
   - reads manifests first
   - may use normalized filenames only as a recovery path
   - does not recursively treat arbitrary old folders as valid batch inputs

This split keeps user import forgiving while keeping batch execution stable.

When multiple contour sources are available for the same subject, session, VOI, stack, and role, tools should prefer them in this order:

1. `ImportedContours`
2. `BoneContours`
3. other compatible segmentation/ROI derivatives

The rationale is that imported scanner/IPL/manual contours are often curated upstream and should not be overwritten by generated contours unless the user explicitly selects a different source.

## Tool Requirements

### Contouring

Input: raw XCT image.

Outputs:

- bone segmentation
- full/periosteal ROI
- trabecular ROI
- cortical ROI
- optional additional ROI masks

Contours should write `BoneContours`. Imported scanner/IPL/manual masks should normalize into `ImportedContours`.

### Timelapse

Input: at least two sessions for one subject and VOI.

Outputs:

- `Registration`
- `CommonRegion`
- `Timelapse`

Timelapse requires registration and common region. It may reuse existing transforms and common-region artifacts. It must not generate bone contours. Users prepare masks with Contouring or imported IPL/Scanco masks.

Minimum Timelapse prerequisites:

- at least two session images for the same subject and VOI
- bone segmentation for each session
- full/periosteal ROI for each session
- registration ROI, usually the full/periosteal ROI
- trabecular and cortical ROIs for profiles that stratify or report trabecular/cortical remodelling

If profile-required ROIs are missing, Timelapse should stop with a prerequisite message and direct the user to Contouring or ImportedContours normalization. It should not generate missing contours itself.

If only one timepoint is present, Timelapse is not runnable, but other tools remain runnable.

### Microarchitecture

Input: one session image plus segmentation and ROI masks. Optional common region.

Modes:

- unregistered single-session analysis
- registered/common-region analysis when common-region artifacts exist

Outputs:

- measurement table
- maps for all generated map-backed parameters

It should use common region automatically when the selected profile or registered mode requires it.

Minimum Microarchitecture prerequisites:

- grayscale/BMD image
- bone segmentation
- full/periosteal ROI
- trabecular ROI
- cortical ROI for cortical metrics
- optional common region for registered/common-region mode

Total measures such as `Tt.BMD` use the full/periosteal ROI clipped by the common region when one is active. Trabecular and cortical measures use the corresponding ROI intersected with bone segmentation where appropriate.

### Plate/Rod Morphometry

Input: bone segmentation and trabecular or selected ROI mask. Optional common region.

It should match Microarchitecture behavior: use common region automatically when present and relevant, without exposing a confusing top-level checkbox.

### FEA

Input: model/material label image or workflow-specific source artifacts.

Outputs:

- model/material image
- solver inputs
- solver outputs
- mechanical field maps

XCT profile discovery may target material label maps. Other profiles can declare their own input artifact queries.

### Mechanoregulation

Input: Timelapse remodelling output plus matching FEA mechanical field.

Outputs:

- stimulus/remodelling association maps
- summary tables

Scene mode can use loaded remodelling and FEA nodes. Batch mode should discover matching derivative records.

### Motion Scoring

Input: one XCT image.

Outputs:

- score
- diagnostic PNG or image
- optional CSV/table

It can run from Batch Processor as a single-session quality-control job. It does not need the same ROI/mask dependency model as analysis tools.

## Batch Processor Module

Batch Processor is a stable dataset processor, not a loose-file importer.

Flow:

1. Select dataset root.
2. Verify that it is normalized.
3. If not normalized, send the user to Dataset Naming Helper.
4. Analyze available subjects, sessions, VOIs, stacks, derivatives, and missing prerequisites.
5. Select tool or tool sequence.
6. Select profile.
7. Select execution backend: local or server.
8. Queue jobs.
9. Monitor progress.
10. Load selected outputs.

The module should not replace expert batch tabs immediately. Tool-specific batch tabs remain useful for focused workflows and debugging. Batch Processor becomes the recommended stable path for full-dataset processing.

## Batch Table Interaction Contract

Every current and future batch mode should follow the same table interaction model, whether it appears in Batch Processor or inside a tool-specific expert tab.

Required controls:

- dataset root selector
- tool selector when the batch module can run more than one tool
- profile selector
- `Register` checkbox when the selected tool supports both registered and unregistered execution
- `Skip existing` checkbox, enabled by default
- per-row action button
- `Run all` button below the table
- load action for completed rows
- queue/cancel behavior for pending and running rows

The discovered table is dynamic. Columns are derived from the selected tool, profile, and registration mode instead of being hardcoded globally.

Examples:

- Contouring: action, subject, session, VOI, stack, image, output status
- Microarchitecture unregistered: action, subject, session, VOI, image, segmentation, analysis ROIs, existing measurements, status
- Microarchitecture registered: action, subject, VOI, sessions, registration/common-region status, analysis ROIs, existing measurements, status
- Timelapse: action, subject, VOI, sessions, registration ROI, segmentation, analysis ROIs, pair mode, existing remodelling outputs, status
- Plate/Rod Morphometry: action, subject, session, VOI, segmentation, analysis ROI, existing maps/table, status
- FEA: action, subject, session, VOI, material/model source, load profile, existing solver outputs, status
- Mechanoregulation: action, subject, VOI, remodelling map, mechanical field, existing mechanoregulation outputs, status
- Motion Scoring: action, subject, session, VOI, image, existing score, status

Row action states:

- `Run`: prerequisites are available and no compatible output exists, or recomputation is requested.
- `Queued`: row is waiting behind an active job.
- `Cancel`: row is queued or running and can be removed/interrupted.
- `Load`: compatible outputs exist and can be loaded.
- `Missing`: required inputs are absent or ambiguous; hovering or selecting the row should show the missing roles.
- `Review`: automatic parsing found conflicting identity, VOI, session, stack, or metadata evidence.

`Run all` queues every runnable row and skips rows that are already loadable when `Skip existing` is enabled. Finished rows switch to `Load`. Failed rows keep the error message attached to the row, not only in a global log.

The `Register` checkbox controls row grouping:

- unchecked: one row is usually one subject-session-VOI-stack case
- checked: one row is usually one subject-VOI-stack series across sessions

When registration is enabled, common-region generation is implied. There should not be a second top-level `Use common region` checkbox for tools where common region is a consequence of registered analysis.

## Execution Backend Contract

Backends share a common job description:

```json
{
  "dataset_root": "/path/to/dataset",
  "tool": "Microarchitecture",
  "profile": "xct2-standard",
  "filters": {
    "subject_id": "001",
    "session_id": "001",
    "voi": "radiusleft",
    "stack_index": null
  },
  "inputs": [],
  "outputs": [],
  "settings": {},
  "execution_backend": "local"
}
```

Public backends:

- `local`: runs with local Python or Slicer Python.

Private or future backends:

- `ssh`
- `slurm`
- `arc`

Server configuration can remain private. The public interface only needs a backend slot and a job model that a private adapter can consume.

Remote execution must write manifests and progress events in the same format as local execution. Slicer should be able to reconnect to a running job, read progress, and load results after completion.

## Reuse And Skip Rules

Every batch job computes a settings hash from:

- input record IDs
- relevant software versions
- profile name
- explicit settings
- coordinate/reference choices
- common-region use

If compatible outputs exist, the job status is `loadable` or `reused`. If `skip_existing` is enabled, it does not recompute them. If recomputation is forced, new outputs overwrite or supersede records according to the derivative family policy, but they must not silently create parallel ambiguous outputs.

## Stability Commitments

This contract is the forward path:

- Batch Processor requires normalized datasets.
- Loose filenames are for normalization only.
- Derivatives are written through `bone-imaging-derivatives`.
- VOI is encoded in filenames and manifest metadata, not folder levels.
- Session is always present in normalized raw data.
- Single-timepoint data defaults to `ses-001` when no session can be inferred.
- Multistack data keeps explicit `stack_index`.
- Timelapse does not generate masks.
- Common region means scan/FOV overlap, not biological mask intersection.
- Tool-specific Slicer modules keep scene mode and expert controls.
- Remote execution is a backend choice, not a separate layout.
