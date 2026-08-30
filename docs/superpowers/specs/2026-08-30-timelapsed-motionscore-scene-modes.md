# Timelapsed And MotionScore Scene Modes

## Goal

Add first-class Slicer scene modes to Timelapsed HR-pQCT and Motion Scoring while keeping their existing batch modes intact. Scene mode is for loaded MRML nodes and immediate review; batch mode is for folder discovery, reproducible cohort runs, and derivative output.

## Product Contract

Each Slicer-facing analysis module should expose the same mental model:

- `Scene`: select nodes already loaded in Slicer, run or review one case or a small set of explicit timepoints.
- `Batch`: select a dataset root, discover subjects/sessions/artifacts, run outside the UI thread, and write derivative outputs.

The HR-pQCT subsection can keep domain-specific names, but the layout should feel consistent across modules.

## Timelapsed Scene Mode

Scene mode should support explicit longitudinal timepoints selected from the Slicer scene. The user chooses subject/site labels once, then adds timepoint rows with:

- session id
- input scalar volume
- optional full mask
- optional trab mask
- optional cort mask
- optional segmentation mask

When the user runs scene mode, the Slicer adapter exports selected nodes to a temporary MIDS-like input folder under the chosen results root and launches the existing `timelapsed-hrpqct` workflow through the existing background `QProcess` runner. The package remains the owner of mask generation, registration, common-region construction, remodelling, and derivative writing.

Scene mode may create transient NIfTI files because loaded MRML nodes are not guaranteed to have stable source paths. Those exported files must be scoped to the scene run folder and must not replace the no-raw-copy behavior in batch mode.

## MotionScore Scene Mode

Scene mode should support scoring or manual grading of one loaded scalar volume. The user chooses:

- scalar volume
- scan id
- subject id
- site
- session id
- model profile
- reviewer
- run mode: AI prediction or manual review only

The Slicer adapter should use a package-level scene input path whenever available. Until the core package exposes a direct scene-array API, the Slicer module may export the selected volume to a temporary scene-run image and pass that through a small MIDS-like folder to the existing MotionScore CLI. Outputs must be the same prediction/review/index tables used by batch mode so review UI remains shared.

## Derivative Behavior

Scene mode outputs should write under:

```text
<results_root>/derivatives/<Family>/scene_runs/<run_id>/
```

Batch mode outputs should keep using:

```text
<dataset_root>/derivatives/<Family>/
```

Both modes must write or refresh derivative manifests when the package supports it. If a scene run only produces legacy package files, the Slicer adapter may refresh the existing review/index table and defer manifest writing to the package.

## Responsiveness

Long-running scene actions must not block Slicer. Timelapsed scene runs use the existing Timelapsed `QProcess` runner. MotionScore AI scene prediction uses the existing MotionScore `QProcess` runner. Pure review-state updates can remain in process if they are fast.

## Compatibility

Existing batch behavior must remain available and command-line compatible. Existing output folders must remain readable. Existing package APIs and CLIs should not be broken by adding scene wrappers.

## Testing

Testing should cover:

- pure helper path planning and validation without Slicer imports
- source-level smoke tests that the Qt widgets expose `Scene` and `Batch` sections
- command construction for background scene runs
- Slicer Python import smoke for both modules
- existing focused toolbox tests
