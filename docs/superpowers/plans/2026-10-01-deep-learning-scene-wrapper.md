# HR-pQCT U-Net Scene Wrapper Implementation Plan

**Superseded:** This is the earlier external-runner draft. The user's subsequent
minimal-core request replaces its Python/repository/model/output selectors with
the snapshot worker and fixed-default core. See the core repository's
`docs/superpowers/plans/2026-10-01-minimal-inference.md` and current tool guide.
The verification record below describes the earlier draft, not the current build.

> **For agentic workers:** Execute inline using superpowers:executing-plans. User guidance prohibits routine subagent delegation. Steps use checkbox syntax.

**Goal:** Finish the existing scene wrapper after synchronizing its feature branch with main.

**Architecture:** Keep inference, normalization, and morphology in the external Bonelab runner. The toolbox validates one AIM input, launches a cancellable QProcess, and imports native binary compartment masks into one segmentation with a derived union.

**Tech Stack:** Python, PythonQt QProcess, SimpleITK, NumPy, ScancoIOLib, Slicer MRML, pytest.

**Spec:** User-approved design in this conversation: separate U-Net module; AIM-backed inputs; external Python/repository/weights; cortical and trabecular masks plus full union; no automatic standard-contouring peel.

## Global Constraints

- Preserve existing work and current main's standard-contouring changes.
- No model or weights copied into the toolbox; no automatic dependency installation.
- Never write derivatives into the input AIM directory. Allocate a fresh run directory under a selected output root or the Slicer temporary directory.
- Reject missing, empty, overlapping, or geometrically incompatible output compartments. No automatic resampling or morphology.
- CPU and CUDA only; published local model label is `radius_tibia_final`. No knee-validation claim.
- Inference reads original AIM, not edited scene voxels. Reject transformed references and references whose geometry changed before import.
- Leave integration/release to the user; retain the recoverable pre-sync stash.

### Task 1: Safe runner contract and compartment validation

**Files:** `SlicerBoneImagingToolboxLib/deep_learning_segmentation_scene.py`, `tests/test_deep_learning_segmentation_scene.py`.

**Interfaces:** `build_deep_learning_segmentation_command(..., output_directory, device='cpu')`; `expected_deep_learning_mask_paths(image_path, output_directory)`; `validate_deep_learning_compartments(cort, trab, reference)` returns full/trab/cort SimpleITK images.

- [x] Add tests for explicit output, glob escaping, valid model labels and AIM suffixes, missing output files, binary values, geometry, empty/overlapping compartments, and an unchanged union with no peel.
- [x] Run `python -m pytest tests/test_deep_learning_segmentation_scene.py -q`; confirm new behavioral failures.
- [x] Build commands with `--output-directory`, escaped exact basename, safe model labels; require absent expected outputs. Use shared `assert_same_geometry`, validate native labels, and derive union only.
- [x] Run the same tests, plus a tiny external runner exercised through a real subprocess.

### Task 2: Async Slicer execution and scene import

**Files:** `HRpQCTTools/DeepLearningSegmentationHRpQCT/DeepLearningSegmentationHRpQCT.py`, scene lifecycle tests and a headless Slicer smoke script.

**Interfaces:** Logic launches the helper's command; native AIM reader supplies the validated compartment images; one reference-linked segmentation receives Full/Trabecular/Cortical segments.

- [x] Test process lifecycle via Qt/Slicer smoke: successful child, fast completion, nonzero exit, cancellation, startup failure, and reload cleanup.
- [x] Set process ownership before start, drain final output, account for crash status, and detach callbacks on widget cleanup.
- [x] Use a fresh per-run directory; freeze the reference node, model and geometry; reject removed/changed references. Load both AIM masks with `ScancoIOLib.aim_io.read_aim(..., scaling='native')`, validate before creating MRML nodes, and remove partial nodes on failure.
- [x] Verify import arrays and geometry with asymmetric synthetic masks in real Slicer. Check the external environment and report unavailable upstream dependencies without silently installing them.

### Task 3: Documentation and regression checks

**Files:** `docs/tools/deep-learning-segmentation.md`, existing module registry/CMake/linking/docs edits.

- [x] Document weights placement, external environment, original-AIM semantics, output folders, CPU/CUDA, union versus tissue segmentation, and radius/tibia scope.
- [x] Run focused wrapper/registry/linking/label-algebra/contouring tests; compile edited Python; `git diff --check`; confirm main is an ancestor and changes remain isolated to this feature.
- [x] Report test evidence and any real-model validation limitation; do not merge/publish automatically.

## Verification Record

- Feature branch fast-forwarded to main `60fdc76`; draft restored from backup stash `7df12f1`. Main's newer label-export geometry checks retained.
- Full Python suite: **630 passed**. Compilation and diff whitespace checks passed.
- Real Slicer fixture smoke: `DEEP_LEARNING_SLICER_SMOKE_PASSED`; native AIM arrays imported unchanged, repeated widget runs used distinct output directories, cleanup stopped a running child, changed geometry/source symlinks were rejected, and a main-preloaded package did not hide the feature helper.
- One independent review found directory-glob, cached-package and symlink-basename defects; each was reproduced and addressed with regression coverage.
- Published local `radius_tibia_final` weights loaded into the upstream architecture and produced finite CPU output of shape `(1, 2, 32, 40)`. This is a weights-compatibility smoke, **not full AIM inference or accuracy validation**.
- Full trained-runner scan inference was not run: Slicer Python lacks Bonelab/`vtkbone`, and its installed PyTorch reports an incompatible NumPy bridge. No dependencies or settings were changed automatically.
- Full suite exposed an existing main packaging omission: `fea_workflow_labels.py` was absent from the shared CMake install list. The feature checkout now includes it. A linking-test fixture also hard-coded this feature checkout as stale; it now uses a distinct synthetic checkout name.
- Implementation remains uncommitted in the feature worktree; merge/push/release are not performed. Backup stash retained.
