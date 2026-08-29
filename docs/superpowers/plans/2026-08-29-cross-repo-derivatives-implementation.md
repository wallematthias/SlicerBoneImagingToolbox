# Cross-Repository Derivatives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared derivative contract and integrate it across Timelapsed, microarchitecture, plate/rod, and the Slicer toolbox so every Slicer-facing workflow has consistent Scene and Batch modes and batch outputs reuse/write the same `derivatives/<Family>` format.

**Architecture:** Create `bone-imaging-derivatives` as the dependency-light contract package. Update algorithm packages to write their own derivative manifests through that package. Refactor Slicer code so it becomes a scene/batch adapter over package APIs and CLIs rather than the owner of derivative orchestration.

**Tech Stack:** Python 3.10+, setuptools, pytest, SimpleITK in Timelapsed/Slicer workflows, NumPy/SciPy in measurement packages, 3D Slicer Python for extension tests.

**Spec:** `docs/superpowers/specs/2026-08-29-cross-repo-derivatives-architecture.md`

## Global Constraints

- Core algorithms live in Python packages, not Slicer modules.
- Slicer modules are adapters around package APIs.
- Every reusable workflow writes a derivative manifest.
- Batch mode and Slicer scene mode call the same package-level workflow functions.
- Common scan region means scan/FOV overlap only, not biological mask intersection.
- Biological masks are clipped by common region at analysis time.
- Every Slicer-facing package should support both a one-case API and a batch derivative-writing workflow.
- Command-line and Slicer behavior must remain equivalent.
- New code should prefer explicit manifest records over filename inference.
- Existing Timelapsed outputs remain readable during migration.
- Do not touch existing local `BoneMechanoregulation` changes in this implementation pass.

---

### Task 1: Worktree And Baseline Setup

**Files:**
- Modify: `TimelapsedHRpQCT/.gitignore`
- Modify: `bone-microarchitecture/.gitignore`
- Modify: `bone-plate-rod-thinning/.gitignore`
- Create worktree/repo: `active/bone-imaging-derivatives`
- Create worktree: `TimelapsedHRpQCT/.worktrees/derivative-contract`
- Create worktree: `bone-microarchitecture/.worktrees/derivative-batch`
- Create worktree: `bone-plate-rod-thinning/.worktrees/derivative-batch`

**Interfaces:**
- Produces isolated paths used by later tasks.

- [ ] **Step 1: Add `.worktrees/` ignores where missing**

Append this line to each repo `.gitignore` if absent:

```text
.worktrees/
```

- [ ] **Step 2: Commit ignore updates**

Run in each changed repo:

```bash
git add .gitignore
git commit -m "chore: ignore local worktrees"
```

- [ ] **Step 3: Create package and worktrees**

Use:

```bash
git worktree add .worktrees/derivative-contract -b feature/derivative-contract
git worktree add .worktrees/derivative-batch -b feature/derivative-batch
```

Create `active/bone-imaging-derivatives` as a new git repo on branch `main`.

- [ ] **Step 4: Run baseline tests**

Run focused existing tests:

```bash
python3 -m pytest -q
```

For Slicer:

```bash
/Applications/Slicer.app/Contents/bin/PythonSlicer -m pytest -q
```

Expected: either pass or record pre-existing failures before proceeding.

### Task 2: Create `bone-imaging-derivatives`

**Files:**
- Create: `bone-imaging-derivatives/pyproject.toml`
- Create: `bone-imaging-derivatives/README.md`
- Create: `bone-imaging-derivatives/LICENSE`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/__init__.py`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/families.py`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/roles.py`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/records.py`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/manifest.py`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/discovery.py`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/layout.py`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/planning.py`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/progress.py`
- Create: `bone-imaging-derivatives/src/bone_imaging_derivatives/compatibility.py`
- Create: `bone-imaging-derivatives/tests/`

**Interfaces:**
- Produces `DerivativeRecord`, `DerivativeManifest`, `read_manifest`, `write_manifest`, `discover_manifests`, `find_records`, `resolve_workflow_plan`, `DerivativeProgressEvent`, `format_progress_event`, and `parse_progress_event`.
- Consumed by later package and Slicer tasks.

- [ ] **Step 1: Write failing tests for manifest round-trip, discovery, layout, planning, progress, and compatibility**

Use pytest tests that import from `bone_imaging_derivatives` and assert the interfaces above.

- [ ] **Step 2: Verify tests fail because package is missing/incomplete**

Run:

```bash
python3 -m pytest -q
```

- [ ] **Step 3: Implement minimal package**

Use only standard library dependencies. Store relative paths in manifests when possible. Keep role/family values as string constants plus validation helpers.

- [ ] **Step 4: Verify package tests pass**

Run:

```bash
python3 -m pytest -q
python3 -m py_compile src/bone_imaging_derivatives/*.py
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: add bone imaging derivative contract"
```

### Task 3: Update `timelapsed-hrpqct`

**Files:**
- Modify: `TimelapsedHRpQCT/.worktrees/derivative-contract/pyproject.toml`
- Create: `src/timelapsedhrpqct/registration/`
- Create: `src/timelapsedhrpqct/common_region/`
- Modify: `src/timelapsedhrpqct/cli.py`
- Add tests under `tests/`

**Interfaces:**
- Consumes `bone_imaging_derivatives`.
- Produces `run_registration_workflow`, `run_registration_batch`, `build_common_scan_region`, `run_common_region_batch`, and CLI commands.

- [ ] **Step 1: Write failing tests**

Test:

- registration batch writes `Registration/manifest.json`
- common-region batch writes `CommonRegion/manifest.json`
- common region uses scan/FOV support only
- legacy Timelapsed discovery still produces compatibility records
- CLI dry-run and progress output work

- [ ] **Step 2: Verify tests fail**

Run focused tests:

```bash
python3 -m pytest tests/test_derivatives_*.py tests/test_common_region_*.py tests/test_registration_*.py -q
```

- [ ] **Step 3: Implement public package APIs**

Wrap existing `processing.registration`, `processing.transform_chain`, and image IO functions. Do not rewrite registration algorithms.

- [ ] **Step 4: Implement CLI commands**

Add subcommands:

```text
timelapse derivatives inspect
timelapse registration run
timelapse common-region run
timelapse prerequisites ensure
```

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest -q
python3 -m py_compile src/timelapsedhrpqct/**/*.py
```

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "feat: add derivative registration and common region workflows"
```

### Task 4: Update `bone-microarchitecture`

**Files:**
- Modify: `bone-microarchitecture/.worktrees/derivative-batch/pyproject.toml`
- Create: `src/bone_microarchitecture/batch.py`
- Create or modify CLI module if absent
- Add tests under `tests/`

**Interfaces:**
- Consumes `bone_imaging_derivatives`.
- Produces `run_microarchitecture_batch(dataset_root, ..., use_common_region=True)` and optional `bone-microarchitecture run-batch`.

- [ ] **Step 1: Write failing tests**

Test:

- batch discovers image/masks from manifest records
- common-region mask clips biological masks before measurement
- output writes `derivatives/Microarchitecture/manifest.json`
- CLI run-batch works on a tiny NIfTI or NumPy-backed fixture if image IO exists

- [ ] **Step 2: Verify tests fail**

Run:

```bash
python3 -m pytest tests/test_batch*.py -q
```

- [ ] **Step 3: Implement batch workflow**

Keep existing one-case API unchanged. Use the derivative package only for dataset discovery, paths, manifests, and progress events.

- [ ] **Step 4: Verify**

Run:

```bash
python3 -m pytest -q
python3 -m py_compile src/bone_microarchitecture/*.py
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: add derivative batch microarchitecture workflow"
```

### Task 5: Update `plate-rod-thinning`

**Files:**
- Modify: `bone-plate-rod-thinning/.worktrees/derivative-batch/pyproject.toml`
- Create: package batch module
- Create or modify CLI module if absent
- Add tests under `tests/`

**Interfaces:**
- Consumes `bone_imaging_derivatives`.
- Produces `run_plate_rod_batch(dataset_root, ..., use_common_region=True)` and optional `plate-rod-thinning run-batch`.

- [ ] **Step 1: Write failing tests**

Test:

- batch discovers trab/bone masks from manifest records
- common-region mask clips analysis mask
- output writes `derivatives/PlateRodMorphometry/manifest.json`

- [ ] **Step 2: Verify tests fail**

Run:

```bash
python3 -m pytest tests/test_batch*.py -q
```

- [ ] **Step 3: Implement batch workflow**

Do not change the core skeletonization API unless required by tests.

- [ ] **Step 4: Verify**

Run:

```bash
python3 -m pytest -q
python3 -m py_compile plate_rod_thinning/**/*.py
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: add derivative batch plate rod workflow"
```

### Task 6: Refactor Slicer Integration

**Files:**
- Modify: `SlicerBoneImagingToolbox/.worktrees/derivatives-overhaul/SlicerBoneImagingToolboxLib/package_status.py`
- Modify: `SlicerBoneImagingToolbox/.worktrees/derivatives-overhaul/SlicerBoneImagingToolboxLib/derivatives.py`
- Modify: `SlicerBoneImagingToolbox/.worktrees/derivatives-overhaul/SlicerBoneImagingToolboxLib/workflow_planning.py`
- Modify: `HRpQCTTools/RegisteredCommonRegion/RegisteredCommonRegion.py`
- Modify: `HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py`
- Modify: `HRpQCTTools/PlateRodMorphometryHRpQCT/PlateRodMorphometryHRpQCT.py`
- Modify: `HRpQCTTools/TimelapsedHRpQCT/TimelapsedHRpQCT.py`
- Add/update Slicer tests

**Interfaces:**
- Consumes released or local editable `bone-imaging-derivatives`, `timelapsed-hrpqct`, `bone-microarchitecture`, and `plate-rod-thinning`.
- Produces thin Slicer adapters with consistent Scene and Batch modes.

- [ ] **Step 1: Write failing Slicer tests**

Test:

- package status includes `bone-imaging-derivatives`
- derivative discovery imports from shared package or compatibility shim
- RegisteredCommonRegion launches package CLI
- Timelapsed prerequisite status uses shared planner
- Microarchitecture/PlateRod batch call package batch workflows

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/Applications/Slicer.app/Contents/bin/PythonSlicer -m pytest tests/test_registered_common_region_module.py tests/test_microarchitecture_module.py tests/test_plate_rod_morphometry_module.py tests/test_timelapsed_derivative_discovery.py tests/test_package_status.py -q
```

- [ ] **Step 3: Implement adapters**

Keep Slicer-specific node export/import and UI. Move non-Slicer derivative logic behind imports from `bone_imaging_derivatives` and package CLIs.

- [ ] **Step 4: Verify**

Run:

```bash
/Applications/Slicer.app/Contents/bin/PythonSlicer -m pytest -q
python3 -m pytest tests/test_derivatives_lib.py tests/test_toolbox_updater.py tests/test_slicer_extension_packaging.py -q
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: consume shared derivative package workflows"
```

### Task 7: Cross-Repo Integration Verification

**Files:**
- Modify docs only if verification exposes needed usage notes.

**Interfaces:**
- Consumes all previous task outputs.
- Produces final green state report.

- [ ] **Step 1: Install local editable stack**

Use the relevant Python/Slicer Python executables:

```bash
python3 -m pip install -e active/bone-imaging-derivatives
python3 -m pip install -e active/TimelapsedHRpQCT/.worktrees/derivative-contract
python3 -m pip install -e active/bone-microarchitecture/.worktrees/derivative-batch
python3 -m pip install -e active/bone-plate-rod-thinning/.worktrees/derivative-batch
```

- [ ] **Step 2: Run package tests**

Run:

```bash
python3 -m pytest -q
```

in each repo/worktree.

- [ ] **Step 3: Run Slicer tests**

Run:

```bash
/Applications/Slicer.app/Contents/bin/PythonSlicer -m pytest -q
```

in the Slicer worktree.

- [ ] **Step 4: Run a tiny derivative smoke test**

Use synthetic masks to verify:

- `Registration` and `CommonRegion` manifests can be discovered
- `Microarchitecture` can consume common region
- `PlateRodMorphometry` can consume common region
- Slicer package status reports required packages

- [ ] **Step 5: Commit docs or fixes**

Commit any integration fixes with targeted messages.
