# Derivatives Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor SlicerBoneImagingToolbox so registration, common regions, microarchitecture, plate/rod, timelapsed analysis, and future FEA/mechanoregulation/void-space workflows share derivative manifests and reusable backend services.

**Architecture:** Add shared derivative, registration, common-region, and execution-planning services under `SlicerBoneImagingToolboxLib`. Public Slicer modules expose consistent Scene and Batch tabs while delegating algorithmic work to shared services or existing core packages. Existing `BoneMicroarchitecture` registered batch logic is migrated to shared services and then reduced to common-region consumption.

**Tech Stack:** 3D Slicer scripted modules, Python 3.12, SimpleITK, VTK/MRML, `timelapsedhrpqct`, `bone-microarchitecture`, `plate-rod-thinning`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-derivatives-overhaul-design.md`

## Global Constraints

- Keep `pre-derivatives-overhaul-2026-08-29` as the rollback tag.
- Use manifests as the primary contract for derivative discovery.
- Keep Scene and Batch mode backed by the same service functions.
- Do not copy Timelapsed registration algorithms into new modules; wrap shared `timelapsedhrpqct` functions.
- Common-region outputs represent scan/FOV overlap only and do not intersect biological masks.
- Preserve existing public module behavior until a replacement path is implemented and tested.
- Commit after each task with only the files relevant to that task.

---

### Task 1: Create Isolated Overhaul Worktree

**Files:**
- Modify: none
- Test: repository status and baseline focused tests

**Interfaces:**
- Consumes: clean `main` at or after commit `2a2efa7`
- Produces: branch `feature/derivatives-overhaul` in an isolated worktree

- [ ] **Step 1: Verify current repository state**

Run:

```bash
git -C /Users/matthias.walle/Documents/14_GitHub/active/SlicerBoneImagingToolbox status --short --branch
git -C /Users/matthias.walle/Documents/14_GitHub/active/SlicerBoneImagingToolbox tag --list pre-derivatives-overhaul-2026-08-29
```

Expected: `main...origin/main` with no modified files, and the rollback tag is listed.

- [ ] **Step 2: Create ignored worktree folder if needed**

Run:

```bash
cd /Users/matthias.walle/Documents/14_GitHub/active/SlicerBoneImagingToolbox
git check-ignore -q .worktrees || printf '\n.worktrees/\n' >> .gitignore
```

If `.gitignore` changes, commit it:

```bash
git add .gitignore
git commit -m "chore: ignore local worktrees"
git push
```

- [ ] **Step 3: Create the feature worktree**

Run:

```bash
cd /Users/matthias.walle/Documents/14_GitHub/active/SlicerBoneImagingToolbox
git worktree add .worktrees/derivatives-overhaul -b feature/derivatives-overhaul origin/main
```

- [ ] **Step 4: Run baseline focused tests**

Run:

```bash
cd /Users/matthias.walle/Documents/14_GitHub/active/SlicerBoneImagingToolbox/.worktrees/derivatives-overhaul
python3 -m pytest tests/test_microarchitecture_module.py tests/test_plate_rod_morphometry_module.py tests/test_toolbox_updater.py -q
```

Expected: tests pass or known pre-existing failures are recorded before implementation.

---

### Task 2: Add Derivative Manifest Library

**Files:**
- Create: `SlicerBoneImagingToolboxLib/derivatives.py`
- Test: `tests/test_derivatives_lib.py`

**Interfaces:**
- Produces: `DerivativeRecord`, `DerivativeManifest`, `write_manifest`, `read_manifest`, `discover_manifests`, `find_records`

- [ ] **Step 1: Write failing manifest tests**

Create `tests/test_derivatives_lib.py` with tests for:

```python
from pathlib import Path

from SlicerBoneImagingToolboxLib.derivatives import (
    DerivativeManifest,
    DerivativeRecord,
    discover_manifests,
    find_records,
    read_manifest,
    write_manifest,
)


def test_manifest_round_trip_preserves_records(tmp_path: Path) -> None:
    manifest = DerivativeManifest(
        workflow="CommonRegion",
        version="1",
        dataset_root=str(tmp_path),
        records=[
            DerivativeRecord(
                derivative="CommonRegion",
                role="scan_region_native_common",
                subject_id="SAMPLE001",
                site="tibia",
                session_id="1",
                stack_index=1,
                space="native",
                path="sub-SAMPLE001/site-tibia/native_space/ses-1/masks/mask.nii.gz",
                source="generated",
                metadata={"reference_session": "1"},
            )
        ],
    )
    path = tmp_path / "derivatives" / "CommonRegion" / "manifest.json"

    write_manifest(path, manifest)
    loaded = read_manifest(path)

    assert loaded.workflow == "CommonRegion"
    assert loaded.records[0].role == "scan_region_native_common"
    assert loaded.records[0].metadata["reference_session"] == "1"


def test_find_records_filters_by_subject_site_role(tmp_path: Path) -> None:
    manifest = DerivativeManifest(
        workflow="Registration",
        version="1",
        dataset_root=str(tmp_path),
        records=[
            DerivativeRecord("Registration", "transform_composed", "S1", "tibia", "1", 1, "reference", "a.tfm", "generated", {}),
            DerivativeRecord("Registration", "transform_pairwise", "S1", "tibia", "2", 1, "native", "b.tfm", "generated", {}),
            DerivativeRecord("Registration", "transform_composed", "S2", "radius", "1", 1, "reference", "c.tfm", "generated", {}),
        ],
    )

    matches = find_records(manifest, derivative="Registration", role="transform_composed", subject_id="S1", site="tibia")

    assert [record.path for record in matches] == ["a.tfm"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_derivatives_lib.py -q
```

Expected: import failure for `SlicerBoneImagingToolboxLib.derivatives`.

- [ ] **Step 3: Implement minimal manifest library**

Create `SlicerBoneImagingToolboxLib/derivatives.py` with frozen dataclasses, JSON serialization, recursive manifest discovery, and field filtering.

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_derivatives_lib.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add SlicerBoneImagingToolboxLib/derivatives.py tests/test_derivatives_lib.py
git commit -m "feat: add derivative manifest library"
```

---

### Task 3: Add Shared Mask and Image Adapter Helpers

**Files:**
- Create: `SlicerBoneImagingToolboxLib/image_io.py`
- Create: `SlicerBoneImagingToolboxLib/masks.py`
- Test: `tests/test_shared_image_mask_helpers.py`

**Interfaces:**
- Consumes: SimpleITK images and MRML-compatible file paths
- Produces: `read_image`, `read_mask`, `write_mask`, `scan_region_mask`, `clip_mask_to_region`, `assert_same_geometry`

- [ ] **Step 1: Write failing tests**

Create tests that verify:

```python
def test_scan_region_mask_matches_image_geometry():
    image = sitk.Image([4, 5, 6], sitk.sitkFloat32)
    image.SetSpacing((0.061, 0.061, 0.061))

    mask = scan_region_mask(image)

    assert mask.GetSize() == image.GetSize()
    assert mask.GetSpacing() == image.GetSpacing()
    assert set(np.unique(sitk.GetArrayFromImage(mask))) == {1}


def test_clip_mask_to_region_keeps_only_shared_support():
    mask = sitk.Image([3, 3, 1], sitk.sitkUInt8) + 1
    region = sitk.Image([3, 3, 1], sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(region)
    arr[:, 1, 1] = 1
    region = sitk.GetImageFromArray(arr)
    region.CopyInformation(mask)

    clipped = clip_mask_to_region(mask, region)

    assert int(sitk.GetArrayFromImage(clipped).sum()) == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_shared_image_mask_helpers.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement helpers**

Move equivalent logic out of `BoneMicroarchitecture.py` where possible:

- `_read_registered_series_image`
- `_registered_scan_region`
- `_clip_registered_mask_to_scan_region`
- `_resample_registered_mask`

Keep wrappers in `BoneMicroarchitectureLogic` temporarily so behavior remains unchanged.

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_shared_image_mask_helpers.py tests/test_microarchitecture_module.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add SlicerBoneImagingToolboxLib/image_io.py SlicerBoneImagingToolboxLib/masks.py tests/test_shared_image_mask_helpers.py HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py
git commit -m "feat: add shared image and mask helpers"
```

---

### Task 4: Add Shared Registration Service

**Files:**
- Create: `SlicerBoneImagingToolboxLib/registration.py`
- Test: `tests/test_registration_service.py`
- Modify: `HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py`

**Interfaces:**
- Consumes: ordered session records with `session_id`, `image`, `registration_mask`
- Produces: `RegistrationResult`, `TransformRecord`, `register_sequential_series`

- [ ] **Step 1: Write failing unit tests using mocked registration**

Test that `register_sequential_series`:

- creates adjacent pairs `ses-2 -> ses-1`, `ses-3 -> ses-2`
- composes transforms to baseline
- includes identity for baseline
- returns derivative records with `transform_pairwise` and `transform_composed` roles

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_registration_service.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement registration service**

Wrap:

```python
from timelapsedhrpqct.processing.registration import RegistrationSettings, register_images
from timelapsedhrpqct.processing.transform_chain import PairwiseTransform, compose_sequential_to_baseline, flatten_transform
```

Do not reimplement registration algorithms.

- [ ] **Step 4: Migrate BoneMicroarchitecture wrapper usage**

Replace direct registration helper internals in `BoneMicroarchitectureLogic` with calls to the shared service, while keeping public method names stable for existing tests.

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m pytest tests/test_registration_service.py tests/test_microarchitecture_module.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add SlicerBoneImagingToolboxLib/registration.py tests/test_registration_service.py HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py
git commit -m "feat: add shared registration service"
```

---

### Task 5: Add Shared Common-Region Service

**Files:**
- Create: `SlicerBoneImagingToolboxLib/common_region.py`
- Test: `tests/test_common_region_service.py`
- Modify: `HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py`

**Interfaces:**
- Consumes: images, composed transforms, reference session
- Produces: `build_common_scan_region`, common-space mask, native-space masks, derivative records

- [ ] **Step 1: Write failing tests for scan/FOV-only semantics**

Test that common region:

- intersects scan-region masks in reference space
- writes no `trab`, `cort`, `full`, or `seg` common masks
- returns native common-region masks per session
- labels roles as `scan_region_common` and `scan_region_native_common`

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_common_region_service.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement common-region service**

Move common-region logic from `BoneMicroarchitectureLogic._build_registered_common_regions` into the service. Keep file-path construction separate from image math.

- [ ] **Step 4: Replace BoneMicroarchitecture internal common-region implementation**

Use the shared service inside the existing registered mode so current functionality remains available during migration.

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m pytest tests/test_common_region_service.py tests/test_microarchitecture_module.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add SlicerBoneImagingToolboxLib/common_region.py tests/test_common_region_service.py HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py
git commit -m "feat: add shared common-region service"
```

---

### Task 6: Add Dependency Planning Service

**Files:**
- Create: `SlicerBoneImagingToolboxLib/workflow_planning.py`
- Test: `tests/test_workflow_planning.py`

**Interfaces:**
- Consumes: requested workflow, available derivative records, available raw inputs, settings
- Produces: `WorkflowStep`, `WorkflowPlan`, `resolve_workflow_plan`

- [ ] **Step 1: Write failing tests**

Test these cases:

- Timelapsed with no registration/common region plans `Registration`, `CommonRegion`, `Timelapsed`
- Timelapsed with registration only plans `CommonRegion`, `Timelapsed`
- Microarchitecture with common region already available plans only `Microarchitecture`
- Missing masks returns a blocked plan with explicit missing roles

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_workflow_planning.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement planning service**

Define explicit workflow dependencies:

```python
WORKFLOW_DEPENDENCIES = {
    "CommonRegion": ("Registration",),
    "Timelapsed": ("Registration", "CommonRegion"),
    "Microarchitecture": (),
    "PlateRodMorphometry": (),
    "FEA": (),
    "Mechanoregulation": ("Registration", "CommonRegion", "FEA"),
    "VoidSpace": (),
}
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_workflow_planning.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add SlicerBoneImagingToolboxLib/workflow_planning.py tests/test_workflow_planning.py
git commit -m "feat: add derivative workflow planning"
```

---

### Task 7: Add Registered Common Region Slicer Module

**Files:**
- Create: `HRpQCTTools/RegisteredCommonRegion/CMakeLists.txt`
- Create: `HRpQCTTools/RegisteredCommonRegion/RegisteredCommonRegion.py`
- Modify: `CMakeLists.txt`
- Modify: `toolbox_modules.json`
- Modify: `scripts/link_local_toolbox_modules.py`
- Test: `tests/test_registered_common_region_module.py`

**Interfaces:**
- Consumes: shared registration/common-region services
- Produces: public Slicer module with `Scene` and `Batch` tabs

- [ ] **Step 1: Write failing packaging and UI tests**

Test that:

- `CMakeLists.txt` includes `HRpQCTTools/RegisteredCommonRegion`
- `toolbox_modules.json` lists the module
- module classes exist: `RegisteredCommonRegion`, `RegisteredCommonRegionLogic`, `RegisteredCommonRegionWidget`, `RegisteredCommonRegionTest`
- UI setup contains `Scene` and `Batch`
- batch run uses `QProcess`

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_registered_common_region_module.py -q
```

Expected: packaging assertions fail because the module does not exist.

- [ ] **Step 3: Implement module scaffold**

Build a thin Slicer wrapper with:

- Scene tab for loaded baseline/follow-up volume nodes and optional registration masks
- Batch tab for dataset root, derivatives root, discover, selected rows, run
- background job entry point `--registered-common-region-job`
- manifest writing through `SlicerBoneImagingToolboxLib.derivatives`

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_registered_common_region_module.py tests/test_toolbox_updater.py -q
/Applications/Slicer.app/Contents/bin/PythonSlicer -m py_compile HRpQCTTools/RegisteredCommonRegion/RegisteredCommonRegion.py
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add HRpQCTTools/RegisteredCommonRegion CMakeLists.txt toolbox_modules.json scripts/link_local_toolbox_modules.py tests/test_registered_common_region_module.py
git commit -m "feat: add registered common region module"
```

---

### Task 8: Make BoneMicroarchitecture Consume Common Region

**Files:**
- Modify: `HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py`
- Modify: `tests/test_microarchitecture_module.py`
- Modify: `docs/tools/microarchitecture.md`

**Interfaces:**
- Consumes: optional common-region MRML node in Scene mode and optional manifest/path in Batch mode
- Produces: clipped masks passed to `bone_microarchitecture.compute_microarchitecture`

- [ ] **Step 1: Write failing tests**

Test that:

- Scene mode exposes `Common scan region mask`
- logic intersects `seg`, `full`, `trab`, and `cort` with the common-region mask before measurement
- registered batch tab is either removed or marked legacy after `RegisteredCommonRegion` exists
- measurement metadata records `common_region_path` or node name when used

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_microarchitecture_module.py -q
```

Expected: new assertions fail.

- [ ] **Step 3: Implement common-region input**

Add optional selector/path handling. Reuse `SlicerBoneImagingToolboxLib.masks.clip_mask_to_region`.

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_microarchitecture_module.py tests/test_common_region_service.py -q
/Applications/Slicer.app/Contents/bin/PythonSlicer -m pytest tests/test_microarchitecture_module.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py tests/test_microarchitecture_module.py docs/tools/microarchitecture.md
git commit -m "feat: let microarchitecture consume common regions"
```

---

### Task 9: Make PlateRodMorphometry Consume Common Region

**Files:**
- Modify: `HRpQCTTools/PlateRodMorphometryHRpQCT/PlateRodMorphometryHRpQCT.py`
- Modify: `tests/test_plate_rod_morphometry_module.py`
- Modify: `docs/tools/plate-rod-morphometry.md`

**Interfaces:**
- Consumes: optional common-region MRML node/path
- Produces: plate/rod analysis constrained to `trabecular_mask & common_region`

- [ ] **Step 1: Write failing tests**

Test that:

- UI exposes optional common-region mask
- logic clips bone and trabecular masks to common region
- metadata records common-region use

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_plate_rod_morphometry_module.py -q
```

Expected: new assertions fail.

- [ ] **Step 3: Implement common-region clipping**

Use shared helper:

```python
trab = clip_mask_to_region(trab, common_region)
bone = clip_mask_to_region(bone, common_region)
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_plate_rod_morphometry_module.py tests/test_shared_image_mask_helpers.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add HRpQCTTools/PlateRodMorphometryHRpQCT/PlateRodMorphometryHRpQCT.py tests/test_plate_rod_morphometry_module.py docs/tools/plate-rod-morphometry.md
git commit -m "feat: let plate rod analysis consume common regions"
```

---

### Task 10: Add Derivative Discovery to TimelapsedHRpQCT Slicer Wrapper

**Files:**
- Modify: `HRpQCTTools/TimelapsedHRpQCT/TimelapsedHRpQCT.py`
- Test: `tests/test_timelapsed_derivative_discovery.py`

**Interfaces:**
- Consumes: `Registration` and `CommonRegion` manifests
- Produces: Timelapsed prerequisite status and reuse/generate behavior

- [ ] **Step 1: Write failing tests**

Test that:

- Timelapsed discovers existing `Registration` manifest
- Timelapsed discovers existing `CommonRegion` manifest
- if common region is missing but registration exists, plan includes `CommonRegion`
- if both are missing, plan includes `Registration` then `CommonRegion`

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_timelapsed_derivative_discovery.py -q
```

Expected: import or assertion failure.

- [ ] **Step 3: Implement discovery/planning integration**

Use `discover_manifests` and `resolve_workflow_plan`. Keep Timelapsed UI behavior stable and add prerequisite status display.

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_timelapsed_derivative_discovery.py tests/test_geodesic_contour_integration.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add HRpQCTTools/TimelapsedHRpQCT/TimelapsedHRpQCT.py tests/test_timelapsed_derivative_discovery.py
git commit -m "feat: let timelapsed reuse derivative prerequisites"
```

---

### Task 11: Add Public Derivatives Documentation

**Files:**
- Create: `docs/derivatives.md`
- Modify: `README.md`
- Test: `tests/test_derivatives_documentation.py`

**Interfaces:**
- Consumes: final manifest field names from earlier tasks
- Produces: user-facing derivative contract documentation

- [ ] **Step 1: Write failing documentation test**

Test that `docs/derivatives.md` documents:

- `Registration`
- `CommonRegion`
- `scan_region_native_common`
- Scene mode
- Batch mode
- dependency generation

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_derivatives_documentation.py -q
```

Expected: file missing or content assertions fail.

- [ ] **Step 3: Write concise documentation**

Create `docs/derivatives.md` with examples of manifest roles and module consumption.

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_derivatives_documentation.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add docs/derivatives.md README.md tests/test_derivatives_documentation.py
git commit -m "docs: describe derivative workflow contracts"
```

---

### Task 12: Final Verification and Integration Review

**Files:**
- Modify: files needed for fixes found during verification
- Test: focused and Slicer Python test suites

**Interfaces:**
- Consumes: all previous tasks
- Produces: reviewed branch ready for merge or PR

- [ ] **Step 1: Run focused system Python tests**

Run:

```bash
python3 -m pytest \
  tests/test_derivatives_lib.py \
  tests/test_shared_image_mask_helpers.py \
  tests/test_registration_service.py \
  tests/test_common_region_service.py \
  tests/test_workflow_planning.py \
  tests/test_registered_common_region_module.py \
  tests/test_microarchitecture_module.py \
  tests/test_plate_rod_morphometry_module.py \
  tests/test_timelapsed_derivative_discovery.py \
  tests/test_derivatives_documentation.py \
  -q
```

- [ ] **Step 2: Run Slicer Python compile and focused tests**

Run:

```bash
/Applications/Slicer.app/Contents/bin/PythonSlicer -m py_compile \
  HRpQCTTools/RegisteredCommonRegion/RegisteredCommonRegion.py \
  HRpQCTTools/BoneMicroarchitecture/BoneMicroarchitecture.py \
  HRpQCTTools/PlateRodMorphometryHRpQCT/PlateRodMorphometryHRpQCT.py \
  HRpQCTTools/TimelapsedHRpQCT/TimelapsedHRpQCT.py

/Applications/Slicer.app/Contents/bin/PythonSlicer -m pytest \
  tests/test_registered_common_region_module.py \
  tests/test_microarchitecture_module.py \
  tests/test_plate_rod_morphometry_module.py \
  tests/test_timelapsed_derivative_discovery.py \
  -q
```

- [ ] **Step 3: Run manual smoke test in Slicer**

Use loaded sample nodes:

- run `Registered Common Region` Scene mode for two timepoints
- load native common-region masks
- run `Bone Microarchitecture` with common-region mask selected
- run `Plate/Rod Morphometry` with common-region mask selected

Expected: no geometry mismatch errors, output tables/maps load, and logs report common-region clipping.

- [ ] **Step 4: Review git diff**

Run:

```bash
git diff origin/main...HEAD --stat
git diff origin/main...HEAD -- SlicerBoneImagingToolboxLib HRpQCTTools tests docs
```

Check that registration/common-region code is no longer duplicated in `BoneMicroarchitecture.py`.

- [ ] **Step 5: Commit verification fixes**

If verification required fixes:

```bash
git add <changed-files>
git commit -m "test: verify derivatives overhaul integration"
```

- [ ] **Step 6: Push branch**

Run:

```bash
git push -u origin feature/derivatives-overhaul
```
