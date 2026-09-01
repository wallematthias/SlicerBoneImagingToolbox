# Timelapsed And MotionScore Scene Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scene-mode adapters to Timelapsed HR-pQCT and Motion Scoring so loaded Slicer nodes can be processed without losing the existing batch workflows.

**Architecture:** Add small pure-Python helper modules for scene-run path planning and validation, then wire top-level `Scene` and `Batch` tabs into the existing Slicer modules. Scene mode exports selected MRML nodes to scoped temporary run folders and launches the existing package CLIs through each module's current background runner.

**Tech Stack:** Python 3.10+, pytest, 3D Slicer Python/Qt, SimpleITK/Slicer volume storage APIs, existing `timelapsed-hrpqct` and `motionscore` CLIs.

**Spec:** `docs/superpowers/specs/2026-08-30-timelapsed-motionscore-scene-modes.md`

## Global Constraints

- Core algorithms live in Python packages, not Slicer modules.
- Slicer modules are adapters around package APIs.
- `Scene` mode selects loaded Slicer nodes.
- `Batch` mode discovers a dataset root and writes derivative outputs.
- Long-running scene actions must not block Slicer.
- Existing Timelapsed and MotionScore batch behavior must remain available.
- Scene exports are scoped to scene-run folders and do not change batch no-raw-copy behavior.
- Tests must include pure helper tests and Slicer module source/import smoke checks.

---

### Task 1: Timelapsed Scene Helper

**Files:**
- Create: `SlicerBoneImagingToolboxLib/timelapsed_scene.py`
- Test: `tests/test_timelapsed_scene_mode.py`

**Interfaces:**
- Produces `TimelapsedSceneTimepoint`, `TimelapsedScenePlan`, `build_timelapsed_scene_plan(...)`, `timelapsed_scene_run_args(...)`.
- Consumes no Slicer imports.

- [ ] **Step 1: Write failing helper tests**

Add tests that construct two timepoints and assert deterministic scene paths:

```python
from pathlib import Path

from SlicerBoneImagingToolboxLib.timelapsed_scene import (
    TimelapsedSceneTimepoint,
    build_timelapsed_scene_plan,
    timelapsed_scene_run_args,
)


def test_timelapsed_scene_plan_paths(tmp_path):
    plan = build_timelapsed_scene_plan(
        results_root=tmp_path,
        subject_id="SAMPLE001",
        site="tibia",
        timepoints=[
            TimelapsedSceneTimepoint(session_id="ses-1", image_node_id="v1", full_mask_node_id="f1"),
            TimelapsedSceneTimepoint(session_id="ses-2", image_node_id="v2", full_mask_node_id="f2"),
        ],
        run_id="scene-test",
    )

    assert plan.input_root == tmp_path / "derivatives" / "Timelapsed" / "scene_runs" / "scene-test" / "input"
    assert plan.timepoints[0].image_path.name == "sub-SAMPLE001_ses-1_site-tibia_image.nii.gz"
    assert plan.timepoints[1].full_mask_path.name == "sub-SAMPLE001_ses-2_site-tibia_mask-full.nii.gz"


def test_timelapsed_scene_run_args_include_existing_pipeline_options(tmp_path):
    plan = build_timelapsed_scene_plan(
        results_root=tmp_path,
        subject_id="SAMPLE001",
        site="radius",
        timepoints=[
            TimelapsedSceneTimepoint(session_id="ses-1", image_node_id="v1"),
            TimelapsedSceneTimepoint(session_id="ses-2", image_node_id="v2"),
        ],
        run_id="abc",
    )

    args = timelapsed_scene_run_args(plan, mode="regular", config_path=Path("/tmp/config.toml"))

    assert args[:2] == ["run", str(plan.input_root)]
    assert "--output-root" in args
    assert str(plan.output_root) in args
    assert "--config" in args
```

- [ ] **Step 2: Run failing tests**

Run: `python3 -m pytest tests/test_timelapsed_scene_mode.py -q`
Expected: import failure because the helper does not exist.

- [ ] **Step 3: Implement the helper**

Create frozen dataclasses:

```python
@dataclass(frozen=True)
class TimelapsedSceneTimepoint:
    session_id: str
    image_node_id: str
    full_mask_node_id: str = ""
    trab_mask_node_id: str = ""
    cort_mask_node_id: str = ""
    seg_mask_node_id: str = ""
    image_path: Path | None = None
    full_mask_path: Path | None = None
    trab_mask_path: Path | None = None
    cort_mask_path: Path | None = None
    seg_mask_path: Path | None = None
```

`build_timelapsed_scene_plan` sanitizes subject/site/session tokens, requires at least two timepoints with image node ids, and assigns file paths under `derivatives/Timelapsed/scene_runs/<run_id>/input/sub-<subject>/site-<site>/native_space/<session>/`.

`timelapsed_scene_run_args` returns the existing CLI shape:

```python
["run", str(plan.input_root), "--output-root", str(plan.output_root), "--mode", mode, "--config", str(config_path)]
```

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/test_timelapsed_scene_mode.py -q`

- [ ] **Step 5: Commit**

Run:

```bash
git add SlicerBoneImagingToolboxLib/timelapsed_scene.py tests/test_timelapsed_scene_mode.py
git commit -m "feat: add Timelapsed scene planning helper"
```

### Task 2: MotionScore Scene Helper

**Files:**
- Create: `SlicerBoneImagingToolboxLib/motionscore_scene.py`
- Test: `tests/test_motionscore_scene_mode.py`

**Interfaces:**
- Produces `MotionScoreScenePlan`, `build_motionscore_scene_plan(...)`, `motionscore_scene_predict_args(...)`.
- Consumes no Slicer imports.

- [ ] **Step 1: Write failing helper tests**

Add tests:

```python
from SlicerBoneImagingToolboxLib.motionscore_scene import (
    build_motionscore_scene_plan,
    motionscore_scene_predict_args,
)


def test_motionscore_scene_plan_uses_derivative_scene_folder(tmp_path):
    plan = build_motionscore_scene_plan(
        results_root=tmp_path,
        scan_id="scan-1",
        subject_id="SAMPLE001",
        site="tibia",
        session_id="ses-1",
        volume_node_id="node-1",
        run_id="scene-test",
    )

    assert plan.input_root.name == "input"
    assert "scene_runs" in str(plan.input_root)
    assert plan.image_path.name == "sub-SAMPLE001_ses-1_site-tibia_scan-scan-1_image.nii.gz"


def test_motionscore_scene_predict_args_can_run_manual_only(tmp_path):
    plan = build_motionscore_scene_plan(
        results_root=tmp_path,
        scan_id="scan-1",
        subject_id="SAMPLE001",
        site="radius",
        session_id="ses-1",
        volume_node_id="node-1",
        run_id="scene-test",
    )

    args = motionscore_scene_predict_args(
        plan,
        model_root=tmp_path / "models",
        model_id="base-v1",
        manual_only=True,
        confidence_threshold=75,
        slice_step=1,
        device="auto",
    )

    assert args[0] == str(plan.volume_npz_path)
    assert "--manual-only" in args
    assert "--output-root" in args
```

- [ ] **Step 2: Run failing tests**

Run: `python3 -m pytest tests/test_motionscore_scene_mode.py -q`
Expected: import failure because the helper does not exist.

- [ ] **Step 3: Implement the helper**

Create a frozen dataclass with `results_root`, `run_root`, `input_root`, `output_root`, `image_path`, `scan_id`, `subject_id`, `site`, `session_id`, and `volume_node_id`. Build CLI args matching current `MotionScoreHRpQCTWidget.onRunPredict`, adding `--scan-id` for the generated scan id and omitting `--device` when `auto`.

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/test_motionscore_scene_mode.py -q`

- [ ] **Step 5: Commit**

Run:

```bash
git add SlicerBoneImagingToolboxLib/motionscore_scene.py tests/test_motionscore_scene_mode.py
git commit -m "feat: add MotionScore scene planning helper"
```

### Task 3: Timelapsed Scene UI

**Files:**
- Modify: `HRpQCTTools/TimelapsedHRpQCT/TimelapsedHRpQCT.py`
- Test: `tests/test_timelapsed_scene_mode.py`

**Interfaces:**
- Consumes `TimelapsedSceneTimepoint`, `build_timelapsed_scene_plan`, and `timelapsed_scene_run_args`.
- Produces widget attributes `timelapsedModeTabs`, `sceneTimepointTable`, `sceneRunButton`, and method `_on_run_scene_pipeline`.

- [ ] **Step 1: Add source-level UI tests**

Assert the module source contains:

```python
assert "self.timelapsedModeTabs" in source
assert "Scene" in source
assert "Batch" in source
assert "def _on_run_scene_pipeline" in source
assert "build_timelapsed_scene_plan" in source
```

- [ ] **Step 2: Run focused tests**

Run: `python3 -m pytest tests/test_timelapsed_scene_mode.py -q`
Expected: source-level assertions fail.

- [ ] **Step 3: Wrap existing controls in a Batch tab**

Inside `_build_ui`, create `self.timelapsedModeTabs = qt.QTabWidget()`. Add a `Scene` page for new controls and a `Batch` page containing the existing dependency, dataset, parse, advanced, run, and review widgets. Preserve existing widget names and signal connections.

- [ ] **Step 4: Add scene controls and export/run method**

Add subject/site/results fields, an editable timepoint table, add/remove row buttons, and `sceneRunButton`. Implement `_on_run_scene_pipeline` so it builds a plan, exports nodes via `slicer.util.saveNode`, creates the override config, then calls `self._run(timelapsed_scene_run_args(...))`.

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest tests/test_timelapsed_scene_mode.py tests/test_package_status.py -q
python3 -m py_compile HRpQCTTools/TimelapsedHRpQCT/TimelapsedHRpQCT.py SlicerBoneImagingToolboxLib/timelapsed_scene.py
```

- [ ] **Step 6: Commit**

Run:

```bash
git add HRpQCTTools/TimelapsedHRpQCT/TimelapsedHRpQCT.py tests/test_timelapsed_scene_mode.py
git commit -m "feat: add Timelapsed scene mode tab"
```

### Task 4: MotionScore Scene UI

**Files:**
- Modify: `HRpQCTTools/MotionScoreHRpQCT/MotionScoreHRpQCT.py`
- Test: `tests/test_motionscore_scene_mode.py`

**Interfaces:**
- Consumes `build_motionscore_scene_plan` and `motionscore_scene_predict_args`.
- Produces widget attributes `motionScoreModeTabs`, `sceneVolumeSelector`, `sceneRunButton`, and method `onRunScenePredict`.

- [ ] **Step 1: Add source-level UI tests**

Assert the module source contains:

```python
assert "self.motionScoreModeTabs" in source
assert "Scene" in source
assert "Batch" in source
assert "def onRunScenePredict" in source
assert "build_motionscore_scene_plan" in source
```

- [ ] **Step 2: Run focused tests**

Run: `python3 -m pytest tests/test_motionscore_scene_mode.py -q`
Expected: source-level assertions fail.

- [ ] **Step 3: Wrap existing run/review controls in a Batch tab**

Create `self.motionScoreModeTabs = qt.QTabWidget()`. Add `Scene` and `Batch` pages while preserving existing widgets and signal connections. Move current dataset prediction and review controls into the `Batch` page.

- [ ] **Step 4: Add scene scoring controls**

Add volume selector, scan id, subject id, site, session id, results root, model profile reuse, run mode, reviewer, and run button. Implement `onRunScenePredict` to export the selected volume to the helper plan NPZ path, launch the toolbox scene runner through `_run_cli`, then refresh review outputs on completion.

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest tests/test_motionscore_scene_mode.py tests/test_fea_mechanoregulation_modules.py tests/test_package_status.py -q
python3 -m py_compile HRpQCTTools/MotionScoreHRpQCT/MotionScoreHRpQCT.py SlicerBoneImagingToolboxLib/motionscore_scene.py
```

- [ ] **Step 6: Commit**

Run:

```bash
git add HRpQCTTools/MotionScoreHRpQCT/MotionScoreHRpQCT.py tests/test_motionscore_scene_mode.py
git commit -m "feat: add MotionScore scene mode tab"
```

### Task 5: Documentation And Smoke Verification

**Files:**
- Modify: `docs/tools/timelapsed-hrpqct.md`
- Modify: `docs/tools/motion-scoring.md`

**Interfaces:**
- Consumes scene-mode UI from Tasks 3 and 4.
- Produces user-facing docs and verification evidence.

- [ ] **Step 1: Update docs**

Add short `Scene Mode` and `Batch Mode` sections to both tool docs. State that scene mode uses loaded Slicer nodes and batch mode uses dataset roots.

- [ ] **Step 2: Run focused tests**

Run:

```bash
PYTHONPATH=/Users/matthias.walle/Documents/14_GitHub/active/bone-imaging-derivatives/src python3 -m pytest tests/test_timelapsed_scene_mode.py tests/test_motionscore_scene_mode.py tests/test_fea_mechanoregulation_modules.py tests/test_fea_batch_discovery.py tests/test_package_status.py -q
```

- [ ] **Step 3: Run Slicer import smoke**

Run:

```bash
/Applications/Slicer.app/Contents/bin/PythonSlicer - <<'PY'
import sys
from pathlib import Path
root = Path("/Users/matthias.walle/Documents/14_GitHub/active/SlicerBoneImagingToolbox/.worktrees/derivatives-overhaul")
sys.path.insert(0, str(root / "HRpQCTTools" / "TimelapsedHRpQCT"))
sys.path.insert(0, str(root / "HRpQCTTools" / "MotionScoreHRpQCT"))
import TimelapsedHRpQCT
import MotionScoreHRpQCT
print("scene imports ok", bool(TimelapsedHRpQCT), bool(MotionScoreHRpQCT))
PY
```

- [ ] **Step 4: Commit**

Run:

```bash
git add docs/tools/timelapsed-hrpqct.md docs/tools/motion-scoring.md
git commit -m "docs: describe Timelapsed and MotionScore scene modes"
```
