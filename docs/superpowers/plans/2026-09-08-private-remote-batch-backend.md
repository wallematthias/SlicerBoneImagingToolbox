# Private Remote Batch Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the public Batch Processor maintainable while allowing private ARC/SLURM server execution through an optional backend adapter.

**Architecture:** Public code owns the Batch Processor UI, job snapshots, local execution, artifact loading, and a small backend registry contract. Private code owns server discovery, job submission, polling, cancellation, and file syncing by registering an optional backend at runtime.

**Tech Stack:** Python scripted Slicer modules, Qt widgets, pytest source-level and unit tests, optional private Python adapters.

**Spec:** In-chat design approved on 2026-09-08: queued jobs remain immutable while UI tool/profile switching is allowed; public repo exposes a backend interface; ARC/SLURM details move behind private registration.

## Global Constraints

- Do not regress local batch processing.
- Queued and running jobs must use the tool, profile, row, roots, and force setting captured when they were queued.
- Tool/profile switching is allowed while jobs are active, but must not mutate queued/running jobs.
- Public repository must not require or document ARC-specific configuration.
- Private remote functionality must be optional and discovered only when a backend is available.

---

### Task 1: Immutable Queue UI Behavior

**Files:**
- Modify: `IOTools/BatchProcessor/BatchProcessor.py`
- Modify: `tests/test_batch_processor_module.py`

**Interfaces:**
- Consumes: existing `BatchProcessorWidget._batch_job_for_row(row_index)` snapshot dictionary.
- Produces: `_job_description(job: dict) -> str` and active-run UI logging that references snapshotted jobs, not current combo state.

- [ ] **Step 1: Write the failing test**

```python
def test_active_batches_allow_selector_changes_but_jobs_keep_snapshots() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "self.toolCombo.enabled = True" in source
    assert "self.profileCombo.enabled = True" in source
    assert "def _job_description(self, job):" in source
    assert 'job.get("tool")' in source
    assert 'job.get("profile")' in source
    assert "Tool/profile change will not affect already queued jobs." in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/matthias.walle/Documents/14_GitHub/active/bone-imaging-derivatives/src python3 -m pytest tests/test_batch_processor_module.py::test_active_batches_allow_selector_changes_but_jobs_keep_snapshots -q`

Expected: FAIL because the helper/logging text does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Update `_on_tool_changed()` and `_on_profile_changed()` so they no longer return early during an active batch. They should repopulate/reanalyze the visible table for new user actions and log that already queued jobs are unaffected. Keep `_batch_job_for_row()` as the immutable snapshot source.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/matthias.walle/Documents/14_GitHub/active/bone-imaging-derivatives/src python3 -m pytest tests/test_batch_processor_module.py::test_active_batches_allow_selector_changes_but_jobs_keep_snapshots -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add IOTools/BatchProcessor/BatchProcessor.py tests/test_batch_processor_module.py docs/superpowers/plans/2026-09-08-private-remote-batch-backend.md
git commit -m "Allow safe batch workflow switching"
```

### Task 2: Public Backend Registry Boundary

**Files:**
- Create: `SlicerBoneImagingToolboxLib/batch_backends.py`
- Modify: `IOTools/BatchProcessor/BatchProcessor.py`
- Modify: `tests/test_remote_batch_backend.py`

**Interfaces:**
- Produces: `BatchBackendProtocol`, `LocalBatchBackend`, `register_batch_backend(backend)`, `available_batch_backends()`, and `get_batch_backend(key)`.
- Consumes: current local execution and existing remote backend helper logic.

- [ ] **Step 1: Write failing tests**

```python
def test_batch_backend_registry_exposes_local_backend() -> None:
    from SlicerBoneImagingToolboxLib.batch_backends import available_batch_backends, get_batch_backend
    assert "local" in available_batch_backends()
    assert get_batch_backend("local").label == "Local"

def test_batch_backend_registry_accepts_private_backend() -> None:
    from SlicerBoneImagingToolboxLib.batch_backends import available_batch_backends, get_batch_backend, register_batch_backend
    class PrivateBackend:
        key = "arc-slurm"
        label = "ARC / SLURM"
    register_batch_backend(PrivateBackend())
    assert available_batch_backends()["arc-slurm"].label == "ARC / SLURM"
    assert get_batch_backend("arc-slurm").key == "arc-slurm"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/Users/matthias.walle/Documents/14_GitHub/active/bone-imaging-derivatives/src python3 -m pytest tests/test_remote_batch_backend.py::test_batch_backend_registry_exposes_local_backend tests/test_remote_batch_backend.py::test_batch_backend_registry_accepts_private_backend -q`

Expected: FAIL because `batch_backends.py` does not exist.

- [ ] **Step 3: Implement registry**

Add the registry module with a local backend and tolerant optional import hook for private backend registration. The import hook should try known private modules and silently continue when unavailable.

- [ ] **Step 4: Run tests to verify they pass**

Run the same two tests.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add SlicerBoneImagingToolboxLib/batch_backends.py IOTools/BatchProcessor/BatchProcessor.py tests/test_remote_batch_backend.py
git commit -m "Add optional batch backend registry"
```

### Task 3: Move Server Controls Behind Backend Availability

**Files:**
- Modify: `IOTools/BatchProcessor/BatchProcessor.py`
- Modify: `SlicerBoneImagingToolboxLib/remote_batch.py`
- Modify: `tests/test_batch_processor_module.py`
- Modify: `tests/test_remote_batch_backend.py`

**Interfaces:**
- Consumes: `available_batch_backends()`.
- Produces: UI backend dropdown populated from registered backends; server root/resource controls visible only for remote-capable backends.

- [ ] **Step 1: Write failing tests**

```python
def test_batch_processor_populates_backend_combo_from_registry() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "available_batch_backends()" in source
    assert 'self.backendCombo.addItem(backend.label, backend.key)' in source
    assert 'server_selected = self._selected_backend_key() != "local"' in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/matthias.walle/Documents/14_GitHub/active/bone-imaging-derivatives/src python3 -m pytest tests/test_batch_processor_module.py::test_batch_processor_populates_backend_combo_from_registry -q`

Expected: FAIL because the combo is still hard-coded.

- [ ] **Step 3: Implement UI integration**

Populate backend choices from the registry. Keep Local always available. Only show backend selection if more than one backend is registered or `SLICER_BONE_BATCH_BACKEND_MODE=server` is set. Show server path/resources for non-local backends.

- [ ] **Step 4: Run test to verify it passes**

Run the same test.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add IOTools/BatchProcessor/BatchProcessor.py SlicerBoneImagingToolboxLib/remote_batch.py tests/test_batch_processor_module.py tests/test_remote_batch_backend.py
git commit -m "Gate server batch UI behind registered backends"
```

### Task 4: Verification

**Files:**
- Test only.

**Interfaces:**
- Consumes: all tasks above.
- Produces: green focused test suite and clean branch ready for review.

- [ ] **Step 1: Run focused tests**

Run: `PYTHONPATH=/Users/matthias.walle/Documents/14_GitHub/active/bone-imaging-derivatives/src python3 -m pytest tests/test_batch_processor_module.py tests/test_remote_batch_backend.py tests/test_toolbox_updater.py tests/test_centralized_setup_ui.py -q`

Expected: PASS.

- [ ] **Step 2: Compile touched files**

Run: `PYTHONPATH=/Users/matthias.walle/Documents/14_GitHub/active/bone-imaging-derivatives/src python3 -m py_compile IOTools/BatchProcessor/BatchProcessor.py SlicerBoneImagingToolboxLib/batch_backends.py SlicerBoneImagingToolboxLib/remote_batch.py`

Expected: exit code 0.

- [ ] **Step 3: Push feature branch**

```bash
git push -u origin feature/private-remote-batch-backend
```

Expected: branch exists on remote for private-backend follow-up.
