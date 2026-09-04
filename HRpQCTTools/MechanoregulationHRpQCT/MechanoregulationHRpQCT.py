from __future__ import annotations

import csv
import importlib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path

import ctk
import numpy as np
import qt
import SimpleITK as sitk
import slicer
import vtk

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


MODULE_VERSION = "0.1.0"
CORE_REQUIREMENT = "bone-mechanoregulation"
MIN_CORE_VERSION = "0.1.5"
TOOLBOX_ROOT = Path(__file__).resolve().parents[2]


def _active_repositories_root(toolbox_root):
    root = Path(toolbox_root).resolve()
    if root.parent.name == ".worktrees":
        return root.parent.parent.parent
    return root.parent


CORE_LOCAL_REPO = _active_repositories_root(TOOLBOX_ROOT) / "BoneMechanoregulation"


def _use_local_core_checkout():
    value = str(os.environ.get("SLICER_BONE_MECHANOREGULATION_SOURCE", "") or "").strip()
    if value:
        return True
    try:
        settings = slicer.app.settings()
        return str(settings.value("BoneMechanoregulation/useSourceCheckout", "false") or "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


def _prefer_local_core(force_reload=False):
    if not _use_local_core_checkout():
        return
    if not (CORE_LOCAL_REPO / "bonemechreg").is_dir():
        return
    repo_text = str(CORE_LOCAL_REPO)
    if repo_text in sys.path:
        sys.path.remove(repo_text)
    sys.path.insert(0, repo_text)
    loaded = sys.modules.get("bonemechreg")
    if loaded is None and not force_reload:
        return
    loaded_path = Path(getattr(loaded, "__file__", "") or ".").resolve() if loaded is not None else None
    local_root = CORE_LOCAL_REPO.resolve()
    if force_reload or loaded_path is None or local_root not in loaded_path.parents:
        for name in list(sys.modules):
            if name == "bonemechreg" or name.startswith("bonemechreg."):
                sys.modules.pop(name, None)


_prefer_local_core(force_reload=True)

METRIC_KEYS = ("CCR", "OR_F_ratio", "OR_F", "OR_R_ratio", "OR_R")
COUNT_KEYS = ("n_sampled_voxels", "n_formation", "n_resorption", "n_quiescence")
METRIC_LABELS = {
    "CCR": "CCR",
    "OR_F_ratio": "OR_F ratio",
    "OR_F": "OR_F change (%)",
    "OR_R_ratio": "OR_R ratio (decreasing SED)",
    "OR_R": "OR_R change (%)",
    "n_sampled_voxels": "Sampled voxels",
    "n_formation": "Formation voxels",
    "n_resorption": "Resorption voxels",
    "n_quiescence": "Quiescent voxels",
}


def _version_tuple(version_text):
    values = []
    for token in str(version_text or "0").replace("-", ".").split("."):
        if token.isdigit():
            values.append(int(token))
    return tuple(values or [0])


def resolve_timelapsed_root(path):
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"TimelapsedHRpQCT root does not exist: {root}")
    if (root / "derivatives" / "TimelapsedHRpQCT").is_dir():
        return root
    if root.name == "TimelapsedHRpQCT":
        return root
    if _looks_like_timelapsed_results_root(root):
        return root
    candidate = root / "TimelapsedHRpQCT"
    if candidate.is_dir():
        return candidate
    raise ValueError(
        "Could not resolve a TimelapsedHRpQCT dataset root. "
        "Select a TimelapsedHRpQCT results folder or a project folder containing one."
    )


def _mechanoregulation_derivative_roots(path):
    """Return likely shared derivative roots for mechanoregulation discovery."""
    root = Path(path).expanduser().resolve()
    candidates = (
        root / "derivatives" / "Mechanoregulation",
        root / "Mechanoregulation",
        root,
    )
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        unique.append(candidate)
    return tuple(unique)


def discover_mechanoregulation_manifests(path):
    """Discover shared Mechanoregulation and FEA manifests below a selected root."""
    manifests = []
    for root in _mechanoregulation_derivative_roots(path):
        for manifest_path in sorted(root.rglob("manifest.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            family = str(payload.get("derivative_family") or payload.get("workflow") or "")
            if family not in {"Mechanoregulation", "FEA"}:
                continue
            manifests.append(
                {
                    "path": manifest_path,
                    "derivative_family": family,
                    "records": list(payload.get("records", []) or []),
                    "metadata": dict(payload.get("metadata", {}) or {}),
                }
            )
    return manifests


def _looks_like_timelapsed_results_root(root):
    if (root / "analysis" / "visualize").is_dir():
        return True
    if (root / "analysis" / "pairwise_t0").is_dir():
        return True
    patterns = (
        "sub-*/site-*/analysis/visualize",
        "sub-*/analysis/pairwise_t0",
        "site-*/analysis/visualize",
    )
    return any(next(root.glob(pattern), None) is not None for pattern in patterns)


def _format_metric_value(value):
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.4g}"


def _short_text(value, max_chars=64):
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    head = max(8, (max_chars - 3) // 2)
    tail = max(8, max_chars - 3 - head)
    return f"{text[:head]}...{text[-tail:]}"


def _run_summary_text(summary):
    if not summary:
        return "processed=0 skipped=0 failed=0"
    parts = [
        f"processed={int(summary.get('processed', 0))}",
        f"skipped={int(summary.get('skipped', 0))}",
        f"failed={int(summary.get('failed', 0))}",
    ]
    if summary.get("cancelled"):
        parts.append("cancelled=True")
    return " ".join(parts)


def _metric_payload_from_json(payload):
    ccr_value = payload.get("CCR")
    if ccr_value is None:
        ccr = payload.get("conditional_curves", {}).get("ccr", {})
        if isinstance(ccr, dict):
            ccr_value = ccr.get("max")
    return {
        "CCR": ccr_value,
        "OR_F_ratio": payload.get("OR_F_ratio"),
        "OR_F": payload.get("OR_F"),
        "OR_R_ratio": payload.get("OR_R_ratio"),
        "OR_R": payload.get("OR_R"),
        "n_sampled_voxels": payload.get("sample_counts", {}).get("n_sampled_voxels", payload.get("n_sampled_voxels")),
        "n_formation": payload.get("sample_counts", {}).get("n_formation", payload.get("n_formation")),
        "n_resorption": payload.get("sample_counts", {}).get("n_resorption", payload.get("n_resorption")),
        "n_quiescence": payload.get("sample_counts", {}).get("n_quiescence", payload.get("n_quiescence")),
    }


def read_metric_summary(path):
    summary_path = Path(path).expanduser().resolve()
    if not summary_path.exists():
        return {}
    if summary_path.suffix.lower() == ".json":
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        values = _metric_payload_from_json(payload)
    else:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            values = next(reader, {})
    return {key: _format_metric_value(values.get(key)) for key in (*METRIC_KEYS, *COUNT_KEYS)}


class MechanoregulationHRpQCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Mechanoregulation"
        parent.categories = ["Bone Imaging.Microstructural Analysis"]
        parent.icon = qt.QIcon(str(Path(__file__).with_name("Resources") / "Icons" / "MechanoregulationHRpQCT.png"))
        parent.index = 60
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Slicer wrapper for post-TimelapsedHRpQCT bone mechanoregulation analysis. "
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = (
            "Author: Matthias Walle. "
            "Mechanoregulation analysis after TimelapsedHRpQCT. "
            "If used scientifically, cite the Timelapsed HR-pQCT mechanoregulation work."
        )


class MechanoregulationHRpQCTLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._lastDerivativeManifestSummary = {}

    def is_core_available(self):
        ok, _message = self.core_status()
        return ok

    def core_status(self):
        _prefer_local_core(force_reload=True)
        importlib.invalidate_caches()
        try:
            import bonemechreg
        except Exception as exc:
            return False, f"Not installed ({exc})"
        version = str(getattr(bonemechreg, "__version__", "0"))
        package_path = str(Path(getattr(bonemechreg, "__file__", "")).resolve())
        if _version_tuple(version) < _version_tuple(MIN_CORE_VERSION):
            return False, f"Out of date ({version}); install/update to {CORE_REQUIREMENT} >= {MIN_CORE_VERSION}."
        return True, f"Installed ({version}) from {package_path}"

    def install_or_update_core(self):
        if _use_local_core_checkout() and (CORE_LOCAL_REPO / "pyproject.toml").is_file():
            slicer.util.pip_install(str(CORE_LOCAL_REPO))
        else:
            slicer.util.pip_install(f"--upgrade --force-reinstall --no-cache-dir {CORE_REQUIREMENT}>={MIN_CORE_VERSION}")
        self._purge_core_modules()
        _prefer_local_core(force_reload=True)

    def _purge_core_modules(self):
        for name in list(sys.modules):
            if name == "bonemechreg" or name.startswith("bonemechreg."):
                sys.modules.pop(name, None)

    def resolve_root(self, path):
        return resolve_timelapsed_root(path)

    def discover_cases(self, path):
        _prefer_local_core()
        from bonemechreg.timelapse import (
            discover_timelapse_cases,
        )

        root = self.resolve_root(path)
        manifests = discover_mechanoregulation_manifests(root)
        self._lastDerivativeManifestSummary = {
            "FEA": sum(1 for item in manifests if item.get("derivative_family") == "FEA"),
            "Mechanoregulation": sum(
                1 for item in manifests if item.get("derivative_family") == "Mechanoregulation"
            ),
        }
        return root, discover_timelapse_cases(root)

    def available_rois(self, case):
        _prefer_local_core()
        try:
            from bonemechreg.timelapse import available_case_rois
        except Exception:
            return ["full"]
        return list(available_case_rois(case))

    def case_output_record(self, case, roi="full"):
        _prefer_local_core(force_reload=True)
        from bonemechreg.timelapse import (
            case_outputs,
        )

        outputs = case_outputs(case, roi=str(roi or "full"))
        summary_path = outputs["summary"] if outputs["summary"].exists() else outputs["csv"]
        metrics = read_metric_summary(summary_path)
        complete = all(
            outputs[key].exists()
            for key in ("sed", "material", "summary", "csv", "surface_events", "curves")
        )
        return {
            "case": case,
            "outputs": outputs,
            "status": "complete" if complete else "pending",
            "metrics": metrics,
        }

    def run_selected_case(self, case, dataset_root, profile, overwrite=False, n_boot=100):
        _prefer_local_core()
        from bonemechreg.timelapse import case_outputs

        root = self.resolve_root(dataset_root)
        outputs = case_outputs(case)
        run_dir = outputs["mechanoregulation_run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        use_local_core = _use_local_core_checkout() and (CORE_LOCAL_REPO / "bonemechreg").is_dir()
        if use_local_core:
            launcher = (
                "import sys; "
                f"sys.path.insert(0, {str(CORE_LOCAL_REPO)!r}); "
                "from bonemechreg.cli import main; "
                "raise SystemExit(main(sys.argv[1:]))"
            )
        else:
            launcher = "from bonemechreg.cli import main; import sys; raise SystemExit(main(sys.argv[1:]))"
        command = [
            sys.executable,
            "-c",
            launcher,
            "run",
            str(root),
            "--profile",
            str(profile),
            "--case-id",
            str(getattr(case, "case_id")),
            "--n-boot",
            str(int(n_boot)),
            "--verbose",
        ]
        if overwrite:
            command.append("--overwrite")
        env = dict(os.environ)
        if use_local_core:
            local_repo = str(CORE_LOCAL_REPO)
            env["PYTHONPATH"] = local_repo if not env.get("PYTHONPATH") else f"{local_repo}{os.pathsep}{env['PYTHONPATH']}"
        outputs["mechanoregulation_command"].write_text(" ".join(command) + "\n", encoding="utf-8")
        with outputs["mechanoregulation_stdout"].open("w", encoding="utf-8") as stdout_handle, outputs[
            "mechanoregulation_stderr"
        ].open("w", encoding="utf-8") as stderr_handle:
            proc = subprocess.Popen(
                command,
                cwd=str(root),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            returncode = proc.wait()
        outputs["mechanoregulation_exit_code"].write_text(f"{returncode}\n", encoding="utf-8")
        if returncode != 0:
            return {
                "discovered": 1,
                "processed": 0,
                "skipped": 0,
                "failed": 1,
                "dry_run": False,
                "case_id": getattr(case, "case_id", ""),
                "output_dir": str(getattr(case, "output_dir", "")),
                "run_dir": str(run_dir),
                "exit_code": int(returncode),
            }
        return self.case_output_record(case) | {
            "discovered": 1,
            "processed": int(outputs["summary"].exists()),
            "skipped": int(not outputs["summary"].exists()),
            "failed": 0,
            "dry_run": False,
            "case_id": getattr(case, "case_id", ""),
            "output_dir": str(getattr(case, "output_dir", "")),
            "run_dir": str(run_dir),
            "exit_code": int(returncode),
        }


class MechanoregulationHRpQCTWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = MechanoregulationHRpQCTLogic()
        self._cases = []
        self._resolvedRoot = None
        self._running = False
        self._updatingSelection = False
        self._runQueue = queue.Queue()
        self._runThread = None
        self._runTotal = 0
        self._runCaseIds = []
        self._sceneCases = []
        self._sceneRunRows = []
        self._sceneProcess = None
        self._sceneProcessText = ""
        self._cancelEvent = threading.Event()
        self._runPollTimer = qt.QTimer()
        self._runPollTimer.setInterval(150)
        self._runPollTimer.timeout.connect(self._poll_run_queue)

        self._build_runtime_section()
        self._build_scene_section(self.layout)

        self.layout.addStretch(1)
        self.update_runtime_status()

    def _build_runtime_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Runtime"
        self.layout.addWidget(box)
        layout = qt.QFormLayout(box)

        self.coreStatusLabel = qt.QLabel()
        self.coreStatusLabel.wordWrap = True
        row = qt.QHBoxLayout()
        self.checkButton = qt.QPushButton("Check Runtime")
        self.checkButton.clicked.connect(self.update_runtime_status)
        row.addWidget(self.checkButton)
        layout.addRow(row)
        layout.addRow("Status", self.coreStatusLabel)

    def _build_scene_section(self, parent_layout):
        box = ctk.ctkCollapsibleButton()
        box.text = "Scene"
        parent_layout.addWidget(box)
        layout = qt.QVBoxLayout(box)

        self.sceneInputsGroup = qt.QGroupBox("Inputs")
        scene_inputs_group_layout = qt.QVBoxLayout(self.sceneInputsGroup)
        layout.addWidget(self.sceneInputsGroup)
        scene_inputs = qt.QFormLayout()
        self.sceneRemodellingSelector = slicer.qMRMLNodeComboBox()
        self.sceneRemodellingSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode", "vtkMRMLSegmentationNode"]
        self.sceneRemodellingSelector.noneEnabled = True
        self.sceneRemodellingSelector.addEnabled = False
        self.sceneRemodellingSelector.removeEnabled = False
        self.sceneRemodellingSelector.setMRMLScene(slicer.mrmlScene)
        self.sceneSedSelector = slicer.qMRMLNodeComboBox()
        self.sceneSedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.sceneSedSelector.noneEnabled = True
        self.sceneSedSelector.addEnabled = False
        self.sceneSedSelector.removeEnabled = False
        self.sceneSedSelector.setMRMLScene(slicer.mrmlScene)
        self.sceneAnalysisMaskSelector = slicer.qMRMLNodeComboBox()
        self.sceneAnalysisMaskSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode", "vtkMRMLSegmentationNode"]
        self.sceneAnalysisMaskSelector.noneEnabled = True
        self.sceneAnalysisMaskSelector.addEnabled = False
        self.sceneAnalysisMaskSelector.removeEnabled = False
        self.sceneAnalysisMaskSelector.setMRMLScene(slicer.mrmlScene)
        self.sceneAnalysisMaskSegmentCombo = qt.QComboBox()
        self.sceneAnalysisMaskSegmentCombo.addItem("Auto", "")
        self.sceneAnalysisMaskSegmentCombo.enabled = False
        scene_inputs.addRow("Remodelling map", self.sceneRemodellingSelector)
        scene_inputs.addRow("ParOSol / FEA SED", self.sceneSedSelector)
        scene_inputs.addRow("Analysis mask", self.sceneAnalysisMaskSelector)
        scene_inputs.addRow("Mask segment", self.sceneAnalysisMaskSegmentCombo)
        scene_inputs_group_layout.addLayout(scene_inputs)

        self.sceneWorkflowGroup = qt.QGroupBox("Workflow")
        controls = qt.QFormLayout(self.sceneWorkflowGroup)
        layout.addWidget(self.sceneWorkflowGroup)
        self.sceneBootstrapSpinBox = qt.QSpinBox()
        self.sceneBootstrapSpinBox.minimum = 1
        self.sceneBootstrapSpinBox.maximum = 10000
        self.sceneBootstrapSpinBox.value = 100
        controls.addRow("Bootstraps", self.sceneBootstrapSpinBox)
        self.sceneNumericRemodellingRows = []
        self.sceneSegmentRemodellingRows = []
        self.sceneResorptionLabelSpinBox = self._label_value_spinbox(1)
        self.sceneQuiescenceLabelSpinBox = self._label_value_spinbox(2)
        self.sceneFormationLabelSpinBox = self._label_value_spinbox(3)
        self.sceneResorptionLabel = qt.QLabel("Resorption")
        self.sceneQuiescenceLabel = qt.QLabel("Quiescence")
        self.sceneFormationLabel = qt.QLabel("Formation")
        controls.addRow(self.sceneResorptionLabel, self.sceneResorptionLabelSpinBox)
        controls.addRow(self.sceneQuiescenceLabel, self.sceneQuiescenceLabelSpinBox)
        controls.addRow(self.sceneFormationLabel, self.sceneFormationLabelSpinBox)
        self.sceneNumericRemodellingRows.extend(
            (
                self.sceneResorptionLabel,
                self.sceneResorptionLabelSpinBox,
                self.sceneQuiescenceLabel,
                self.sceneQuiescenceLabelSpinBox,
                self.sceneFormationLabel,
                self.sceneFormationLabelSpinBox,
            )
        )
        self.sceneRemodellingResorptionSegmentCombo = qt.QComboBox()
        self.sceneRemodellingQuiescenceSegmentCombo = qt.QComboBox()
        self.sceneRemodellingFormationSegmentCombo = qt.QComboBox()
        self.sceneRemodellingResorptionSegmentLabel = qt.QLabel("Resorption segment")
        self.sceneRemodellingQuiescenceSegmentLabel = qt.QLabel("Quiescence segment")
        self.sceneRemodellingFormationSegmentLabel = qt.QLabel("Formation segment")
        controls.addRow(self.sceneRemodellingResorptionSegmentLabel, self.sceneRemodellingResorptionSegmentCombo)
        controls.addRow(self.sceneRemodellingQuiescenceSegmentLabel, self.sceneRemodellingQuiescenceSegmentCombo)
        controls.addRow(self.sceneRemodellingFormationSegmentLabel, self.sceneRemodellingFormationSegmentCombo)
        self.sceneSegmentRemodellingRows.extend(
            (
                self.sceneRemodellingResorptionSegmentLabel,
                self.sceneRemodellingResorptionSegmentCombo,
                self.sceneRemodellingQuiescenceSegmentLabel,
                self.sceneRemodellingQuiescenceSegmentCombo,
                self.sceneRemodellingFormationSegmentLabel,
                self.sceneRemodellingFormationSegmentCombo,
            )
        )

        self.sceneStatusLabel = qt.QLabel("Select a remodelling map and a matching ParOSol/FEA SED map.")
        self.sceneStatusLabel.wordWrap = True
        layout.addWidget(self.sceneStatusLabel)
        self.sceneProgressBar = qt.QProgressBar()
        self.sceneProgressBar.minimum = 0
        self.sceneProgressBar.maximum = 1
        self.sceneProgressBar.value = 0
        self.sceneProgressBar.visible = False
        self.sceneProgressBar.toolTip = "Current scene mechanoregulation progress."
        self.sceneCurrentStepLabel = qt.QLabel("Current step: idle")
        self.sceneCurrentStepLabel.wordWrap = True
        self.sceneCurrentStepLabel.setMinimumWidth(0)
        self.sceneCurrentStepLabel.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        layout.addWidget(self.sceneProgressBar)
        layout.addWidget(self.sceneCurrentStepLabel)

        button_row = qt.QHBoxLayout()
        self.sceneRunButton = qt.QPushButton("Run")
        self.sceneLoadButton = qt.QPushButton("Load")
        self.sceneStopButton = qt.QPushButton("Stop")
        self.sceneStopButton.enabled = False
        self.sceneRunButton.minimumHeight = 34
        self.sceneRunButton.setStyleSheet(
            "QPushButton { font-weight: 700; background-color: #2f7ed8; color: white; "
            "border: 1px solid #1f5fa8; border-radius: 4px; padding: 6px 10px; }"
            "QPushButton:disabled { background-color: #9aa9b8; color: #f0f0f0; border-color: #8794a0; }"
        )
        button_row.addWidget(self.sceneRunButton, 2)
        button_row.addWidget(self.sceneLoadButton, 1)
        button_row.addWidget(self.sceneStopButton, 1)
        layout.addLayout(button_row)

        self.logText = qt.QPlainTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setLineWrapMode(qt.QPlainTextEdit.WidgetWidth)
        self.logText.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.logText.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        self.logText.maximumBlockCount = 500
        layout.addWidget(self.logText)

        self.sceneAnalysisMaskSelector.currentNodeChanged.connect(
            lambda _node: self._refresh_scene_analysis_mask_segment_combo()
        )
        self.sceneRemodellingSelector.currentNodeChanged.connect(
            lambda _node: self._refresh_scene_remodelling_role_controls()
        )
        self.sceneRunButton.clicked.connect(self.run_scene)
        self.sceneLoadButton.clicked.connect(self._load_scene_mechanoregulation_outputs)
        self.sceneStopButton.clicked.connect(self.request_stop)
        parent_layout.addStretch(1)
        self._refresh_scene_remodelling_role_controls()

    def _label_value_spinbox(self, value):
        spin = qt.QSpinBox()
        spin.minimum = 0
        spin.maximum = 255
        spin.value = int(value)
        return spin

    def _show(self, message):
        text = self._status_text(message)
        if text.startswith("[scene]"):
            if hasattr(self, "sceneStatusLabel"):
                self.sceneStatusLabel.text = text
            if hasattr(self, "sceneCurrentStepLabel"):
                self.sceneCurrentStepLabel.text = text
        elif hasattr(self, "statusLabel"):
            self.statusLabel.text = text
            if hasattr(self, "currentStepLabel"):
                self.currentStepLabel.text = text
        if hasattr(self, "logText") and self.logText is not None:
            self.logText.appendPlainText(text)
        qt.QApplication.processEvents()

    def _status_text(self, message):
        text = str(message)
        if len(text) <= 220:
            return text
        return _short_text(text, 220)

    def update_runtime_status(self):
        ok, message = self.logic.core_status()
        self.coreStatusLabel.text = message
        self.coreStatusLabel.styleSheet = "color: #267326;" if ok else "color: #a15c00;"

    def install_core(self):
        self._show("[runtime] installing/updating bone-mechanoregulation")
        try:
            self.logic.install_or_update_core()
        except Exception as exc:
            slicer.util.errorDisplay(f"Install failed: {exc}")
            self._show(f"[runtime] install failed: {exc}")
            return
        self.update_runtime_status()
        self._show("[runtime] installation finished")

    def on_root_changed(self, *_args):
        root_text = self.datasetRootSelector.currentPath
        if not str(root_text).strip():
            self.resolvedRootLabel.text = "Not resolved"
            self._cases = []
            self._populate_case_combo()
            return
        try:
            self._resolvedRoot = self.logic.resolve_root(root_text)
            self.resolvedRootLabel.text = str(self._resolvedRoot)
            if not self._running:
                self._show("[discover] waiting for folder selection")
                self._discoverTimer.start()
        except Exception as exc:
            self._resolvedRoot = None
            self.resolvedRootLabel.text = str(exc)
            self._cases = []
            self._populate_case_combo()

    def discover_cases(self, show_errors=True):
        if not self.logic.is_core_available():
            message = "Install bone-mechanoregulation first."
            if show_errors:
                slicer.util.errorDisplay(message)
            self._show(f"[discover] {message}")
            return
        root_text = self.datasetRootSelector.currentPath
        if not str(root_text).strip():
            message = "Select a TimelapsedHRpQCT dataset root first."
            if show_errors:
                slicer.util.errorDisplay(message)
            self._show(f"[discover] {message}")
            return
        try:
            root, cases = self.logic.discover_cases(root_text)
        except Exception as exc:
            if show_errors:
                slicer.util.errorDisplay(str(exc))
            self._show(f"[discover] failed: {exc}")
            return
        self._resolvedRoot = root
        self.resolvedRootLabel.text = str(root)
        self._cases = list(cases)
        self._populate_case_combo()
        self._populate_roi_combo()
        manifest_summary = getattr(self.logic, "_lastDerivativeManifestSummary", {}) or {}
        manifest_text = (
            f"; FEA manifests {int(manifest_summary.get('FEA', 0))}, "
            f"Mechanoregulation manifests {int(manifest_summary.get('Mechanoregulation', 0))}"
        )
        self._show(f"[discover] discovered {len(self._cases)} case(s){manifest_text}")
        self.refresh_review()

    def _populate_case_combo(self):
        self._updatingSelection = True
        self.caseCombo.clear()
        if not self._cases:
            self.caseCombo.addItem("No cases discovered", "none")
            self.caseCombo.enabled = False
            self._updatingSelection = False
            self.refresh_review()
            return
        self.caseCombo.addItem("All cases", "all")
        for index, case in enumerate(self._cases):
            site = self._case_site(case)
            subject = getattr(case, "subject_id", "")
            case_id = getattr(case, "case_id", "")
            label = " | ".join(part for part in (subject, site, _short_text(case_id, 42)) if part)
            self.caseCombo.addItem(label, index)
            self.caseCombo.setItemData(index + 1, str(case_id), qt.Qt.ToolTipRole)
        self.caseCombo.enabled = True
        self.caseCombo.setCurrentIndex(0)
        self._updatingSelection = False

    def _select_case_by_id(self, case_id):
        if not case_id:
            return False
        for index, case in enumerate(self._cases):
            if str(getattr(case, "case_id", "")) == str(case_id):
                self.caseCombo.setCurrentIndex(index + 1)
                return True
        return False

    def _case_site(self, case):
        path = Path(getattr(case, "remodelling_image_path", ""))
        for part in path.parts:
            if str(part).startswith("site-"):
                return part
        return ""

    def on_case_combo_changed(self, *_args):
        if self._updatingSelection:
            return
        self._populate_roi_combo()
        self.refresh_review()

    def _populate_roi_combo(self):
        if not hasattr(self, "roiCombo"):
            return
        current = str(self.roiCombo.currentData or "full")
        self.roiCombo.blockSignals(True)
        self.roiCombo.clear()
        case = self._selected_case()
        rois = ["full"]
        if case is not None:
            try:
                rois = self.logic.available_rois(case) or ["full"]
            except Exception:
                rois = ["full"]
        labels = {"full": "Full", "trab": "Trabecular", "cort": "Cortical"}
        for roi in rois:
            self.roiCombo.addItem(labels.get(str(roi), str(roi)), str(roi))
        match_index = max(0, rois.index(current) if current in rois else 0)
        self.roiCombo.setCurrentIndex(match_index)
        self.roiCombo.enabled = case is not None
        self.roiCombo.blockSignals(False)

    def _scene_volume_nodes(self, node_classes):
        classes = tuple(str(value) for value in node_classes)
        nodes = []
        for index in range(slicer.mrmlScene.GetNumberOfNodes()):
            node = slicer.mrmlScene.GetNthNode(index)
            if node is None:
                continue
            try:
                if any(node.IsA(node_class) for node_class in classes):
                    nodes.append(node)
            except Exception:
                continue
        return nodes

    def _node_name_contains(self, node, tokens, *, exclude=()):
        name = str(node.GetName() if node is not None else "").lower()
        if any(str(token).lower() in name for token in exclude):
            return False
        return any(str(token).lower() in name for token in tokens)

    def _scene_remodelling_candidates(self):
        return [
            node
            for node in self._scene_volume_nodes(("vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"))
            if self._node_name_contains(node, ("remodelling", "remodeling"))
        ]

    def _scene_sed_candidates(self):
        return [
            node
            for node in self._scene_volume_nodes(("vtkMRMLScalarVolumeNode",))
            if self._node_name_contains(
                node,
                ("sed", "strain", "fea", "parosol", "loadhistory"),
                exclude=("remodelling", "remodeling", "material", "label"),
            )
        ]

    def _scene_parosol_output_candidates(self):
        return self._scene_sed_candidates()

    def _scene_mask_candidates(self, role):
        role = str(role).lower()
        if role == "seg":
            tokens = ("_seg", " seg", "segmentation", "bone")
            exclude = ("remodelling", "remodeling", "mask-full", "mask-trab", "mask-cort", "sed", "strain")
        elif role == "trab":
            tokens = ("trab", "mask-trab")
            exclude = ("remodelling", "remodeling", "sed", "strain")
        elif role == "cort":
            tokens = ("cort", "mask-cort")
            exclude = ("remodelling", "remodeling", "sed", "strain")
        else:
            tokens = ("full", "mask-full", "common")
            exclude = ("remodelling", "remodeling", "sed", "strain")
        return [
            node
            for node in self._scene_volume_nodes(("vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"))
            if self._node_name_contains(node, tokens, exclude=exclude)
        ]

    def _node_combo(self, nodes, *, include_generate=False, include_none=True, default_generate=False):
        combo = qt.QComboBox()
        if include_generate:
            combo.addItem("Generate", "generate")
        if include_none:
            combo.addItem("None", "none")
        for node in nodes:
            combo.addItem(str(node.GetName()), str(node.GetID()))
        if include_generate and default_generate:
            combo.setCurrentIndex(0)
        elif include_generate and nodes:
            combo.setCurrentIndex(2 if include_none else 1)
        elif combo.count > 0 and not include_generate:
            combo.setCurrentIndex(0)
        return combo

    def _scene_selected_rows(self):
        return [0] if self.sceneRemodellingSelector.currentNode() is not None else []

    def _scene_combo_value(self, row, column):
        widget = {1: self.sceneRemodellingSelector, 2: self.sceneSedSelector, 6: self.sceneAnalysisMaskSelector}.get(int(column))
        if widget is None or widget.currentNode() is None:
            return "none"
        return str(widget.currentNode().GetID())

    def _scene_node_from_combo(self, row, column):
        node_id = self._scene_combo_value(row, column)
        if node_id in {"", "none"}:
            return None
        return slicer.mrmlScene.GetNodeByID(node_id)

    def _scene_run_root(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Path(tempfile.gettempdir()) / "SlicerBoneImagingToolbox" / "MechanoregulationScene" / "scene_runs" / stamp

    def _save_scene_node(self, node, path):
        if node is None:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not slicer.util.saveNode(node, str(path)):
            raise RuntimeError(f"Could not save scene node to {path}")
        return path

    def _resample_saved_scene_image_to_reference_node(self, path, reference_node, *, nearest=False):
        if path is None or reference_node is None:
            return path
        path = Path(path)
        try:
            reference_path = path.with_name(f"{path.stem}_reference-grid.nii.gz")
            if not slicer.util.saveNode(reference_node, str(reference_path)):
                return path
            image = sitk.ReadImage(str(path))
            reference = sitk.ReadImage(str(reference_path))
            reference_path.unlink(missing_ok=True)
            if (
                image.GetSize() == reference.GetSize()
                and np.allclose(image.GetSpacing(), reference.GetSpacing())
                and np.allclose(image.GetOrigin(), reference.GetOrigin())
                and np.allclose(image.GetDirection(), reference.GetDirection())
            ):
                return path
            interpolator = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
            resampled = sitk.Resample(
                image,
                reference,
                sitk.Transform(),
                interpolator,
                0,
                image.GetPixelID(),
            )
            sitk.WriteImage(resampled, str(path))
        except Exception as exc:
            self._show(f"[scene] could not resample {path.name} to reference grid: {exc}")
        return path

    def _align_saved_scene_image_to_reference_image(self, path, reference_path, *, nearest=False):
        if path is None or reference_path is None:
            return path
        path = Path(path)
        reference_path = Path(reference_path)
        try:
            image = sitk.ReadImage(str(path))
            reference = sitk.ReadImage(str(reference_path))
            if (
                image.GetSize() == reference.GetSize()
                and np.allclose(image.GetSpacing(), reference.GetSpacing())
                and np.allclose(image.GetOrigin(), reference.GetOrigin())
                and np.allclose(image.GetDirection(), reference.GetDirection())
            ):
                return path
            if image.GetSize() == reference.GetSize() and np.allclose(image.GetSpacing(), reference.GetSpacing()):
                aligned = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
                aligned.CopyInformation(reference)
                sitk.WriteImage(aligned, str(path))
                return path
            interpolator = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
            resampled = sitk.Resample(
                image,
                reference,
                sitk.Transform(),
                interpolator,
                0,
                image.GetPixelID(),
            )
            sitk.WriteImage(resampled, str(path))
        except Exception as exc:
            self._show(f"[scene] could not align {path.name} to remodelling grid: {exc}")
        return path

    def _align_saved_scene_scalar_to_reference_image(self, path, reference_path):
        return self._align_saved_scene_image_to_reference_image(path, reference_path, nearest=False)

    def _refresh_scene_remodelling_role_controls(self):
        node = self.sceneRemodellingSelector.currentNode() if hasattr(self, "sceneRemodellingSelector") else None
        is_segmentation = bool(node is not None and node.IsA("vtkMRMLSegmentationNode"))
        for spin in (
            getattr(self, "sceneResorptionLabelSpinBox", None),
            getattr(self, "sceneQuiescenceLabelSpinBox", None),
            getattr(self, "sceneFormationLabelSpinBox", None),
        ):
            if spin is not None:
                spin.enabled = not is_segmentation
        for widget in getattr(self, "sceneNumericRemodellingRows", []):
            if widget is not None:
                widget.visible = not is_segmentation
        segment_combos = (
            ("resorption", getattr(self, "sceneRemodellingResorptionSegmentCombo", None)),
            ("quiescence", getattr(self, "sceneRemodellingQuiescenceSegmentCombo", None)),
            ("formation", getattr(self, "sceneRemodellingFormationSegmentCombo", None)),
        )
        for widget in getattr(self, "sceneSegmentRemodellingRows", []):
            if widget is not None:
                widget.visible = is_segmentation
        for role, combo in segment_combos:
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("None", "")
            combo.enabled = is_segmentation
            if is_segmentation:
                segmentation = node.GetSegmentation()
                best_index = 0
                for index in range(segmentation.GetNumberOfSegments()):
                    segment_id = segmentation.GetNthSegmentID(index)
                    segment = segmentation.GetSegment(segment_id)
                    segment_name = str(segment.GetName() if segment is not None else segment_id)
                    combo.addItem(segment_name, segment_id)
                    if self._segment_name_matches_role(segment_name, role):
                        best_index = combo.count - 1
                combo.setCurrentIndex(best_index)
            combo.blockSignals(False)

    @staticmethod
    def _segment_name_matches_role(name, role):
        text = str(name or "").lower()
        terms = {
            "resorption": ("resorption", "resorb", "erosion", "loss"),
            "quiescence": ("quiescence", "quiescent", "stable", "unchanged"),
            "formation": ("formation", "form", "gain"),
        }.get(str(role), ())
        return any(term in text for term in terms)

    def _selected_remodelling_segment_id(self, role):
        combos = {
            "resorption": getattr(self, "sceneRemodellingResorptionSegmentCombo", None),
            "quiescence": getattr(self, "sceneRemodellingQuiescenceSegmentCombo", None),
            "formation": getattr(self, "sceneRemodellingFormationSegmentCombo", None),
        }
        combo = combos.get(str(role))
        if combo is None:
            return None
        value = str(combo.currentData or "").strip()
        return value or None

    def _save_scene_remodelling_map(self, node, path, reference_node=None):
        if node is None:
            return None
        if node.IsA("vtkMRMLSegmentationNode"):
            return self._save_scene_remodelling_segmentation(node, path, reference_node=reference_node)
        saved_path = self._save_scene_node(node, path)
        saved_path = self._resample_saved_scene_image_to_reference_node(
            saved_path,
            reference_node,
            nearest=True,
        )
        return self._canonicalize_scene_remodelling_labels(saved_path)

    def _canonicalize_scene_remodelling_labels(self, path):
        resorption_label = int(self.sceneResorptionLabelSpinBox.value)
        quiescence_label = int(self.sceneQuiescenceLabelSpinBox.value)
        formation_label = int(self.sceneFormationLabelSpinBox.value)
        if (resorption_label, quiescence_label, formation_label) == (1, 2, 3):
            return Path(path)
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        canonical = np.zeros(array.shape, dtype=np.uint8)
        canonical[array == resorption_label] = 1
        canonical[array == quiescence_label] = 2
        canonical[array == formation_label] = 3
        out = sitk.GetImageFromArray(canonical)
        out.CopyInformation(image)
        sitk.WriteImage(out, str(path))
        return Path(path)

    def _save_scene_remodelling_segmentation(self, node, path, reference_node=None):
        selected = {
            "resorption": self._selected_remodelling_segment_id("resorption"),
            "quiescence": self._selected_remodelling_segment_id("quiescence"),
            "formation": self._selected_remodelling_segment_id("formation"),
        }
        missing = [role for role, segment_id in selected.items() if not segment_id]
        if missing:
            raise ValueError(f"Select remodelling segment(s) for: {', '.join(missing)}.")
        label_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{node.GetName()}_mechanoregulation_remodelling",
        )
        segment_ids = vtk.vtkStringArray()
        for segment_id in selected.values():
            segment_ids.InsertNextValue(str(segment_id))
        try:
            export_args = [node, segment_ids, label_node]
            if reference_node is not None:
                extent_mode = getattr(slicer.vtkSlicerSegmentationsModuleLogic, "EXTENT_REFERENCE_GEOMETRY", None)
                export_args.append(reference_node)
                if extent_mode is not None:
                    export_args.append(extent_mode)
            ok = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(*export_args)
            if not ok:
                raise RuntimeError(f"Could not export remodelling segments from {node.GetName()}.")
            source_array = np.asarray(slicer.util.arrayFromVolume(label_node))
            canonical = np.zeros(source_array.shape, dtype=np.uint8)
            for exported_label, output_label in enumerate((1, 2, 3), start=1):
                canonical[source_array == exported_label] = output_label
            slicer.util.updateVolumeFromArray(label_node, canonical)
            return self._save_scene_node(label_node, path)
        finally:
            slicer.mrmlScene.RemoveNode(label_node)

    def _refresh_scene_analysis_mask_segment_combo(self):
        combo = getattr(self, "sceneAnalysisMaskSegmentCombo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto", "")
        node = self.sceneAnalysisMaskSelector.currentNode() if hasattr(self, "sceneAnalysisMaskSelector") else None
        is_segmentation = bool(node is not None and node.IsA("vtkMRMLSegmentationNode"))
        combo.enabled = is_segmentation
        if is_segmentation:
            segmentation = node.GetSegmentation()
            for index in range(segmentation.GetNumberOfSegments()):
                segment_id = segmentation.GetNthSegmentID(index)
                segment = segmentation.GetSegment(segment_id)
                segment_name = str(segment.GetName() if segment is not None else segment_id)
                combo.addItem(segment_name, segment_id)
        combo.blockSignals(False)

    def _selected_scene_analysis_mask_segment_id(self):
        combo = getattr(self, "sceneAnalysisMaskSegmentCombo", None)
        if combo is None:
            return None
        selected_segment_id = str(combo.currentData or "").strip()
        return selected_segment_id or None

    def _save_scene_analysis_mask(self, node, path, reference_node=None, reference_path=None):
        if node is None:
            return None
        if not node.IsA("vtkMRMLSegmentationNode"):
            saved_path = self._save_scene_node(node, path)
            saved_path = self._resample_saved_scene_image_to_reference_node(
                saved_path,
                reference_node,
                nearest=True,
            )
            return self._align_saved_scene_image_to_reference_image(saved_path, reference_path, nearest=True)
        segment_id = self._selected_scene_analysis_mask_segment_id()
        segmentation = node.GetSegmentation()
        if not segment_id:
            if segmentation.GetNumberOfSegments() != 1:
                raise ValueError("Select which segment to use as the scene analysis mask.")
            segment_id = segmentation.GetNthSegmentID(0)
        elif segmentation.GetSegment(segment_id) is None:
            raise ValueError(f"Selected analysis mask segment was not found in {node.GetName()}.")
        label_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{node.GetName()}_mechanoregulation_analysis_mask",
        )
        segment_ids = vtk.vtkStringArray()
        segment_ids.InsertNextValue(str(segment_id))
        try:
            export_args = [node, segment_ids, label_node]
            if reference_node is not None:
                extent_mode = getattr(slicer.vtkSlicerSegmentationsModuleLogic, "EXTENT_REFERENCE_GEOMETRY", None)
                export_args.append(reference_node)
                if extent_mode is not None:
                    export_args.append(extent_mode)
            ok = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(*export_args)
            if not ok:
                raise RuntimeError(f"Could not export selected segment from {node.GetName()}.")
            saved_path = self._save_scene_node(label_node, path)
            saved_path = self._resample_saved_scene_image_to_reference_node(
                saved_path,
                reference_node,
                nearest=True,
            )
            return self._align_saved_scene_image_to_reference_image(saved_path, reference_path, nearest=True)
        finally:
            slicer.mrmlScene.RemoveNode(label_node)

    def _stage_scene_case(self, row):
        remodelling_node = self.sceneRemodellingSelector.currentNode()
        if remodelling_node is None:
            raise ValueError("Select a loaded Timelapsed remodelling map.")
        sed_node = self.sceneSedSelector.currentNode()
        if sed_node is None:
            raise ValueError("Select a loaded ParOSol/FEA SED map for each selected scene row.")
        run_root = self._scene_run_root()
        input_dir = run_root / "input"
        output_dir = run_root / "output" / "derivatives" / "Mechanoregulation" / "sub-scene" / "site-scene" / "runs" / f"scene-row-{row + 1:02d}"
        remodelling_path = self._save_scene_remodelling_map(
            remodelling_node,
            input_dir / f"scene-row-{row + 1:02d}_remodelling.nii.gz",
        )
        baseline_sed_path = self._save_scene_node(sed_node, input_dir / f"scene-row-{row + 1:02d}_sed.nii.gz")
        baseline_sed_path = self._align_saved_scene_scalar_to_reference_image(
            baseline_sed_path,
            remodelling_path,
        )
        analysis_mask_path = self._save_scene_analysis_mask(
            self.sceneAnalysisMaskSelector.currentNode(),
            input_dir / f"scene-row-{row + 1:02d}_analysis-mask.nii.gz",
            reference_node=remodelling_node if not remodelling_node.IsA("vtkMRMLSegmentationNode") else None,
            reference_path=remodelling_path,
        )
        if analysis_mask_path is not None:
            self._show(f"[scene] using analysis mask: {Path(analysis_mask_path).name}")
        else:
            self._show("[scene] no analysis mask selected; using whole remodelling image")
        return {
            "subject_id": "sub-scene",
            "case_id": f"scene-row-{row + 1:02d}",
            "baseline_image_path": str(remodelling_path),
            "remodelling_image_path": str(remodelling_path),
            "output_dir": str(output_dir),
            "baseline_segmentation_path": None,
            "trab_mask_path": None,
            "cort_mask_path": None,
            "full_mask_path": str(analysis_mask_path) if analysis_mask_path is not None else None,
            "baseline_sed_path": str(baseline_sed_path) if baseline_sed_path is not None else None,
            "run_root": str(run_root),
        }

    def run_scene(self):
        if self._running:
            slicer.util.warningDisplay("Mechanoregulation is already running.")
            return
        rows = self._scene_selected_rows()
        if not rows:
            slicer.util.warningDisplay("Select at least one scene remodelling map.")
            return
        try:
            staged = [self._stage_scene_case(row) for row in rows]
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
            self.sceneStatusLabel.text = str(exc)
            return
        self._running = True
        self._cancelEvent.clear()
        self._runTotal = len(staged)
        self._sceneRunRows = staged
        self.sceneRunButton.enabled = False
        self.sceneLoadButton.enabled = False
        self.sceneStopButton.enabled = True
        self.sceneProgressBar.visible = True
        self.sceneProgressBar.setRange(0, max(len(staged), 1))
        self.sceneProgressBar.value = 0
        profile = "standard"
        overwrite = False
        n_boot = int(self.sceneBootstrapSpinBox.value)
        self.sceneStatusLabel.text = f"Running 0/{len(staged)} scene mechanoregulation case(s)..."
        self.sceneCurrentStepLabel.text = "Current step: preparing"
        self._start_scene_process(staged, profile, overwrite, n_boot)

    def _python_slicer_executable(self):
        executable = Path(sys.executable)
        candidates = []
        if executable.name == "python-real":
            candidates.append(executable.with_name("PythonSlicer"))
        try:
            app_dir = Path(slicer.app.applicationDirPath())
            candidates.extend(
                [
                    app_dir / "PythonSlicer",
                    app_dir.parent / "bin" / "PythonSlicer",
                    app_dir.parent / "MacOS" / "PythonSlicer",
                ]
            )
        except Exception:
            pass
        candidates.append(executable)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(executable)

    @staticmethod
    def _qbytearray_to_text(raw):
        if isinstance(raw, (bytes, bytearray)):
            data = bytes(raw)
        else:
            try:
                data = raw.data()
                if isinstance(data, str):
                    data = data.encode("utf-8", errors="replace")
                else:
                    data = bytes(data)
            except Exception:
                try:
                    data = bytes(raw)
                except Exception:
                    data = str(raw).encode("utf-8", errors="replace")
        return data.decode("utf-8", errors="replace")

    def _scene_process_environment(self):
        environment = qt.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("MPLBACKEND", "Agg")
        for key in ("ITK_AUTOLOAD_PATH", "SITK_AUTOLOAD_PATH"):
            if environment.contains(key):
                environment.remove(key)
            environment.insert(key, "")
        python_paths = [str(TOOLBOX_ROOT), str(CORE_LOCAL_REPO)]
        derivatives_src = TOOLBOX_ROOT.parent / "bone-imaging-derivatives" / "src"
        if derivatives_src.exists():
            python_paths.append(str(derivatives_src))
        existing = str(environment.value("PYTHONPATH") or "")
        if existing:
            python_paths.append(existing)
        environment.insert("PYTHONPATH", os.pathsep.join(path for path in python_paths if path))
        return environment

    def _scene_process_script(self, config_path):
        local_repo = str(CORE_LOCAL_REPO)
        derivatives_src = str(TOOLBOX_ROOT.parent / "bone-imaging-derivatives" / "src")
        return (
            "import json, sys, shutil, traceback\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, {local_repo!r})\n"
            f"sys.path.insert(0, {derivatives_src!r})\n"
            "from bonemechreg.timelapse import TimelapseCase, case_outputs\n"
            "from bonemechreg.post_timelapse import run_post_timelapse_case\n"
            f"config = json.loads(Path({str(config_path)!r}).read_text(encoding='utf-8'))\n"
            "def optional_path(value):\n"
            "    return Path(value) if value else None\n"
            "summary = {'discovered': len(config['cases']), 'processed': 0, 'skipped': 0, 'failed': 0, 'cancelled': False, 'dry_run': False}\n"
            "try:\n"
            "    for index, staged in enumerate(config['cases'], start=1):\n"
            "        print(f\"[scene] {index}/{len(config['cases'])} {staged['case_id']}\", flush=True)\n"
            "        case = TimelapseCase(\n"
            "            subject_id=staged['subject_id'],\n"
            "            case_id=staged['case_id'],\n"
            "            baseline_image_path=Path(staged['baseline_image_path']),\n"
            "            remodelling_image_path=Path(staged['remodelling_image_path']),\n"
            "            output_dir=Path(staged['output_dir']),\n"
            "            baseline_segmentation_path=optional_path(staged.get('baseline_segmentation_path')),\n"
            "            trab_mask_path=optional_path(staged.get('trab_mask_path')),\n"
            "            cort_mask_path=optional_path(staged.get('cort_mask_path')),\n"
            "            full_mask_path=optional_path(staged.get('full_mask_path')),\n"
            "        )\n"
            "        baseline_sed_path = staged.get('baseline_sed_path')\n"
            "        if baseline_sed_path:\n"
            "            outputs = case_outputs(case)\n"
            "            outputs['sed'].parent.mkdir(parents=True, exist_ok=True)\n"
            "            shutil.copyfile(str(baseline_sed_path), str(outputs['sed']))\n"
            "        result = run_post_timelapse_case(\n"
            "            case,\n"
            "            profile=config.get('profile', 'standard'),\n"
            "            overwrite=bool(config.get('overwrite', False) and not baseline_sed_path),\n"
            "            reanalyze=True,\n"
            "            n_boot=int(config.get('n_boot', 100)),\n"
            "            verbose=True,\n"
            "        )\n"
            "        summary['processed'] += int(result.get('processed', 0))\n"
            "        summary['skipped'] += int(result.get('skipped', 0))\n"
            "        summary['failed'] += int(result.get('failed', 0))\n"
            "        print(f\"[scene] finished {index}/{len(config['cases'])} {staged['case_id']}\", flush=True)\n"
            "except Exception:\n"
            "    traceback.print_exc()\n"
            "    summary['failed'] += 1\n"
            "    print('BONE_MECHREG_SCENE_SUMMARY ' + json.dumps(summary), flush=True)\n"
            "    raise SystemExit(1)\n"
            "print('BONE_MECHREG_SCENE_SUMMARY ' + json.dumps(summary), flush=True)\n"
        )

    def _start_scene_process(self, staged_cases, profile, overwrite, n_boot):
        run_root = Path(staged_cases[0]["run_root"])
        config_path = run_root / "mechanoregulation_scene_cases.json"
        config_path.write_text(
            json.dumps(
                {
                    "cases": staged_cases,
                    "profile": str(profile),
                    "overwrite": bool(overwrite),
                    "n_boot": int(n_boot),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._sceneProcessText = ""
        process = qt.QProcess()
        process.setProcessChannelMode(qt.QProcess.MergedChannels)
        process.setProcessEnvironment(self._scene_process_environment())
        process.readyRead.connect(lambda process=process: self._append_scene_process_output(process))
        process.finished.connect(
            lambda *signal_args, process=process: self._scene_process_finished(process, *signal_args)
        )
        self._sceneProcess = process
        args = ["-c", self._scene_process_script(config_path)]
        self._show(f"[scene] launching: {self._python_slicer_executable()} -c <mechanoregulation scene runner>")
        process.start(self._python_slicer_executable(), args)
        if not process.waitForStarted(1000):
            self._sceneProcess = None
            self._finish_scene_run(None, error="Could not start the scene mechanoregulation process.")

    def _append_scene_process_output(self, process):
        text = self._qbytearray_to_text(process.readAll())
        if not text:
            return
        self._sceneProcessText += text
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "MRMLIDImageIO" in stripped:
                continue
            if "ImageIO factory did not return an ImageIOBase" in stripped:
                continue
            if stripped.startswith("itk version "):
                continue
            if stripped.startswith("[scene]"):
                self.sceneCurrentStepLabel.text = self._status_text(stripped)
                parts = stripped.split()
                if len(parts) > 1 and "/" in parts[1]:
                    try:
                        value = int(parts[1].split("/", 1)[0])
                        self.sceneProgressBar.value = max(0, value - 1 if "finished" not in stripped else value)
                    except Exception:
                        pass
            if not stripped.startswith("BONE_MECHREG_SCENE_SUMMARY "):
                self._show(stripped)

    def _scene_process_finished(self, process, *signal_args):
        self._append_scene_process_output(process)
        exit_code = int(signal_args[0]) if signal_args else int(process.exitCode())
        if self._sceneProcess is process:
            self._sceneProcess = None
        summary = None
        for line in self._sceneProcessText.splitlines():
            if line.startswith("BONE_MECHREG_SCENE_SUMMARY "):
                try:
                    summary = json.loads(line.split(" ", 1)[1])
                except Exception:
                    summary = None
        if exit_code != 0:
            details = self._sceneProcessText.strip() or f"Scene mechanoregulation failed with exit code {exit_code}."
            self._finish_scene_run(summary, error=details)
            return
        if summary is None:
            summary = {
                "discovered": len(self._sceneRunRows),
                "processed": len(self._sceneRunRows),
                "skipped": 0,
                "failed": 0,
                "cancelled": False,
                "dry_run": False,
            }
        self._finish_scene_run(summary)

    def _case_combo_data(self):
        if not hasattr(self, "caseCombo"):
            return None
        return self.caseCombo.currentData

    def _case_combo_index(self):
        data = self._case_combo_data()
        try:
            index = int(data)
        except (TypeError, ValueError):
            return None
        if 0 <= index < len(self._cases):
            return index
        return None

    def _selected_case_index(self):
        row = self._case_combo_index()
        if row is None:
            return None
        if row < 0 and self._cases:
            row = 0
        if row < 0 or row >= len(self._cases):
            return None
        return int(row)

    def _selected_case(self):
        index = self._selected_case_index()
        return None if index is None else self._cases[index]

    def run(self):
        if self._running:
            slicer.util.warningDisplay("Mechanoregulation is already running.")
            return
        if not self._cases:
            self.discover_cases(show_errors=True)
        if not self._cases:
            return
        if self._case_combo_data() == "all":
            cases = list(self._cases)
        else:
            case = self._selected_case()
            if case is None:
                slicer.util.warningDisplay("Select a case first.")
                return
            cases = [case]
        profile = self.profileCombo.currentText
        overwrite = bool(self.overwriteCheckBox.checked)
        n_boot = int(self.bootstrapSpinBox.value)
        dataset_root = self.datasetRootSelector.currentPath
        self._running = True
        self._runCaseIds = [str(getattr(case, "case_id", "")) for case in cases]
        self._cancelEvent.clear()
        self._runTotal = len(cases)
        self.runButton.enabled = False
        self.stopButton.enabled = True
        self.datasetRootSelector.enabled = False
        self.caseCombo.enabled = False
        self.profileCombo.enabled = False
        self.bootstrapSpinBox.enabled = False
        self.progressBar.visible = True
        self.progressBar.setRange(0, max(len(cases), 1))
        self.progressBar.value = 0
        self._show(f"[run] starting mechanoregulation for {len(cases)} case(s)")
        self._runThread = threading.Thread(
            target=self._run_worker,
            args=(cases, dataset_root, profile, overwrite, n_boot),
            daemon=True,
        )
        self._runThread.start()
        self._runPollTimer.start()

    def request_stop(self):
        if not self._running:
            return
        self._cancelEvent.set()
        scene_process = getattr(self, "_sceneProcess", None)
        if scene_process is not None:
            self._show("[scene] cancelling running process")
            scene_process.terminate()
            try:
                if not scene_process.waitForFinished(3000):
                    scene_process.kill()
            except Exception:
                pass
        if hasattr(self, "stopButton"):
            self.stopButton.enabled = False
        if hasattr(self, "sceneStopButton"):
            self.sceneStopButton.enabled = False
        self._show("[scene] stop requested")

    def _run_worker(self, cases, dataset_root, profile, overwrite, n_boot):
        summary = {
            "discovered": len(cases),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "cancelled": False,
            "dry_run": False,
        }
        try:
            for index, case in enumerate(cases, start=1):
                if self._cancelEvent.is_set():
                    summary["cancelled"] = True
                    break
                case_id = getattr(case, "case_id", f"case {index}")
                case_label = _short_text(case_id, 54)
                self._runQueue.put(("progress", index - 1, f"[run] {index}/{len(cases)} {case_label} (n_boot={int(n_boot)})"))
                result = self.logic.run_selected_case(
                    case,
                    dataset_root,
                    profile,
                    overwrite=overwrite,
                    n_boot=int(n_boot),
                )
                summary["processed"] += int(result.get("processed", 0))
                summary["skipped"] += int(result.get("skipped", 0))
                summary["failed"] += int(result.get("failed", 0))
                self._runQueue.put(("progress", index, f"[run] finished {index}/{len(cases)} {case_label}"))
                if self._cancelEvent.is_set():
                    summary["cancelled"] = True
                    break
            self._runQueue.put(("finished", summary))
        except Exception:
            self._runQueue.put(("error", traceback.format_exc()))

    def _poll_run_queue(self):
        while True:
            try:
                event = self._runQueue.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "progress":
                _kind, value, message = event
                if str(message).startswith("[scene]") and hasattr(self, "sceneProgressBar"):
                    self.sceneProgressBar.value = int(value)
                    self.sceneStatusLabel.text = self._status_text(message)
                    self.sceneCurrentStepLabel.text = self._status_text(message)
                else:
                    self.progressBar.value = int(value)
                self._show(message)
            elif kind == "finished":
                _kind, summary = event
                self._finish_run(summary)
                return
            elif kind == "error":
                _kind, details = event
                self._finish_run(None, error=details)
                return
            elif kind == "scene_finished":
                _kind, summary = event
                self._finish_scene_run(summary)
                return
            elif kind == "scene_error":
                _kind, details = event
                self._finish_scene_run(None, error=details)
                return

    def _finish_run(self, summary, error=None):
        self._runPollTimer.stop()
        self._running = False
        self.runButton.enabled = True
        self.stopButton.enabled = False
        self.datasetRootSelector.enabled = True
        self.caseCombo.enabled = bool(self._cases)
        self.profileCombo.enabled = True
        self.bootstrapSpinBox.enabled = True
        self.progressBar.setRange(0, max(self._runTotal, 1))
        self.progressBar.value = 0 if error else max(self._runTotal, 1)
        if error:
            slicer.util.errorDisplay(error)
            self._show(f"[run] failed: {error}")
            self.progressBar.visible = False
            return
        final_message = f"[run] finished: {_run_summary_text(summary)}"
        if summary and summary.get("cancelled"):
            final_message = f"[run] stopped: {_run_summary_text(summary)}"
        self._show(final_message)
        preferred_case_id = self._runCaseIds[0] if len(self._runCaseIds) == 1 else None
        try:
            self.discover_cases(show_errors=False)
            if preferred_case_id and self._select_case_by_id(preferred_case_id):
                self.refresh_review()
        except Exception as exc:
            self._show(f"[discover] refresh failed after run: {exc}")
        finally:
            self.statusLabel.text = self._status_text(final_message)
            self.currentStepLabel.text = self._status_text(final_message)
            self.progressBar.visible = False

    def _finish_scene_run(self, summary, error=None):
        self._runPollTimer.stop()
        self._running = False
        self.sceneRunButton.enabled = True
        self.sceneLoadButton.enabled = True
        self.sceneStopButton.enabled = False
        self.sceneProgressBar.value = 0 if error else max(self._runTotal, 1)
        if error:
            slicer.util.errorDisplay(error)
            self.sceneStatusLabel.text = self._status_text(f"Scene run failed: {error}")
            self.sceneCurrentStepLabel.text = "Current step: failed"
            self._show(f"[scene] failed: {error}")
            self.sceneProgressBar.visible = False
            return
        final_message = f"[scene] finished: {_run_summary_text(summary)}"
        if summary and summary.get("cancelled"):
            final_message = f"[scene] stopped: {_run_summary_text(summary)}"
        self.sceneStatusLabel.text = self._status_text(final_message)
        self.sceneCurrentStepLabel.text = self._status_text(final_message)
        self._show(final_message)
        self._load_scene_mechanoregulation_outputs()
        self.sceneProgressBar.visible = False

    def _load_scene_mechanoregulation_outputs(self):
        loaded = 0
        staged_rows = list(getattr(self, "_sceneRunRows", []) or [])
        if not staged_rows:
            slicer.util.warningDisplay("No scene mechanoregulation outputs are available yet.")
            return
        summary_paths = []
        for staged in staged_rows:
            try:
                _prefer_local_core(force_reload=True)
                from bonemechreg.timelapse import TimelapseCase, case_outputs

                def optional_path(value):
                    return Path(value) if value else None

                case = TimelapseCase(
                    subject_id=staged["subject_id"],
                    case_id=staged["case_id"],
                    baseline_image_path=Path(staged["baseline_image_path"]),
                    remodelling_image_path=Path(staged["remodelling_image_path"]),
                    output_dir=Path(staged["output_dir"]),
                    baseline_segmentation_path=optional_path(staged.get("baseline_segmentation_path")),
                    trab_mask_path=optional_path(staged.get("trab_mask_path")),
                    cort_mask_path=optional_path(staged.get("cort_mask_path")),
                    full_mask_path=optional_path(staged.get("full_mask_path")),
                )
                outputs = case_outputs(case, roi="full")
                sed_node = None
                sed_path = outputs.get("sed")
                if sed_path is not None and Path(sed_path).exists():
                    sed_node = self._load_scalar_volume(sed_path, self._loaded_node_name(staged.get("case_id", "case"), "sed"))
                    if sed_node is not None:
                        self._style_fe_scalar_volume(sed_node)
                        loaded += 1
                elif self._style_selected_scene_sed():
                    sed_node = self.sceneSedSelector.currentNode()
                    loaded += 1
                event_path = self._scene_surface_events_path(outputs, staged)
                is_surface_events = event_path is not None and Path(event_path).exists()
                if not is_surface_events:
                    event_path = staged.get("remodelling_image_path")
                    self._show(f"[load] using volume remodelling fallback: {Path(event_path).name}")
                else:
                    self._show(f"[load] using analysed surface events: {Path(event_path).name}")
                if self._load_event_segmentation(
                    event_path,
                    self._loaded_node_name(staged.get("case_id", "case"), "events"),
                    reference_node=sed_node,
                    mask_path=None if is_surface_events else staged.get("full_mask_path"),
                ) is not None:
                    loaded += 1
                csv_path = outputs.get("csv")
                if csv_path is not None and Path(csv_path).exists():
                    summary_paths.append(Path(csv_path))
            except Exception as exc:
                self._show(f"[scene] could not load outputs: {exc}")
        if summary_paths:
            try:
                compact_path = self._write_scene_mechanoregulation_summary_table_csv(summary_paths)
                if self._load_table(compact_path, self._loaded_node_name("scene", "summary")):
                    loaded += 1
            except Exception as exc:
                self._show(f"[scene] could not load compact mechanoregulation table: {exc}")
        if loaded:
            self.sceneStatusLabel.text = f"{self.sceneStatusLabel.text} Loaded {loaded} output artifact(s)."

    def _scene_surface_events_path(self, outputs, staged):
        path = outputs.get("surface_events") if outputs else None
        if path is not None and Path(path).exists():
            return Path(path)
        output_dir = Path(staged.get("output_dir") or "")
        if not output_dir.exists():
            return path
        case_id = str(staged.get("case_id") or "").strip()
        patterns = []
        if case_id:
            patterns.append(f"{case_id}_roi-full_surface-events.nii.gz")
            patterns.append(f"{case_id}*_surface-events.nii.gz")
        patterns.append("*_roi-full_surface-events.nii.gz")
        patterns.append("*_surface-events.nii.gz")
        for pattern in patterns:
            matches = sorted(output_dir.glob(pattern))
            if matches:
                return matches[0]
        return path

    def _style_selected_scene_sed(self):
        sed_node = self.sceneSedSelector.currentNode() if hasattr(self, "sceneSedSelector") else None
        if sed_node is None:
            return False
        self._style_fe_scalar_volume(sed_node)
        self._show(f"[load] styled selected SED: {sed_node.GetName()}")
        return True

    def _selected_record(self):
        if self._case_combo_data() == "all":
            return None
        case = self._selected_case()
        if case is None:
            return None
        return self.logic.case_output_record(case, roi=self._selected_roi())

    def _selected_roi(self):
        if not hasattr(self, "roiCombo"):
            return "full"
        return str(self.roiCombo.currentData or "full")

    def refresh_review(self):
        record = self._selected_record()
        if record is None:
            for row in range(self.metricsTable.rowCount):
                item = self.metricsTable.item(row, 1)
                if item is not None:
                    item.setText("")
            self._clear_curve_preview(self.conditionalCurvePreviewLabel, "Select one case to review outputs")
            self._clear_curve_preview(self.schulteCurvePreviewLabel, "Select one case to review outputs")
            return
        metrics = record["metrics"]
        for row, key in enumerate((*METRIC_KEYS, *COUNT_KEYS)):
            item = self.metricsTable.item(row, 1) or qt.QTableWidgetItem("")
            item.setText(str(metrics.get(key, "")))
            if self.metricsTable.item(row, 1) is None:
                self.metricsTable.setItem(row, 1, item)
        self._show_curve_preview(self.conditionalCurvePreviewLabel, record["outputs"].get("curves"))
        self._show_curve_preview(self.schulteCurvePreviewLabel, record["outputs"].get("schulte_curves"))

    def _configure_curve_preview_label(self, label):
        label.setMinimumHeight(320)
        label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        label.setAlignment(qt.Qt.AlignCenter)
        label.setText("No curves loaded")
        label.setWordWrap(True)

    def _clear_curve_preview(self, label, text):
        label.setText(text)
        label.setPixmap(qt.QPixmap())
        label.setMinimumHeight(160)

    def _show_curve_preview(self, label, curves_path):
        if curves_path is None or not Path(curves_path).exists():
            self._clear_curve_preview(label, "No curves loaded")
            return
        pixmap = qt.QPixmap(str(curves_path))
        if pixmap.isNull():
            self._clear_curve_preview(label, f"Could not load curves: {curves_path}")
            return
        width = max(int(label.width or 640), 360)
        height = 460
        scaled = pixmap.scaled(
            width,
            height,
            qt.Qt.KeepAspectRatio,
            qt.Qt.SmoothTransformation,
        )
        label.setMinimumHeight(max(180, int(scaled.height()) + 12))
        label.setPixmap(scaled)

    def load_selected_output(self, output_key):
        record = self._selected_record()
        if record is None:
            slicer.util.warningDisplay("Select a case first.")
            return
        if output_key == "remodelling":
            path = Path(getattr(record["case"], "remodelling_image_path"))
        else:
            path = Path(record["outputs"].get(output_key, ""))
        if not path.exists():
            slicer.util.warningDisplay(f"Output does not exist: {path}")
            return
        self._load_volume(path, output_key)

    def _load_volume(self, path, output_key):
        case = self._selected_case()
        case_id = getattr(case, "case_id", "case") if case is not None else "case"
        name = self._loaded_node_name(case_id, output_key)
        if output_key == "remodelling":
            node = self._load_label_volume(path, name)
            self._style_remodelling_labelmap(node)
        else:
            node = self._load_scalar_volume(path, name)
            if output_key == "sed":
                self._style_fe_scalar_volume(node)
        if node:
            self._show(f"[load] loaded {output_key}: {Path(path).name}")
        return node

    def _loaded_node_name(self, case_id, output_key):
        short_case = str(case_id or "case")
        if len(short_case) > 54:
            short_case = short_case[:51] + "..."
        return f"Mechanoregulation {output_key} {short_case}"

    def _load_scalar_volume(self, path, name):
        loaded = slicer.util.loadVolume(str(path), {"name": name})
        return self._loaded_node_from_result(loaded)

    def _load_label_volume(self, path, name):
        loaded = slicer.util.loadLabelVolume(str(path), {"name": name})
        return self._loaded_node_from_result(loaded)

    def _load_table(self, path, name):
        existing = slicer.mrmlScene.GetFirstNodeByName(name)
        if existing is not None:
            slicer.mrmlScene.RemoveNode(existing)
        loaded = slicer.util.loadTable(str(path))
        node = self._loaded_node_from_result(loaded)
        if node:
            try:
                node.SetName(name)
            except Exception:
                pass
            self._show(f"[load] loaded table: {Path(path).name}")
        return node

    def _write_scene_mechanoregulation_summary_table_csv(self, summary_paths):
        output_dir = Path(tempfile.gettempdir()) / "SlicerBoneImagingToolbox" / "MechanoregulationScene" / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(summary_paths[0]).stem}_compact.csv"
        rows = []
        for summary_path in summary_paths:
            with Path(summary_path).open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                source_row = next(reader, {})
            roi = self._mechanoregulation_summary_roi(Path(summary_path), source_row)
            rows.extend(self._mechanoregulation_compact_rows(roi, source_row))
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["ROI", "Metric", "Unit", "Low conf", "Median", "High conf"])
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    @staticmethod
    def _mechanoregulation_summary_roi(summary_path, source_row):
        value = str(source_row.get("roi") or source_row.get("compartment") or "").strip()
        if value:
            return value
        match = re.search(r"_roi-([^_]+)", Path(summary_path).stem)
        return match.group(1) if match else "ROI"

    @classmethod
    def _mechanoregulation_compact_rows(cls, roi, source_row):
        def first_value(*names):
            for name in names:
                value = source_row.get(name)
                if value not in (None, ""):
                    return value
            return ""

        return [
            cls._mechanoregulation_compact_row(roi, "CCR", "fraction", "", first_value("CCR", "ccr"), ""),
            cls._mechanoregulation_compact_row(
                roi,
                "Lazy min",
                "% normalized SED",
                "",
                first_value("CCR_low_threshold", "lazy_min", "lazy_zone_low", "sed_lazy_min"),
                "",
            ),
            cls._mechanoregulation_compact_row(
                roi,
                "Lazy max",
                "% normalized SED",
                "",
                first_value("CCR_high_threshold", "lazy_max", "lazy_zone_high", "sed_lazy_max"),
                "",
            ),
            cls._mechanoregulation_compact_row(
                roi,
                "ORR",
                "% per 1% SED decrease",
                first_value("OR_R_CI_low", "ORR_low_conf", "orr_low_conf"),
                first_value("OR_R", "ORR", "orr"),
                first_value("OR_R_CI_high", "ORR_high_conf", "orr_high_conf"),
            ),
            cls._mechanoregulation_compact_row(
                roi,
                "ORF",
                "% per 1% SED increase",
                first_value("OR_F_CI_low", "ORF_low_conf", "orf_low_conf"),
                first_value("OR_F", "ORF", "orf"),
                first_value("OR_F_CI_high", "ORF_high_conf", "orf_high_conf"),
            ),
        ]

    @classmethod
    def _mechanoregulation_compact_row(cls, roi, metric, unit, low, median, high):
        return {
            "ROI": str(roi),
            "Metric": str(metric),
            "Unit": str(unit),
            "Low conf": cls._format_compact_table_value(low),
            "Median": cls._format_compact_table_value(median),
            "High conf": cls._format_compact_table_value(high),
        }

    @staticmethod
    def _format_compact_table_value(value):
        text = str(value).strip()
        if not text:
            return ""
        try:
            number = float(text)
        except ValueError:
            return text
        if not np.isfinite(number):
            return ""
        return f"{number:.4g}"

    def _load_event_segmentation(self, path, name, reference_node=None, mask_path=None):
        if path is None or not Path(path).exists():
            return None
        label_node = self._load_label_volume(path, f"{name}_source")
        if label_node is None:
            return None
        try:
            array = np.asarray(slicer.util.arrayFromVolume(label_node))
            events = np.zeros(array.shape, dtype=np.uint8)
            events[array == 1] = 1
            events[(array == 3) | (array == 4)] = 2
            mask_array = self._event_display_mask_array(mask_path, events.shape, reference_path=path)
            if mask_array is not None:
                events[~mask_array] = 0
            slicer.util.updateVolumeFromArray(label_node, events)
            label_node.SetName(f"{name}_event_source")
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None:
                slicer.mrmlScene.RemoveNode(existing)
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
            segmentation_node.CreateDefaultDisplayNodes()
            try:
                segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_node or label_node)
            except Exception:
                pass
            imported = slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label_node, segmentation_node)
            if imported is False:
                raise RuntimeError(f"Could not import event labels from {Path(path).name}.")
            self._style_event_segmentation(segmentation_node)
            self._show(f"[load] loaded formation/resorption events: {Path(path).name}")
            return segmentation_node
        finally:
            try:
                slicer.mrmlScene.RemoveNode(label_node)
            except Exception:
                pass

    def _style_event_segmentation(self, segmentation_node):
        if segmentation_node is None:
            return
        segmentation_node.CreateDefaultDisplayNodes()
        segmentation = segmentation_node.GetSegmentation()
        labels = [
            ("Resorption", (1.0, 0.05, 0.70)),
            ("Formation", (1.0, 0.48, 0.0)),
        ]
        for index, (label, color) in enumerate(labels):
            if index >= segmentation.GetNumberOfSegments():
                break
            segment_id = segmentation.GetNthSegmentID(index)
            segment = segmentation.GetSegment(segment_id)
            if segment is None:
                continue
            segment.SetName(label)
            segment.SetColor(float(color[0]), float(color[1]), float(color[2]))
        display = segmentation_node.GetDisplayNode()
        if display is not None:
            display.SetVisibility(True)
            display.SetVisibility2DFill(True)
            display.SetVisibility2DOutline(True)
            display.SetVisibility3D(False)
            display.SetOpacity(0.65)
        try:
            segmentation_node.CreateClosedSurfaceRepresentation()
        except Exception:
            pass

    def _event_display_mask_array(self, mask_path, expected_shape, reference_path=None):
        if mask_path is None or not Path(mask_path).exists():
            return None
        try:
            import SimpleITK as sitk

            mask_image = sitk.ReadImage(str(mask_path))
            if reference_path is not None and Path(reference_path).exists():
                reference_image = sitk.ReadImage(str(reference_path))
                same_grid = (
                    mask_image.GetSize() == reference_image.GetSize()
                    and mask_image.GetSpacing() == reference_image.GetSpacing()
                    and mask_image.GetOrigin() == reference_image.GetOrigin()
                    and mask_image.GetDirection() == reference_image.GetDirection()
                )
                if not same_grid:
                    mask_image = sitk.Resample(
                        sitk.Cast(mask_image > 0, sitk.sitkUInt8),
                        reference_image,
                        sitk.Transform(3, sitk.sitkIdentity),
                        sitk.sitkNearestNeighbor,
                        0,
                        sitk.sitkUInt8,
                    )
            mask_array = sitk.GetArrayFromImage(mask_image) > 0
        except Exception as exc:
            self._show(f"[load] could not apply analysis mask to event display: {exc}")
            return None
        if tuple(mask_array.shape) != tuple(expected_shape):
            self._show(
                "[load] skipped event display mask because its grid does not match "
                f"the remodelling map ({mask_array.shape} vs {tuple(expected_shape)})"
            )
            return None
        return mask_array

    def _loaded_node_from_result(self, loaded):
        if isinstance(loaded, tuple):
            ok, node = loaded
            return node if ok else None
        if isinstance(loaded, bool):
            return None
        return loaded

    def _style_remodelling_labelmap(self, label_node):
        if label_node is None:
            return
        label_node.CreateDefaultDisplayNodes()
        display = label_node.GetDisplayNode()
        if display is None:
            return
        color_node = self._remodelling_color_node()
        if color_node is not None and hasattr(display, "SetAndObserveColorNodeID"):
            display.SetAndObserveColorNodeID(color_node.GetID())
        display.SetVisibility(True)
        if hasattr(display, "SetOpacity"):
            display.SetOpacity(1.0)
        try:
            slicer.util.setSliceViewerLayers(label=label_node, fit=False)
        except Exception:
            pass

    def _remodelling_color_node(self):
        name = "TimelapsedHRpQCT_RemodellingColors"
        existing = slicer.mrmlScene.GetFirstNodeByName(name)
        if existing is not None:
            return existing
        color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", name)
        if hasattr(color_node, "SetTypeToUser"):
            color_node.SetTypeToUser()
        if hasattr(color_node, "SetNumberOfColors"):
            color_node.SetNumberOfColors(6)
        colors = {
            0: ("background", 0.0, 0.0, 0.0, 0.0),
            1: ("resorption", 1.00, 0.05, 0.70, 1.0),
            2: ("quiescent", 0.62, 0.62, 0.62, 0.32),
            3: ("formation", 1.00, 0.48, 0.00, 1.0),
            4: ("formation", 1.00, 0.48, 0.00, 1.0),
            5: ("quiescent", 0.62, 0.62, 0.62, 0.32),
        }
        for value, color in colors.items():
            try:
                color_node.SetColor(int(value), color[0], float(color[1]), float(color[2]), float(color[3]), float(color[4]))
            except Exception:
                pass
        try:
            color_node.NamesInitialisedOn()
            color_node.HideFromEditorsOn()
        except Exception:
            pass
        return color_node

    def _style_fe_scalar_volume(self, volume_node):
        if volume_node is None:
            return
        volume_node.CreateDefaultDisplayNodes()
        display = volume_node.GetDisplayNode()
        if display is None:
            return
        color_node = self._fe_jet_color_node()
        if color_node is not None and hasattr(display, "SetAndObserveColorNodeID"):
            display.SetAndObserveColorNodeID(color_node.GetID())
        values = self._positive_finite_volume_values(volume_node)
        if values.size:
            lower, upper, maximum = self._fe_display_range(values)
            if np.isfinite(upper) and upper > lower:
                display.AutoWindowLevelOff()
                display.SetWindowLevelMinMax(lower, upper)
            if hasattr(display, "ApplyThresholdOn"):
                display.ApplyThresholdOn()
                display.SetLowerThreshold(lower)
                display.SetUpperThreshold(maximum)
        if hasattr(display, "SetInterpolate"):
            display.SetInterpolate(0)

    def _fe_jet_color_node(self):
        name = "ParOSol_SED_jet"
        existing = slicer.mrmlScene.GetFirstNodeByName(name)
        if existing is not None:
            return existing
        color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", name)
        color_node.SetTypeToUser()
        color_node.SetNumberOfColors(256)
        color_node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
        for index in range(1, 256):
            value = index / 255.0
            red = min(max(1.5 - abs(4.0 * value - 3.0), 0.0), 1.0)
            green = min(max(1.5 - abs(4.0 * value - 2.0), 0.0), 1.0)
            blue = min(max(1.5 - abs(4.0 * value - 1.0), 0.0), 1.0)
            color_node.SetColor(index, f"fe_{index}", red, green, blue, 1.0)
        color_node.HideFromEditorsOn()
        return color_node

    def _positive_finite_volume_values(self, volume_node):
        try:
            array = np.asarray(slicer.util.arrayFromVolume(volume_node), dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
        values = array[np.isfinite(array) & (array > 0)]
        return values.astype(float, copy=False)

    def _fe_display_range(self, values):
        lower = float(np.percentile(values, 1.0))
        upper = float(np.percentile(values, 99.0))
        maximum = float(np.max(values))
        if upper <= lower:
            lower = float(np.min(values))
            upper = maximum
        return lower, upper, maximum

    def show_parosol_command(self):
        record = self._selected_record()
        if record is None:
            slicer.util.warningDisplay("Select one case first.")
            return
        outputs = record["outputs"]
        mechanoregulation_command = Path(outputs.get("mechanoregulation_command", ""))
        wrapper_command = Path(outputs.get("parosol_wrapper_command", ""))
        native_command = Path(outputs.get("parosol_native_command", ""))
        sections = []
        if mechanoregulation_command.exists():
            sections.append("mechanoregulation child process:\n" + mechanoregulation_command.read_text(encoding="utf-8").strip())
        if wrapper_command.exists():
            sections.append("bone-mechanoregulation wrapper:\n" + wrapper_command.read_text(encoding="utf-8").strip())
        if native_command.exists():
            sections.append("parosol-py native solver:\n" + native_command.read_text(encoding="utf-8").strip())
        if not sections:
            slicer.util.warningDisplay(f"No command log found yet. Expected: {mechanoregulation_command}")
            return
        message = "\n\n".join(sections)
        slicer.util.infoDisplay(message, windowTitle="Mechanoregulation command")
        self._show(f"[logs] command: {mechanoregulation_command}")

    def open_parosol_logs(self):
        record = self._selected_record()
        if record is None:
            slicer.util.warningDisplay("Select one case first.")
            return
        run_dir = Path(record["outputs"].get("mechanoregulation_run_dir", ""))
        if not run_dir.exists():
            run_dir = Path(record["outputs"].get("parosol_solve_dir", ""))
        if not run_dir.exists():
            slicer.util.warningDisplay(f"Log folder does not exist yet: {run_dir}")
            return
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(run_dir)))
        self._show(f"[logs] opened {run_dir}")


class MechanoregulationHRpQCTTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("MechanoregulationHRpQCT smoke test passed.")
