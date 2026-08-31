from __future__ import annotations

import csv
import importlib
import json
import os
import queue
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
import slicer

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


MODULE_VERSION = "0.1.0"
CORE_REQUIREMENT = "bone-mechanoregulation"
MIN_CORE_VERSION = "0.1.2"
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
        parent.title = "Bone Mechanoregulation"
        parent.categories = ["Bone Imaging.Timelapsed Methods"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Slicer wrapper for post-TimelapsedHRpQCT bone mechanoregulation analysis. "
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = (
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
        _prefer_local_core()
        from bonemechreg.timelapse import (
            case_outputs,
        )

        outputs = case_outputs(case, roi=str(roi or "full"))
        summary_path = outputs["summary"] if outputs["summary"].exists() else outputs["csv"]
        metrics = read_metric_summary(summary_path)
        complete = all(
            outputs[key].exists()
            for key in ("sed", "material", "summary", "csv", "curves")
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
        self._cancelEvent = threading.Event()
        self._runPollTimer = qt.QTimer()
        self._runPollTimer.setInterval(150)
        self._runPollTimer.timeout.connect(self._poll_run_queue)
        self._discoverTimer = qt.QTimer()
        self._discoverTimer.setSingleShot(True)
        self._discoverTimer.setInterval(350)
        self._discoverTimer.timeout.connect(lambda: self.discover_cases(show_errors=False))

        self._build_runtime_section()
        self.modeTabs = qt.QTabWidget()
        batch_tab = qt.QWidget()
        scene_tab = qt.QWidget()
        review_tab = qt.QWidget()
        batch_layout = qt.QVBoxLayout(batch_tab)
        scene_layout = qt.QVBoxLayout(scene_tab)
        review_layout = qt.QVBoxLayout(review_tab)
        self.modeTabs.addTab(batch_tab, "Batch")
        self.modeTabs.addTab(scene_tab, "Scene")
        self.modeTabs.addTab(review_tab, "Review")
        self.layout.addWidget(self.modeTabs)
        self._build_input_section(batch_layout)
        self._build_scene_section(scene_layout)
        self._build_review_section(review_layout)

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
        self.installButton = qt.QPushButton("Install / Update bone-mechanoregulation")
        self.checkButton = qt.QPushButton("Check Runtime")
        self.installButton.clicked.connect(self.install_core)
        self.checkButton.clicked.connect(self.update_runtime_status)
        row.addWidget(self.installButton)
        row.addWidget(self.checkButton)
        layout.addRow(row)
        layout.addRow("Status", self.coreStatusLabel)

    def _build_input_section(self, parent_layout=None):
        box = ctk.ctkCollapsibleButton()
        box.text = "Batch"
        (parent_layout or self.layout).addWidget(box)
        layout = qt.QVBoxLayout(box)
        self.batchDiscoveryGroup = qt.QGroupBox("Discovery")
        discovery_form = qt.QFormLayout(self.batchDiscoveryGroup)
        layout.addWidget(self.batchDiscoveryGroup)

        self.datasetRootSelector = ctk.ctkPathLineEdit()
        self.datasetRootSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.datasetRootSelector.currentPathChanged.connect(self.on_root_changed)
        self.resolvedRootLabel = qt.QLabel("Not resolved")
        self.resolvedRootLabel.wordWrap = True
        self.resolvedRootLabel.setMinimumWidth(0)
        self.resolvedRootLabel.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)

        self.profileCombo = qt.QComboBox()
        self.profileCombo.addItems(["XtremeCTII", "XtremeCTI"])
        self.overwriteCheckBox = qt.QCheckBox("Overwrite existing outputs")
        self.bootstrapSpinBox = qt.QSpinBox()
        self.bootstrapSpinBox.minimum = 1
        self.bootstrapSpinBox.maximum = 10000
        self.bootstrapSpinBox.value = 100
        self.bootstrapSpinBox.toolTip = "Number of class-balanced mechanoregulation bootstrap replicates. Use 100 for interactive runs; increase for final confidence intervals."
        self.caseCombo = qt.QComboBox()
        self.caseCombo.enabled = False
        self.caseCombo.minimumContentsLength = 18
        self.caseCombo.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        try:
            self.caseCombo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        except Exception:
            pass
        self.caseCombo.addItem("All cases", "all")
        self.caseCombo.currentIndexChanged.connect(self.on_case_combo_changed)

        discovery_form.addRow("Dataset root", self.datasetRootSelector)
        discovery_form.addRow("Resolved root", self.resolvedRootLabel)
        discovery_form.addRow("Cases", self.caseCombo)

        self.batchWorkflowGroup = qt.QGroupBox("Workflow")
        workflow_form = qt.QFormLayout(self.batchWorkflowGroup)
        layout.addWidget(self.batchWorkflowGroup)
        workflow_form.addRow("Profile", self.profileCombo)
        workflow_form.addRow("Bootstraps", self.bootstrapSpinBox)
        workflow_form.addRow("", self.overwriteCheckBox)

        action_row = qt.QHBoxLayout()
        self.runButton = qt.QPushButton("Run")
        self.stopButton = qt.QPushButton("Stop")
        self.stopButton.enabled = False
        self.runButton.clicked.connect(self.run)
        self.stopButton.clicked.connect(self.request_stop)
        self.runButton.minimumHeight = 34
        self.runButton.setStyleSheet(
            "QPushButton { font-weight: 700; background-color: #2f7ed8; color: white; "
            "border: 1px solid #1f5fa8; border-radius: 4px; padding: 6px 10px; }"
            "QPushButton:disabled { background-color: #9aa9b8; color: #f0f0f0; border-color: #8794a0; }"
        )
        action_row.addWidget(self.runButton, 2)
        action_row.addWidget(self.stopButton, 1)

        self.statusLabel = qt.QLabel("Idle")
        self.statusLabel.wordWrap = True
        self.statusLabel.setMinimumWidth(0)
        self.statusLabel.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        layout.addWidget(self.statusLabel)
        self.progressBar = qt.QProgressBar()
        self.progressBar.minimum = 0
        self.progressBar.maximum = 1
        self.progressBar.value = 0
        self.progressBar.visible = False
        self.progressBar.toolTip = "Current mechanoregulation run progress."
        self.currentStepLabel = qt.QLabel("Current step: idle")
        self.currentStepLabel.wordWrap = True
        self.currentStepLabel.setMinimumWidth(0)
        self.currentStepLabel.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        self.currentStepLabel.toolTip = "Currently running mechanoregulation step."
        layout.addWidget(self.progressBar)
        layout.addWidget(self.currentStepLabel)
        layout.addLayout(action_row)
        if parent_layout is not None:
            parent_layout.addStretch(1)

    def _build_scene_section(self, parent_layout):
        box = ctk.ctkCollapsibleButton()
        box.text = "Scene"
        parent_layout.addWidget(box)
        layout = qt.QVBoxLayout(box)

        self.sceneDiscoveryGroup = qt.QGroupBox("Discovery")
        discovery_layout = qt.QVBoxLayout(self.sceneDiscoveryGroup)
        layout.addWidget(self.sceneDiscoveryGroup)

        discover_row = qt.QHBoxLayout()
        self.sceneDiscoverButton = qt.QPushButton("Discover")
        discover_row.addWidget(self.sceneDiscoverButton)
        discover_row.addStretch(1)
        discovery_layout.addLayout(discover_row)

        self.sceneCaseTable = qt.QTableWidget()
        self.sceneCaseTable.setColumnCount(7)
        self.sceneCaseTable.setHorizontalHeaderLabels(
            ["Run", "Remodelling", "Baseline SED", "Baseline Seg", "Trab", "Cort", "Full"]
        )
        self.sceneCaseTable.minimumHeight = 150
        self.sceneCaseTable.maximumHeight = 260
        self.sceneCaseTable.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
        try:
            self.sceneCaseTable.horizontalHeader().setStretchLastSection(True)
            self.sceneCaseTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        except Exception:
            pass
        discovery_layout.addWidget(self.sceneCaseTable)

        self.sceneWorkflowGroup = qt.QGroupBox("Workflow")
        controls = qt.QFormLayout(self.sceneWorkflowGroup)
        layout.addWidget(self.sceneWorkflowGroup)
        self.sceneProfileCombo = qt.QComboBox()
        self.sceneProfileCombo.addItems(["XtremeCTII", "XtremeCTI"])
        self.sceneBootstrapSpinBox = qt.QSpinBox()
        self.sceneBootstrapSpinBox.minimum = 1
        self.sceneBootstrapSpinBox.maximum = 10000
        self.sceneBootstrapSpinBox.value = 100
        self.sceneOverwriteCheckBox = qt.QCheckBox("Overwrite existing outputs")
        controls.addRow("Profile", self.sceneProfileCombo)
        controls.addRow("Bootstraps", self.sceneBootstrapSpinBox)
        controls.addRow("", self.sceneOverwriteCheckBox)

        self.sceneStatusLabel = qt.QLabel("Discover loaded remodelling maps to run mechanoregulation from the scene.")
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
        self.sceneStopButton = qt.QPushButton("Stop")
        self.sceneStopButton.enabled = False
        self.sceneRunButton.minimumHeight = 34
        self.sceneRunButton.setStyleSheet(
            "QPushButton { font-weight: 700; background-color: #2f7ed8; color: white; "
            "border: 1px solid #1f5fa8; border-radius: 4px; padding: 6px 10px; }"
            "QPushButton:disabled { background-color: #9aa9b8; color: #f0f0f0; border-color: #8794a0; }"
        )
        button_row.addWidget(self.sceneRunButton, 2)
        button_row.addWidget(self.sceneStopButton, 1)
        layout.addLayout(button_row)

        self.sceneDiscoverButton.clicked.connect(self.discover_scene_cases)
        self.sceneRunButton.clicked.connect(self.run_scene)
        self.sceneStopButton.clicked.connect(self.request_stop)
        parent_layout.addStretch(1)

    def _build_review_section(self, parent_layout=None):
        box = ctk.ctkCollapsibleButton()
        box.text = "Review"
        (parent_layout or self.layout).addWidget(box)
        layout = qt.QVBoxLayout(box)

        self.metricsTable = qt.QTableWidget()
        self.metricsTable.setColumnCount(2)
        self.metricsTable.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metricsTable.setRowCount(len(METRIC_KEYS) + len(COUNT_KEYS))
        self.metricsTable.horizontalHeader().setStretchLastSection(True)
        self.metricsTable.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.metricsTable.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        for row, key in enumerate((*METRIC_KEYS, *COUNT_KEYS)):
            self.metricsTable.setItem(row, 0, qt.QTableWidgetItem(METRIC_LABELS.get(key, key)))
            self.metricsTable.setItem(row, 1, qt.QTableWidgetItem(""))
        layout.addWidget(self.metricsTable)

        roi_row = qt.QFormLayout()
        self.roiCombo = qt.QComboBox()
        self.roiCombo.addItem("Full", "full")
        self.roiCombo.currentIndexChanged.connect(self.refresh_review)
        roi_row.addRow("ROI", self.roiCombo)
        layout.addLayout(roi_row)

        button_row = qt.QGridLayout()
        self.refreshReviewButton = qt.QPushButton("Refresh review")
        self.loadSedButton = qt.QPushButton("Load SED")
        self.loadMaterialButton = qt.QPushButton("Load Material")
        self.loadRemodellingButton = qt.QPushButton("Load Remodelling")
        self.showCommandButton = qt.QPushButton("Show Command")
        self.openLogsButton = qt.QPushButton("Open Logs")
        self.refreshReviewButton.clicked.connect(self.refresh_review)
        self.loadSedButton.clicked.connect(lambda: self.load_selected_output("sed"))
        self.loadMaterialButton.clicked.connect(lambda: self.load_selected_output("material"))
        self.loadRemodellingButton.clicked.connect(lambda: self.load_selected_output("remodelling"))
        self.showCommandButton.clicked.connect(self.show_parosol_command)
        self.openLogsButton.clicked.connect(self.open_parosol_logs)
        buttons = (
            self.refreshReviewButton,
            self.loadSedButton,
            self.loadMaterialButton,
            self.loadRemodellingButton,
            self.showCommandButton,
            self.openLogsButton,
        )
        for index, button in enumerate(buttons):
            button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            button_row.addWidget(button, index // 3, index % 3)
        layout.addLayout(button_row)

        conditional_title = qt.QLabel("Conditional probability curves")
        layout.addWidget(conditional_title)
        self.conditionalCurvePreviewLabel = qt.QLabel()
        self._configure_curve_preview_label(self.conditionalCurvePreviewLabel)
        layout.addWidget(self.conditionalCurvePreviewLabel)

        schulte_title = qt.QLabel("Binned Schulte curves")
        layout.addWidget(schulte_title)
        self.schulteCurvePreviewLabel = qt.QLabel()
        self._configure_curve_preview_label(self.schulteCurvePreviewLabel)
        layout.addWidget(self.schulteCurvePreviewLabel)

        self.logText = qt.QPlainTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setLineWrapMode(qt.QPlainTextEdit.WidgetWidth)
        self.logText.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.logText.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        self.logText.maximumBlockCount = 500
        layout.addWidget(self.logText)
        if parent_layout is not None:
            parent_layout.addStretch(1)

    def _show(self, message):
        text = self._status_text(message)
        if text.startswith("[scene]"):
            if hasattr(self, "sceneStatusLabel"):
                self.sceneStatusLabel.text = text
            if hasattr(self, "sceneCurrentStepLabel"):
                self.sceneCurrentStepLabel.text = text
        else:
            self.statusLabel.text = text
            if hasattr(self, "currentStepLabel"):
                self.currentStepLabel.text = text
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
            if self._node_name_contains(node, ("sed", "strain", "fea"), exclude=("remodelling", "remodeling"))
        ]

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

    def discover_scene_cases(self):
        remodelling_nodes = self._scene_remodelling_candidates()
        sed_nodes = self._scene_sed_candidates()
        seg_nodes = self._scene_mask_candidates("seg")
        trab_nodes = self._scene_mask_candidates("trab")
        cort_nodes = self._scene_mask_candidates("cort")
        full_nodes = self._scene_mask_candidates("full")
        self._sceneCases = [{"remodelling_node": node} for node in remodelling_nodes]
        table = self.sceneCaseTable
        table.setRowCount(len(self._sceneCases))
        for row, case in enumerate(self._sceneCases):
            run_item = qt.QTableWidgetItem("")
            run_item.setFlags(run_item.flags() | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
            run_item.setCheckState(qt.Qt.Checked)
            table.setItem(row, 0, run_item)
            remodelling_item = qt.QTableWidgetItem(str(case["remodelling_node"].GetName()))
            remodelling_item.setFlags(remodelling_item.flags() & ~qt.Qt.ItemIsEditable)
            table.setItem(row, 1, remodelling_item)
            table.setCellWidget(row, 2, self._node_combo(sed_nodes, include_generate=True, default_generate=not sed_nodes))
            table.setCellWidget(row, 3, self._node_combo(seg_nodes, include_none=True))
            table.setCellWidget(row, 4, self._node_combo(trab_nodes, include_none=True))
            table.setCellWidget(row, 5, self._node_combo(cort_nodes, include_none=True))
            table.setCellWidget(row, 6, self._node_combo(full_nodes, include_none=True))
        try:
            table.resizeColumnsToContents()
        except Exception:
            pass
        self.sceneStatusLabel.text = (
            f"Discovered {len(remodelling_nodes)} remodelling map(s), {len(sed_nodes)} baseline SED candidate(s), "
            f"and {len(seg_nodes)} baseline segmentation candidate(s)."
        )

    def _scene_selected_rows(self):
        rows = []
        for row in range(self.sceneCaseTable.rowCount):
            item = self.sceneCaseTable.item(row, 0)
            if item is None or item.checkState() == qt.Qt.Checked:
                rows.append(row)
        return rows

    def _scene_combo_value(self, row, column):
        widget = self.sceneCaseTable.cellWidget(row, column)
        if widget is None:
            return "none"
        value = widget.currentData
        return str(value if value is not None else widget.currentText or "none")

    def _scene_node_from_combo(self, row, column):
        node_id = self._scene_combo_value(row, column)
        if node_id in {"", "none", "generate"}:
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

    def _stage_scene_case(self, row):
        case = self._sceneCases[row]
        remodelling_node = case["remodelling_node"]
        run_root = self._scene_run_root()
        input_dir = run_root / "input"
        output_dir = run_root / "output" / "derivatives" / "Mechanoregulation" / "sub-scene" / "site-scene" / "runs" / f"scene-row-{row + 1:02d}"
        remodelling_path = self._save_scene_node(remodelling_node, input_dir / f"scene-row-{row + 1:02d}_remodelling.nii.gz")
        sed_choice = self._scene_combo_value(row, 2)
        baseline_sed_path = None
        if sed_choice not in {"generate", "none"}:
            baseline_sed_path = self._save_scene_node(self._scene_node_from_combo(row, 2), input_dir / f"scene-row-{row + 1:02d}_sed.nii.gz")
        baseline_segmentation_path = self._save_scene_node(
            self._scene_node_from_combo(row, 3),
            input_dir / f"scene-row-{row + 1:02d}_seg.nii.gz",
        )
        trab_mask_path = self._save_scene_node(
            self._scene_node_from_combo(row, 4),
            input_dir / f"scene-row-{row + 1:02d}_mask-trab.nii.gz",
        )
        cort_mask_path = self._save_scene_node(
            self._scene_node_from_combo(row, 5),
            input_dir / f"scene-row-{row + 1:02d}_mask-cort.nii.gz",
        )
        full_mask_path = self._save_scene_node(
            self._scene_node_from_combo(row, 6),
            input_dir / f"scene-row-{row + 1:02d}_mask-full.nii.gz",
        )
        if sed_choice == "generate" and baseline_segmentation_path is None:
            raise ValueError("Generate baseline SED requires a baseline segmentation for each selected scene row.")
        return {
            "subject_id": "sub-scene",
            "case_id": f"scene-row-{row + 1:02d}",
            "baseline_image_path": remodelling_path,
            "remodelling_image_path": remodelling_path,
            "output_dir": output_dir,
            "baseline_segmentation_path": baseline_segmentation_path,
            "trab_mask_path": trab_mask_path,
            "cort_mask_path": cort_mask_path,
            "full_mask_path": full_mask_path,
            "baseline_sed_path": baseline_sed_path,
        }

    def run_scene(self):
        if self._running:
            slicer.util.warningDisplay("Mechanoregulation is already running.")
            return
        if not self._sceneCases:
            self.discover_scene_cases()
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
        self.sceneStopButton.enabled = True
        self.sceneProgressBar.visible = True
        self.sceneProgressBar.setRange(0, max(len(staged), 1))
        self.sceneProgressBar.value = 0
        profile = self.sceneProfileCombo.currentText
        overwrite = bool(self.sceneOverwriteCheckBox.checked)
        n_boot = int(self.sceneBootstrapSpinBox.value)
        self.sceneStatusLabel.text = f"Running 0/{len(staged)} scene mechanoregulation case(s)..."
        self.sceneCurrentStepLabel.text = "Current step: preparing"
        self._runThread = threading.Thread(
            target=self._run_scene_worker,
            args=(staged, profile, overwrite, n_boot),
            daemon=True,
        )
        self._runThread.start()
        self._runPollTimer.start()

    def _run_scene_worker(self, staged_cases, profile, overwrite, n_boot):
        summary = {
            "discovered": len(staged_cases),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "cancelled": False,
            "dry_run": False,
        }
        try:
            _prefer_local_core()
            from bonemechreg.timelapse import TimelapseCase, case_outputs
            from bonemechreg.post_timelapse import run_post_timelapse_case

            for index, staged in enumerate(staged_cases, start=1):
                if self._cancelEvent.is_set():
                    summary["cancelled"] = True
                    break
                self._runQueue.put(("progress", index - 1, f"[scene] {index}/{len(staged_cases)} {staged['case_id']}"))
                case = TimelapseCase(
                    subject_id=staged["subject_id"],
                    case_id=staged["case_id"],
                    baseline_image_path=Path(staged["baseline_image_path"]),
                    remodelling_image_path=Path(staged["remodelling_image_path"]),
                    output_dir=Path(staged["output_dir"]),
                    baseline_segmentation_path=staged.get("baseline_segmentation_path"),
                    trab_mask_path=staged.get("trab_mask_path"),
                    cort_mask_path=staged.get("cort_mask_path"),
                    full_mask_path=staged.get("full_mask_path"),
                )
                baseline_sed_path = staged.get("baseline_sed_path")
                if baseline_sed_path:
                    outputs = case_outputs(case)
                    outputs["sed"].parent.mkdir(parents=True, exist_ok=True)
                    outputs["sed"].write_bytes(Path(baseline_sed_path).read_bytes())
                result = run_post_timelapse_case(
                    case,
                    profile=profile,
                    overwrite=bool(overwrite and not baseline_sed_path),
                    n_boot=int(n_boot),
                    verbose=True,
                )
                summary["processed"] += int(result.get("processed", 0))
                summary["skipped"] += int(result.get("skipped", 0))
                summary["failed"] += int(result.get("failed", 0))
                self._runQueue.put(("progress", index, f"[scene] finished {index}/{len(staged_cases)} {staged['case_id']}"))
            self._runQueue.put(("scene_finished", summary))
        except Exception:
            self._runQueue.put(("scene_error", traceback.format_exc()))

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
        self.stopButton.enabled = False
        if hasattr(self, "sceneStopButton"):
            self.sceneStopButton.enabled = False
        self._show("[run] stop requested; finishing the active case before stopping")

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
        for staged in list(getattr(self, "_sceneRunRows", []) or []):
            try:
                _prefer_local_core()
                from bonemechreg.timelapse import TimelapseCase, case_outputs

                case = TimelapseCase(
                    subject_id=staged["subject_id"],
                    case_id=staged["case_id"],
                    baseline_image_path=Path(staged["baseline_image_path"]),
                    remodelling_image_path=Path(staged["remodelling_image_path"]),
                    output_dir=Path(staged["output_dir"]),
                    baseline_segmentation_path=staged.get("baseline_segmentation_path"),
                    trab_mask_path=staged.get("trab_mask_path"),
                    cort_mask_path=staged.get("cort_mask_path"),
                    full_mask_path=staged.get("full_mask_path"),
                )
                outputs = case_outputs(case)
                for key in ("sed", "material"):
                    path = outputs.get(key)
                    if path is not None and Path(path).exists():
                        if self._load_volume(path, key) is not None:
                            loaded += 1
            except Exception as exc:
                self._show(f"[scene] could not load outputs: {exc}")
        if loaded:
            self.sceneStatusLabel.text = f"{self.sceneStatusLabel.text} Loaded {loaded} output volume(s)."

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
