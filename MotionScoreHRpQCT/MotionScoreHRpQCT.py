import base64
import binascii
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from urllib import error as urllib_error
from urllib import request as urllib_request
from pathlib import Path

import ctk
import qt
import slicer
import vtk

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


MODULE_VERSION = "0.1.6"
MIN_CORE_VERSION = "2.5.4"
DEFAULT_MODEL_CATALOG_URL = (
    "https://github.com/wallematthias/MotionScoreHRpQCT/releases/latest/download/model_catalog.json"
)
LICENSE_HTTP_USER_AGENT = "MotionScoreSlicer/0.1 (+3D-Slicer; Python urllib)"
CORE_PYPI_PACKAGE = "motionscorehrpqct"
CORE_PIP_CONSTRAINTS = ("numpy<2.0",)
PROFILE_PLOT_LEFT_FRACTION = 0.11
PROFILE_PLOT_RIGHT_FRACTION = 0.89
TRAINING_PLOT_WIDTH = 760
TRAINING_PLOT_HEIGHT = 280


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class MotionScoreHRpQCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Motion Scoring"
        parent.categories = ["HR-pQCT"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "GUI wrapper for motionscore core CLI.\n"
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = "Built for streamlined HR-pQCT motion grading workflows."


class MotionScoreHRpQCTLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._proc = None
        self._user_terminated = False

    def is_running(self):
        return self._proc is not None

    def run_cli(self, args, on_output=None, on_finished=None):
        if self._proc is not None:
            raise RuntimeError("A motionscore process is already running")
        self._user_terminated = False

        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.MergedChannels)

        env = qt.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        # Avoid Slicer ITK ImageIO plugin autoloading noise in subprocess logs.
        if env.contains("ITK_AUTOLOAD_PATH"):
            env.remove("ITK_AUTOLOAD_PATH")
        if env.contains("SITK_AUTOLOAD_PATH"):
            env.remove("SITK_AUTOLOAD_PATH")
        env.insert("ITK_AUTOLOAD_PATH", "")
        env.insert("SITK_AUTOLOAD_PATH", "")

        proc.setProcessEnvironment(env)

        def _read_output():
            raw = proc.readAll()
            try:
                data = bytes(raw)
            except Exception:
                try:
                    data = raw.data()
                    if isinstance(data, str):
                        data = data.encode("utf-8", errors="replace")
                    else:
                        data = bytes(data)
                except Exception:
                    data = str(raw).encode("utf-8", errors="replace")
            text = data.decode("utf-8", errors="replace")
            if on_output and text:
                filtered_lines = []
                for line in text.splitlines(keepends=True):
                    if "MRMLIDImageIO" in line:
                        continue
                    if "ImageIO factory did not return an ImageIOBase" in line:
                        continue
                    filtered_lines.append(line)
                filtered = "".join(filtered_lines)
                if filtered:
                    on_output(filtered)

        def _finished(*signal_args):
            interrupted = bool(self._user_terminated)
            self._user_terminated = False
            self._proc = None
            exit_code = int(signal_args[0]) if len(signal_args) >= 1 else int(proc.exitCode())
            exit_status = signal_args[1] if len(signal_args) >= 2 else proc.exitStatus()
            if on_finished:
                on_finished(exit_code, exit_status, interrupted)

        proc.readyRead.connect(_read_output)
        proc.finished.connect(_finished)

        python_exe = (
            shutil.which("PythonSlicer")
            or (sys.executable if Path(sys.executable).exists() else None)
            or shutil.which("python3")
            or shutil.which("python")
        )
        if python_exe is None:
            raise RuntimeError("Could not find Python executable")

        full_args = ["-m", "motionscore.cli"] + list(args)
        if on_output:
            on_output(f"[process] launching: {python_exe} {' '.join(full_args)}\n")

        proc.start(python_exe, full_args)
        if not proc.waitForStarted(3000):
            raise RuntimeError("Failed to start motionscore process")

        self._proc = proc

    def interrupt(self):
        proc = self._proc
        if proc is None:
            return False
        self._user_terminated = True
        proc.terminate()

        def _force_kill_if_needed():
            if proc.state() != qt.QProcess.NotRunning:
                proc.kill()

        qt.QTimer.singleShot(1500, _force_kill_if_needed)
        return True


class _ProfileWheelEventFilter(qt.QObject):
    def __init__(self, owner_widget):
        super().__init__()
        self._owner_widget = owner_widget

    def eventFilter(self, obj, event):
        owner = self._owner_widget
        if owner is None:
            return False
        return bool(owner._handle_profile_wheel_event(obj, event))


class MotionScoreHRpQCTWidget(ScriptedLoadableModuleWidget):
    RUN_SCOPE_ALL = "All scans"
    RUN_MODE_AI = "AI Assisted"
    RUN_MODE_MANUAL = "Manual Grading Only"
    REVIEW_SCOPE_PENDING = "Pending only"
    REVIEW_SCOPE_ALL = "All scans"
    REVIEW_SCOPE_LOW_CONFIDENCE = "Low-confidence scans (re-review)"
    CLEAR_ALL_OPERATORS = "All operators"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic = MotionScoreHRpQCTLogic()
        self._index_rows = {}
        self._review_rows = {}
        self._active_task_name = None
        self._predict_total = 0
        self._predict_done = 0
        self._loaded_scan_id = ""
        self._loaded_volume_node = None
        self._profile_source_pixmap = None
        self._profile_scan_id = ""
        self._slice_observer_node = None
        self._slice_observer_tag = None
        self._all_scan_ids = []
        self._audit_rows = []
        self._auto_loading_scan = False
        self._grading_shortcuts = []
        self._selected_manual_grade = None
        self._grade_history = []
        self._model_profiles = []
        self._training_output_model_dir = None
        self._profile_wheel_filter = _ProfileWheelEventFilter(self)
        self._model_index_rows = {}
        self._model_review_rows = {}
        self._model_audit_rows = {}

    def setup(self):
        super().setup()

        self.licenseBox = ctk.ctkCollapsibleButton()
        self.licenseBox.text = "Models"
        self.licenseBox.collapsed = True
        self.layout.addWidget(self.licenseBox)

        licenseLayout = qt.QVBoxLayout(self.licenseBox)
        licenseForm = qt.QFormLayout()
        licenseLayout.addLayout(licenseForm)

        self.modelsPathEdit = ctk.ctkPathLineEdit()
        self.modelsPathEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.modelsPathEdit.settingKey = "MotionScore/InternalModelsRoot"
        self.modelsPathEdit.currentPath = str(self._default_models_path())
        self.modelsPathEdit.enabled = True
        licenseForm.addRow("Local models folder", self.modelsPathEdit)

        self.modelVersionEdit = qt.QLineEdit()
        self.modelVersionEdit.setText(self._settings().value("MotionScore/ModelVersion", "v1"))

        self.quickSetupButton = qt.QPushButton("Install / Download Models")
        licenseLayout.addWidget(self.quickSetupButton)

        self.licenseFlowHelpLabel = qt.QLabel(
            "Install the MotionScore package and download the default model weights."
        )
        self.licenseFlowHelpLabel.setWordWrap(True)
        licenseLayout.addWidget(self.licenseFlowHelpLabel)

        self.licenseStatusLabel = qt.QLabel("Models: not checked")
        self.licenseStatusLabel.setWordWrap(True)
        licenseLayout.addWidget(self.licenseStatusLabel)

        self.runBox = ctk.ctkCollapsibleButton()
        self.runBox.text = "Run"
        self.runBox.collapsed = False
        self.layout.addWidget(self.runBox)

        runLayout = qt.QVBoxLayout(self.runBox)
        runForm = qt.QFormLayout()
        runLayout.addLayout(runForm)

        self.datasetPathEdit = ctk.ctkPathLineEdit()
        self.datasetPathEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.datasetPathEdit.settingKey = "MotionScore/DatasetRoot"
        self.datasetPathEdit.currentPath = ""
        self.datasetPathEdit.setToolTip("Select dataset folder (browse starts at home when empty).")
        runForm.addRow("Dataset Root", self.datasetPathEdit)

        self.modelProfileCombo = qt.QComboBox()
        runForm.addRow("Model Profile", self.modelProfileCombo)

        self.sliceStepSpin = qt.QSpinBox()
        self.sliceStepSpin.minimum = 1
        self.sliceStepSpin.maximum = 512
        saved_slice_step_raw = self._settings().value("MotionScore/SliceStep", 1)
        try:
            saved_slice_step = int(saved_slice_step_raw or 1)
        except Exception:
            saved_slice_step = 1
        self.sliceStepSpin.value = max(1, saved_slice_step)
        self.sliceStepSpin.setToolTip("Fast mode: process every n-th slice during prediction. 1 means full scan.")

        runButtonsRow = qt.QHBoxLayout()
        self.loadDatasetButton = qt.QPushButton("Load Dataset")
        self.runButton = qt.QPushButton("Predict")
        self.manualRunButton = qt.QPushButton("Grade Manually")
        self.interruptButton = qt.QPushButton("Interrupt")
        self.interruptButton.enabled = False
        runButtonsRow.addWidget(self.loadDatasetButton)
        runButtonsRow.addWidget(self.runButton)
        runButtonsRow.addWidget(self.manualRunButton)
        runButtonsRow.addWidget(self.interruptButton)
        runLayout.addLayout(runButtonsRow)

        self.progressLabel = qt.QLabel("Idle")
        self.progressBar = qt.QProgressBar()
        self.progressBar.minimum = 0
        self.progressBar.maximum = 100
        self.progressBar.value = 0
        self.progressBar.textVisible = True
        runLayout.addWidget(self.progressLabel)
        runLayout.addWidget(self.progressBar)

        # Advanced options (moved out of main Run flow to declutter first-time usage).
        self.confidenceSpin = qt.QSpinBox()
        self.confidenceSpin.minimum = 0
        self.confidenceSpin.maximum = 100
        self.confidenceSpin.value = 75

        self.trainingModeCheck = qt.QCheckBox("Blind operator until manual grade is submitted")
        self.trainingModeCheck.setChecked(False)

        self.runModeCombo = qt.QComboBox()
        self.runModeCombo.addItems([self.RUN_MODE_AI, self.RUN_MODE_MANUAL])
        self.runModeCombo.setCurrentText(self.RUN_MODE_AI)

        self.runScopeCombo = qt.QComboBox()
        self.runScopeCombo.addItem(self.RUN_SCOPE_ALL)
        self.deviceCombo = qt.QComboBox()
        self.deviceCombo.addItems(["auto", "mps", "cpu", "cuda"])
        saved_device = str(self._settings().value("MotionScore/TorchDevice", "auto") or "auto").strip().lower()
        if saved_device not in {"auto", "mps", "cpu", "cuda"}:
            saved_device = "auto"
        self.deviceCombo.setCurrentText(saved_device)
        self.refreshButton = qt.QPushButton("Refresh Review")
        self.exportButton = qt.QPushButton("Export Final Grades")

        reviewBox = ctk.ctkCollapsibleButton()
        reviewBox.text = "Review"
        reviewLayout = qt.QFormLayout(reviewBox)

        self.scanCombo = qt.QComboBox()
        reviewLayout.addRow("Selected Scan", self.scanCombo)

        self.reviewScopeCombo = qt.QComboBox()
        self.reviewScopeCombo.addItems(
            [
                self.REVIEW_SCOPE_PENDING,
                self.REVIEW_SCOPE_ALL,
                self.REVIEW_SCOPE_LOW_CONFIDENCE,
            ]
        )
        reviewLayout.addRow("Review Scope", self.reviewScopeCombo)

        self.reviewQueueLabel = qt.QLabel("Queue: shown=0 | pending=0 | reviewed=0/0")
        reviewLayout.addRow("Review Queue", self.reviewQueueLabel)

        self.autoLabel = qt.QLabel("Auto grade: - | confidence: -")
        reviewLayout.addRow("Suggestion", self.autoLabel)

        self.agreementLabel = qt.QLabel("Agreement: overlap=0 | match=- | exact=-")
        reviewLayout.addRow("Operator vs AI", self.agreementLabel)

        self.agreementMatrixTable = qt.QTableWidget()
        self.agreementMatrixTable.setMinimumHeight(180)
        self.agreementMatrixTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.agreementMatrixTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.agreementMatrixTable.setWordWrap(True)
        self.agreementMatrixTable.setAlternatingRowColors(True)
        self.agreementMatrixTable.setSizeAdjustPolicy(qt.QAbstractScrollArea.AdjustToContents)
        self.agreementMatrixBox = ctk.ctkCollapsibleButton()
        self.agreementMatrixBox.text = "Agreement Matrix"
        self.agreementMatrixBox.collapsed = True
        agreementMatrixLayout = qt.QVBoxLayout(self.agreementMatrixBox)
        agreementMatrixLayout.addWidget(self.agreementMatrixTable)
        reviewLayout.addRow(self.agreementMatrixBox)

        self.trainingRevealLabel = qt.QLabel("Last training reveal: -")
        self.trainingRevealLabel.setWordWrap(True)
        reviewLayout.addRow("Training Reveal", self.trainingRevealLabel)

        self.profileModelCombo = qt.QComboBox()
        reviewLayout.addRow("Profile Model", self.profileModelCombo)

        self.profileLabel = qt.QLabel("Slice profile plot: -")
        self.profileLabel.setMinimumHeight(120)
        self.profileLabel.setMinimumWidth(180)
        self.profileLabel.setMaximumWidth(448)
        self.profileLabel.setMaximumHeight(240)
        self.profileLabel.setAlignment(qt.Qt.AlignCenter)
        self.profileLabel.setStyleSheet("QLabel { background: #ffffff; color: #333333; border: 1px solid #cfcfcf; }")
        self.sliceProfileBox = ctk.ctkCollapsibleButton()
        self.sliceProfileBox.text = "Slice Profile"
        self.sliceProfileBox.collapsed = False
        sliceProfileLayout = qt.QVBoxLayout(self.sliceProfileBox)
        sliceProfileLayout.addWidget(self.profileLabel)
        reviewLayout.addRow(self.sliceProfileBox)

        self.reviewerEdit = qt.QLineEdit()
        self.reviewerEdit.placeholderText = "reviewer id"
        self.reviewerEdit.setText(str(self._settings().value("MotionScore/Reviewer", "") or ""))
        reviewLayout.addRow("Reviewer", self.reviewerEdit)

        quickGradeRow = qt.QHBoxLayout()
        self.quickGradeButtons = {}
        quick_grade_styles = self._grade_style_map()
        for grade in range(1, 6):
            btn = qt.QPushButton(f"Grade {grade}")
            btn.setCheckable(True)
            btn.setToolTip(f"Save grade {grade} and move to next (shortcut: Ctrl+{grade})")
            palette = quick_grade_styles.get(
                grade,
                {"bg": "#555", "hover": "#4a4a4a", "pressed": "#444", "text": "#fff"},
            )
            btn.setStyleSheet(
                "QPushButton {"
                f" background-color: {palette['bg']};"
                f" color: {palette['text']};"
                " border: 1px solid rgba(0, 0, 0, 0.35);"
                " border-radius: 4px;"
                " padding: 4px 8px;"
                " font-weight: 600;"
                "}"
                f"QPushButton:hover {{ background-color: {palette['hover']}; }}"
                f"QPushButton:pressed {{ background-color: {palette['pressed']}; }}"
                "QPushButton:checked { border: 2px solid #ffffff; padding: 3px 7px; }"
                "QPushButton[suggested=\"true\"] { border: 5px solid #000000; padding: 0px 4px; }"
                "QPushButton:disabled { background-color: #888; color: #ddd; }"
            )
            btn.clicked.connect(lambda _checked=False, g=grade: self.onQuickSelectGrade(g))
            quickGradeRow.addWidget(btn)
            self.quickGradeButtons[grade] = btn
        reviewLayout.addRow("Quick Grade", quickGradeRow)

        self.autoLoadCheck = qt.QCheckBox("Auto-load selected scan")
        self.autoLoadCheck.setChecked(True)
        self.autoLoadCheck.setToolTip("Automatically load/reload the selected scan to reduce clicks.")

        reviewActionRow = qt.QHBoxLayout()
        self.backButton = qt.QPushButton("Back")
        self.backButton.setToolTip("Go back to the last graded scan to overwrite if needed.")
        self.backButton.enabled = False
        self.applyButton = qt.QPushButton("Save Grade + Next")
        self.applyButton.setVisible(False)
        self.importButton = qt.QPushButton("Import Final Grades")
        reviewActionRow.addWidget(self.backButton)
        reviewActionRow.addWidget(self.importButton)
        reviewActionRow.addWidget(self.exportButton)
        reviewLayout.addRow(reviewActionRow)

        self.clearReviewerCombo = qt.QComboBox()
        self.clearReviewerCombo.setEditable(True)
        self.clearReviewerCombo.addItem(self.CLEAR_ALL_OPERATORS)
        self.clearButton = qt.QPushButton("Clear Grades")
        self.loadScanButton = qt.QPushButton("Load / Reload Scan")

        self.layout.addWidget(reviewBox)

        self.additionalOptionsBox = ctk.ctkCollapsibleButton()
        self.additionalOptionsBox.text = "Additional Options"
        self.additionalOptionsBox.collapsed = True
        additionalLayout = qt.QVBoxLayout(self.additionalOptionsBox)

        self.setupStatusLabel = qt.QLabel("Core: checking... | Models: checking...")
        self.setupStatusLabel.setWordWrap(True)
        additionalLayout.addWidget(self.setupStatusLabel)

        additionalForm = qt.QFormLayout()
        additionalForm.addRow("Confidence Threshold", self.confidenceSpin)
        additionalForm.addRow("Training Mode", self.trainingModeCheck)
        additionalForm.addRow("Run Mode", self.runModeCombo)
        additionalForm.addRow("Fast: Every N-th Slice", self.sliceStepSpin)
        additionalForm.addRow("Run Scope", self.runScopeCombo)
        additionalForm.addRow("Auto Load", self.autoLoadCheck)
        additionalLayout.addLayout(additionalForm)

        additionalRunButtons = qt.QHBoxLayout()
        additionalRunButtons.addWidget(self.refreshButton)
        additionalLayout.addLayout(additionalRunButtons)

        additionalReviewButtons = qt.QHBoxLayout()
        additionalReviewButtons.addWidget(self.loadScanButton)
        additionalLayout.addLayout(additionalReviewButtons)

        clearRow = qt.QHBoxLayout()
        clearRow.addWidget(self.clearReviewerCombo)
        clearRow.addWidget(self.clearButton)
        additionalForm.addRow("Clear Reviewer", clearRow)

        self.retrainBox = ctk.ctkCollapsibleButton()
        self.retrainBox.text = "Retrain"
        self.retrainBox.collapsed = True
        retrainLayout = qt.QVBoxLayout(self.retrainBox)
        retrainForm = qt.QFormLayout()
        retrainLayout.addLayout(retrainForm)

        self.retrainModelIdEdit = qt.QLineEdit()
        self.retrainModelIdEdit.setText(str(self._settings().value("MotionScore/RetrainModelId", "") or ""))
        self.retrainModelIdEdit.setPlaceholderText("custom-v1")
        retrainForm.addRow("New Model ID", self.retrainModelIdEdit)

        self.retrainDisplayNameEdit = qt.QLineEdit()
        self.retrainDisplayNameEdit.setText(str(self._settings().value("MotionScore/RetrainDisplayName", "") or ""))
        self.retrainDisplayNameEdit.setPlaceholderText("Custom retrain")
        retrainForm.addRow("Display Name", self.retrainDisplayNameEdit)

        self.retrainBaseModelCombo = qt.QComboBox()
        retrainForm.addRow("Base Model", self.retrainBaseModelCombo)

        self.retrainSliceCountSpin = qt.QSpinBox()
        self.retrainSliceCountSpin.minimum = 0
        self.retrainSliceCountSpin.maximum = 128
        self.retrainSliceCountSpin.value = int(self._settings().value("MotionScore/RetrainSliceCount", 8) or 8)
        retrainForm.addRow("Slices Per Scan", self.retrainSliceCountSpin)

        self.retrainIncludeAutoCheck = qt.QCheckBox("Include confident AI-only labels")
        self.retrainIncludeAutoCheck.setChecked(bool(int(self._settings().value("MotionScore/RetrainIncludeAuto", 1) or 1)))
        self.retrainIncludeAutoCheck.setToolTip("Include high-confidence automatic slice labels for scans without manual grades.")
        retrainForm.addRow("Manifest Labels", self.retrainIncludeAutoCheck)

        self.retrainPatienceSpin = qt.QSpinBox()
        self.retrainPatienceSpin.minimum = 0
        self.retrainPatienceSpin.maximum = 100
        self.retrainPatienceSpin.value = int(self._settings().value("MotionScore/RetrainPatience", 10) or 10)
        retrainForm.addRow("Early Stopping Patience", self.retrainPatienceSpin)

        self.retrainSeedSpin = qt.QSpinBox()
        self.retrainSeedSpin.minimum = 0
        self.retrainSeedSpin.maximum = 2_147_483_647
        self.retrainSeedSpin.value = int(self._settings().value("MotionScore/RetrainSeed", 13) or 13)
        retrainForm.addRow("Random Seed", self.retrainSeedSpin)

        self.deviceCombo.setToolTip("Torch device used for prediction and retraining.")
        retrainForm.addRow("Torch Device", self.deviceCombo)

        self.retrainAugHFlipCheck = qt.QCheckBox("Horizontal flip")
        self.retrainAugHFlipCheck.setChecked(bool(int(self._settings().value("MotionScore/RetrainAugHFlip", 1) or 1)))
        retrainForm.addRow("Augmentation", self.retrainAugHFlipCheck)

        self.retrainAugVFlipCheck = qt.QCheckBox("Vertical flip")
        self.retrainAugVFlipCheck.setChecked(bool(int(self._settings().value("MotionScore/RetrainAugVFlip", 1) or 1)))
        retrainForm.addRow("", self.retrainAugVFlipCheck)

        self.retrainAugRotateCheck = qt.QCheckBox("Rotate 90°")
        self.retrainAugRotateCheck.setChecked(bool(int(self._settings().value("MotionScore/RetrainAugRotate", 0) or 0)))
        retrainForm.addRow("", self.retrainAugRotateCheck)

        self.retrainAugCropCheck = qt.QCheckBox("Random crop")
        self.retrainAugCropCheck.setChecked(bool(int(self._settings().value("MotionScore/RetrainAugCrop", 0) or 0)))
        retrainForm.addRow("", self.retrainAugCropCheck)

        self.retrainEpochsHeadSpin = qt.QSpinBox()
        self.retrainEpochsHeadSpin.minimum = 0
        self.retrainEpochsHeadSpin.maximum = 500
        self.retrainEpochsHeadSpin.value = int(self._settings().value("MotionScore/RetrainEpochsHead", 20) or 20)
        retrainForm.addRow("Classifier Epochs", self.retrainEpochsHeadSpin)

        self.retrainEpochsFineSpin = qt.QSpinBox()
        self.retrainEpochsFineSpin.minimum = 0
        self.retrainEpochsFineSpin.maximum = 1000
        self.retrainEpochsFineSpin.value = int(self._settings().value("MotionScore/RetrainEpochsFine", 50) or 50)
        retrainForm.addRow("Full Epochs", self.retrainEpochsFineSpin)

        retrainButtonsRow = qt.QHBoxLayout()
        self.prepareRetrainButton = qt.QPushButton("Prepare Retrain Manifest")
        self.trainHeadButton = qt.QPushButton("Train Classifier")
        self.trainFullButton = qt.QPushButton("Train Full Model")
        self.continueTrainButton = qt.QPushButton("Continue Training")
        self.trainInterruptButton = qt.QPushButton("Interrupt")
        self.trainInterruptButton.enabled = False
        retrainButtonsRow.addWidget(self.prepareRetrainButton)
        retrainButtonsRow.addWidget(self.trainHeadButton)
        retrainButtonsRow.addWidget(self.trainFullButton)
        retrainButtonsRow.addWidget(self.continueTrainButton)
        retrainButtonsRow.addWidget(self.trainInterruptButton)
        retrainLayout.addLayout(retrainButtonsRow)

        self.trainingMetricsLabel = qt.QLabel("Holdout: -")
        self.trainingMetricsLabel.setWordWrap(True)
        retrainLayout.addWidget(self.trainingMetricsLabel)

        self.trainingPlotLabel = qt.QLabel("Training plot: -")
        self.trainingPlotLabel.setAlignment(qt.Qt.AlignCenter)
        self.trainingPlotLabel.setMinimumHeight(TRAINING_PLOT_HEIGHT)
        self.trainingPlotLabel.setMinimumWidth(0)
        self.trainingPlotLabel.setMaximumHeight(TRAINING_PLOT_HEIGHT + 20)
        self.trainingPlotLabel.setStyleSheet("QLabel { background: #ffffff; color: #333333; border: 1px solid #cfcfcf; }")
        retrainLayout.addWidget(self.trainingPlotLabel)

        self.layout.addWidget(self.retrainBox)
        self.layout.addWidget(self.additionalOptionsBox)

        self.consoleBox = ctk.ctkCollapsibleButton()
        self.consoleBox.text = "Console"
        self.consoleBox.collapsed = False
        consoleLayout = qt.QVBoxLayout(self.consoleBox)

        self.logText = qt.QPlainTextEdit()
        self.logText.readOnly = True
        self.logText.setMaximumBlockCount(5000)
        consoleLayout.addWidget(self.logText)
        self.layout.addWidget(self.consoleBox)

        self.loadDatasetButton.clicked.connect(self.onLoadDataset)
        self.runButton.clicked.connect(self.onRunPredict)
        self.manualRunButton.clicked.connect(self.onRunManualOnly)
        self.interruptButton.clicked.connect(self.onInterrupt)
        self.trainInterruptButton.clicked.connect(self.onInterrupt)
        self.refreshButton.clicked.connect(self.onRefreshReview)
        self.exportButton.clicked.connect(self.onExport)
        self.importButton.clicked.connect(self.onImportFinalGrades)
        self.quickSetupButton.clicked.connect(self.onQuickSetup)
        self.backButton.clicked.connect(self.onBackToPreviousScan)
        self.clearButton.clicked.connect(self.onClearGrades)
        self.loadScanButton.clicked.connect(self.onLoadSelectedScan)
        self.prepareRetrainButton.clicked.connect(self.onPrepareRetrainManifest)
        self.trainHeadButton.clicked.connect(self.onTrainClassifierOnly)
        self.trainFullButton.clicked.connect(self.onTrainFullModel)
        self.continueTrainButton.clicked.connect(self.onContinueTraining)
        self.scanCombo.currentTextChanged.connect(self.onScanSelectionChanged)
        self.reviewScopeCombo.currentTextChanged.connect(self.onReviewScopeChanged)
        self.confidenceSpin.valueChanged.connect(self.onConfidenceThresholdChanged)
        self.datasetPathEdit.currentPathChanged.connect(self.onDatasetPathChanged)
        self.trainingModeCheck.toggled.connect(self.onTrainingModeToggled)
        self.reviewerEdit.editingFinished.connect(self._persist_reviewer_setting)
        self.deviceCombo.currentTextChanged.connect(self._persist_runtime_settings)
        self.runModeCombo.currentTextChanged.connect(self._persist_runtime_settings)
        self.modelProfileCombo.currentTextChanged.connect(self._persist_runtime_settings)
        self.retrainBaseModelCombo.currentTextChanged.connect(self._persist_runtime_settings)
        self.profileModelCombo.currentTextChanged.connect(self.onProfileModelChanged)
        self.sliceStepSpin.valueChanged.connect(self._persist_runtime_settings)
        self.retrainModelIdEdit.editingFinished.connect(self._persist_runtime_settings)
        self.retrainDisplayNameEdit.editingFinished.connect(self._persist_runtime_settings)
        self.retrainSliceCountSpin.valueChanged.connect(self._persist_runtime_settings)
        self.retrainIncludeAutoCheck.toggled.connect(self._persist_runtime_settings)
        self.retrainPatienceSpin.valueChanged.connect(self._persist_runtime_settings)
        self.retrainSeedSpin.valueChanged.connect(self._persist_runtime_settings)
        self.retrainAugHFlipCheck.toggled.connect(self._persist_runtime_settings)
        self.retrainAugVFlipCheck.toggled.connect(self._persist_runtime_settings)
        self.retrainAugRotateCheck.toggled.connect(self._persist_runtime_settings)
        self.retrainAugCropCheck.toggled.connect(self._persist_runtime_settings)
        self.retrainEpochsHeadSpin.valueChanged.connect(self._persist_runtime_settings)
        self.retrainEpochsFineSpin.valueChanged.connect(self._persist_runtime_settings)

        self._install_grading_shortcuts()

        self.layout.addStretch(1)
        self._update_setup_status()
        self._refresh_model_profiles()
        qt.QTimer.singleShot(0, self._install_slice_observer)
        qt.QTimer.singleShot(0, self._install_profile_wheel_filter)

    def _install_profile_wheel_filter(self):
        try:
            app = qt.QApplication.instance()
            if app is None:
                return
            try:
                app.removeEventFilter(self._profile_wheel_filter)
            except Exception:
                pass
            app.installEventFilter(self._profile_wheel_filter)
        except Exception as exc:
            self._log(f"[profile] could not install wheel filter: {exc}\n")

    def _default_models_path(self):
        app_data = str(qt.QStandardPaths.writableLocation(qt.QStandardPaths.AppDataLocation) or "").strip()
        if app_data:
            root = Path(app_data)
        else:
            root = Path.home() / ".motionscore"
        models_root = (root / "MotionScore" / "models").resolve()
        models_root.mkdir(parents=True, exist_ok=True)
        return models_root

    def _settings(self):
        return qt.QSettings()

    def _persist_license_settings(self):
        s = self._settings()
        s.setValue("MotionScore/ModelVersion", self.modelVersionEdit.text.strip())

    def _persist_reviewer_setting(self):
        self._settings().setValue("MotionScore/Reviewer", self.reviewerEdit.text.strip())

    def _persist_runtime_settings(self):
        self._settings().setValue("MotionScore/TorchDevice", self._combo_text(self.deviceCombo))
        self._settings().setValue("MotionScore/RunMode", self._combo_text(self.runModeCombo))
        self._settings().setValue("MotionScore/ModelProfile", self._selected_model_id())
        self._settings().setValue("MotionScore/SliceStep", int(self.sliceStepSpin.value))
        self._settings().setValue("MotionScore/RetrainModelId", self.retrainModelIdEdit.text.strip())
        self._settings().setValue("MotionScore/RetrainDisplayName", self.retrainDisplayNameEdit.text.strip())
        self._settings().setValue("MotionScore/RetrainBaseModel", self._selected_retrain_base_model_id())
        self._settings().setValue("MotionScore/RetrainSliceCount", int(self.retrainSliceCountSpin.value))
        auto_attr = self.retrainIncludeAutoCheck.checked
        self._settings().setValue("MotionScore/RetrainIncludeAuto", 1 if bool(auto_attr() if callable(auto_attr) else auto_attr) else 0)
        self._settings().setValue("MotionScore/RetrainPatience", int(self.retrainPatienceSpin.value))
        self._settings().setValue("MotionScore/RetrainSeed", int(self.retrainSeedSpin.value))
        aug_h_attr = self.retrainAugHFlipCheck.checked
        aug_v_attr = self.retrainAugVFlipCheck.checked
        aug_rotate_attr = self.retrainAugRotateCheck.checked
        aug_crop_attr = self.retrainAugCropCheck.checked
        self._settings().setValue("MotionScore/RetrainAugHFlip", 1 if bool(aug_h_attr() if callable(aug_h_attr) else aug_h_attr) else 0)
        self._settings().setValue("MotionScore/RetrainAugVFlip", 1 if bool(aug_v_attr() if callable(aug_v_attr) else aug_v_attr) else 0)
        self._settings().setValue("MotionScore/RetrainAugRotate", 1 if bool(aug_rotate_attr() if callable(aug_rotate_attr) else aug_rotate_attr) else 0)
        self._settings().setValue("MotionScore/RetrainAugCrop", 1 if bool(aug_crop_attr() if callable(aug_crop_attr) else aug_crop_attr) else 0)
        self._settings().setValue("MotionScore/RetrainEpochsHead", int(self.retrainEpochsHeadSpin.value))
        self._settings().setValue("MotionScore/RetrainEpochsFine", int(self.retrainEpochsFineSpin.value))

    def _install_grading_shortcuts(self):
        parent_widget = self.parent if isinstance(self.parent, qt.QWidget) else slicer.util.mainWindow()
        if parent_widget is None:
            return
        self._grading_shortcuts = []
        for grade in range(1, 6):
            shortcut = qt.QShortcut(qt.QKeySequence(f"Ctrl+{grade}"), parent_widget)
            shortcut.setContext(qt.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda g=grade: self.onQuickSelectGrade(g))
            self._grading_shortcuts.append(shortcut)

    def _auto_load_enabled(self):
        checked_attr = self.autoLoadCheck.checked
        return bool(checked_attr() if callable(checked_attr) else checked_attr)

    def _set_selected_manual_grade(self, grade, suggested=False):
        normalized = None
        try:
            value = int(grade)
            if 1 <= value <= 5:
                normalized = value
        except Exception:
            normalized = None
        self._selected_manual_grade = normalized
        for button_grade, button in self.quickGradeButtons.items():
            is_selected = bool(normalized is not None and button_grade == normalized)
            button.setChecked(is_selected)
            button.setProperty("suggested", bool(suggested and is_selected))
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()
        current_scan = self._combo_text(self.scanCombo) if hasattr(self, "scanCombo") else ""
        if current_scan:
            self._render_profile_plot(current_scan)

    def _grade_style_map(self):
        return {
            1: {"bg": "#125FAF", "hover": "#0F518F", "pressed": "#0C4579", "text": "#FFFFFF", "plot": "#125FAF"},
            2: {"bg": "#4B97E0", "hover": "#3F86C9", "pressed": "#366FA8", "text": "#FFFFFF", "plot": "#4B97E0"},
            3: {"bg": "#F2C94C", "hover": "#E2BA45", "pressed": "#C79F39", "text": "#1A1A1A", "plot": "#F2C94C"},
            4: {"bg": "#E56A5D", "hover": "#D25E52", "pressed": "#B85248", "text": "#FFFFFF", "plot": "#E56A5D"},
            5: {"bg": "#B73A31", "hover": "#A4342C", "pressed": "#8E2D26", "text": "#FFFFFF", "plot": "#B73A31"},
        }

    def _grade_plot_color(self, grade):
        style = self._grade_style_map().get(int(grade), self._grade_style_map()[3])
        return qt.QColor(style["plot"])

    def _set_license_status(self, text):
        self.licenseStatusLabel.setText(str(text))
        self._update_setup_status()

    def _default_device_hash(self):
        machine = platform.machine()
        node = platform.node()
        mac = f"{uuid.getnode():012x}"
        raw = f"{machine}|{node}|{mac}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def _http_json(self, method, url, payload=None, headers=None):
        headers = dict(headers or {})
        headers.setdefault("accept", "application/json")
        headers.setdefault("user-agent", LICENSE_HTTP_USER_AGENT)
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"
        req = urllib_request.Request(url=url, data=body, method=str(method).upper())
        for k, v in headers.items():
            req.add_header(str(k), str(v))
        try:
            with urllib_request.urlopen(req, timeout=30) as resp:
                data = resp.read()
        except urllib_error.HTTPError as exc:
            err_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {err_text}") from exc
        except Exception as exc:
            raise RuntimeError(f"Network error: {exc}") from exc

        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON response: {text[:200]}") from exc

    def _log(self, text):
        self.logText.moveCursor(qt.QTextCursor.End)
        self.logText.insertPlainText(text)
        self.logText.moveCursor(qt.QTextCursor.End)

    def _combo_data(self, combo):
        try:
            idx_attr = combo.currentIndex
            idx = idx_attr() if callable(idx_attr) else idx_attr
            return combo.itemData(idx)
        except Exception:
            return None

    def _selected_model_id(self):
        data = self._combo_data(self.modelProfileCombo)
        if data:
            return str(data).strip()
        return "base-v1"

    def _selected_retrain_base_model_id(self):
        data = self._combo_data(self.retrainBaseModelCombo)
        if data:
            return str(data).strip()
        return self._selected_model_id()

    def _selected_profile_model_id(self):
        data = self._combo_data(self.profileModelCombo)
        if data:
            return str(data).strip()
        return ""

    def _model_display_name(self, model_id):
        model_id_txt = str(model_id or "").strip()
        if not model_id_txt:
            return "Unknown model"
        for entry in self._model_profiles:
            if str(entry.get("model_id", "")).strip() == model_id_txt:
                return str(entry.get("display_name", "")).strip() or model_id_txt
        return model_id_txt

    def _model_display_label(self, model_id):
        model_id_txt = str(model_id or "").strip()
        display = self._model_display_name(model_id_txt)
        if display == model_id_txt:
            return display
        return f"{display} ({model_id_txt})"

    def _first_row_from_tsv(self, path):
        if path is None or not path.exists():
            return {}
        try:
            rows = read_tsv(path)
        except Exception:
            return {}
        return dict(rows[0]) if rows else {}

    def _dir_named(self, path, dirname):
        current = Path(path).resolve()
        for candidate in [current] + list(current.parents):
            if candidate.name == dirname:
                return candidate
        return None

    def _discover_model_entries_for_scan(self, derivatives, index_row):
        scan_id = str(index_row.get("scan_id", "")).strip()
        active_model_id = str(index_row.get("model_id", "")).strip()
        out = {}

        def _add_entry(model_id, pred_path=None, review_path=None, audit_path=None, index_template=None):
            model_id_txt = str(model_id or "").strip() or "base-v1"
            pred_row = self._first_row_from_tsv(pred_path)
            if pred_row:
                model_id_txt = str(pred_row.get("model_id", "")).strip() or model_id_txt
            merged_index = dict(index_template or index_row)
            merged_index["model_id"] = model_id_txt
            if pred_path is not None and pred_path.exists():
                merged_index["predictions_tsv"] = os.path.relpath(str(pred_path.resolve()), str(derivatives.resolve()))
            if review_path is not None and review_path.exists():
                merged_index["review_tsv"] = os.path.relpath(str(review_path.resolve()), str(derivatives.resolve()))
            if audit_path is not None and audit_path.exists():
                merged_index["review_audit"] = os.path.relpath(str(audit_path.resolve()), str(derivatives.resolve()))
            if pred_row:
                for key in (
                    "preview_png_path",
                    "slice_profile_png_path",
                    "automatic_grade",
                    "automatic_confidence",
                    "raw_image_path",
                    "model_version",
                    "predicted_at",
                ):
                    if str(pred_row.get(key, "")).strip():
                        merged_index[key] = pred_row.get(key, "")
            review_row = self._first_row_from_tsv(review_path)
            audits = []
            if audit_path is not None and audit_path.exists():
                try:
                    audits = read_tsv(audit_path)
                except Exception:
                    audits = []
            out[model_id_txt] = {
                "index": merged_index,
                "review": review_row,
                "audits": audits,
            }

        pred_rel = str(index_row.get("predictions_tsv", "")).strip()
        review_rel = str(index_row.get("review_tsv", "")).strip()
        audit_rel = str(index_row.get("review_audit", "")).strip()
        pred_path = (derivatives / pred_rel).resolve() if pred_rel else None
        review_path = (derivatives / review_rel).resolve() if review_rel else None
        audit_path = (derivatives / audit_rel).resolve() if audit_rel else None

        if pred_path is not None or review_path is not None:
            _add_entry(active_model_id or "base-v1", pred_path=pred_path, review_path=review_path, audit_path=audit_path)

        pred_root = self._dir_named(pred_path, "predictions") if pred_path is not None else None
        review_root = self._dir_named(review_path, "review") if review_path is not None else None
        models_pred_root = pred_root / "models" if pred_root is not None else None
        if models_pred_root is not None and models_pred_root.exists():
            for model_pred_path in sorted(models_pred_root.glob("*/predictions.tsv")):
                model_id = model_pred_path.parent.name
                model_review_path = (review_root / "models" / model_id / "review.tsv") if review_root is not None else None
                model_audit_path = (review_root / "models" / model_id / "review_audit.tsv") if review_root is not None else None
                _add_entry(model_id, pred_path=model_pred_path, review_path=model_review_path, audit_path=model_audit_path)

        if scan_id and not out:
            self._log(f"[models] no model artifacts discovered for {scan_id}\n")
        return out

    def _refresh_profile_model_combo(self, scan_id):
        previous = self._selected_profile_model_id()
        active_model_id = str(self._index_rows.get(scan_id, {}).get("model_id", "")).strip()
        entries = self._model_index_rows.get(scan_id, {})
        candidate_ids = []
        for model_id, row in sorted(entries.items(), key=lambda item: self._model_display_label(item[0]).casefold()):
            if str(row.get("slice_profile_png_path", "")).strip() or str(row.get("preview_png_path", "")).strip():
                candidate_ids.append(model_id)

        self.profileModelCombo.blockSignals(True)
        self.profileModelCombo.clear()
        for model_id in candidate_ids:
            self.profileModelCombo.addItem(self._model_display_label(model_id), model_id)

        if candidate_ids:
            target = previous if previous in candidate_ids else (active_model_id if active_model_id in candidate_ids else candidate_ids[0])
            for idx in range(self._combo_count(self.profileModelCombo)):
                if str(self.profileModelCombo.itemData(idx) or "").strip() == target:
                    self.profileModelCombo.setCurrentIndex(idx)
                    break
            self.profileModelCombo.enabled = True
        else:
            self.profileModelCombo.addItem("Current model", "")
            self.profileModelCombo.setCurrentIndex(0)
            self.profileModelCombo.enabled = False
        self.profileModelCombo.blockSignals(False)

    def _refresh_model_profiles(self):
        previous = str(self._settings().value("MotionScore/ModelProfile", "base-v1") or "base-v1").strip()
        previous_retrain = str(self._settings().value("MotionScore/RetrainBaseModel", previous) or previous).strip()
        models_dir = self._models_dir()
        profiles = []
        if models_dir is not None:
            try:
                from motionscore.model_registry import list_model_profiles

                profiles = list_model_profiles(models_dir)
            except Exception as exc:
                self._log(f"[models] could not read model registry: {exc}\n")
            if not profiles and self._has_local_models(models_dir):
                profiles = [{"model_id": "base-v1", "display_name": "Base v1", "version": "v1"}]

        self._model_profiles = list(profiles)
        self.modelProfileCombo.blockSignals(True)
        self.retrainBaseModelCombo.blockSignals(True)
        self.modelProfileCombo.clear()
        self.retrainBaseModelCombo.clear()
        for entry in self._model_profiles:
            model_id = str(entry.get("model_id", "")).strip() or "base-v1"
            display = str(entry.get("display_name", "")).strip() or model_id
            version = str(entry.get("version", "")).strip()
            label = f"{display} ({model_id})" if not version else f"{display} ({model_id}@{version})"
            self.modelProfileCombo.addItem(label, model_id)
            self.retrainBaseModelCombo.addItem(label, model_id)
        count_attr = self.modelProfileCombo.count
        count = int(count_attr() if callable(count_attr) else count_attr)
        if count == 0:
            self.modelProfileCombo.addItem("Base v1 (base-v1)", "base-v1")
            self.retrainBaseModelCombo.addItem("Base v1 (base-v1)", "base-v1")
            count = 1
        for idx in range(count):
            data = str(self.modelProfileCombo.itemData(idx) or "").strip()
            if data == previous:
                self.modelProfileCombo.setCurrentIndex(idx)
                break
        retrain_target = previous_retrain or previous or "base-v1"
        for idx in range(count):
            data = str(self.retrainBaseModelCombo.itemData(idx) or "").strip()
            if data == retrain_target:
                self.retrainBaseModelCombo.setCurrentIndex(idx)
                break
        self.modelProfileCombo.blockSignals(False)
        self.retrainBaseModelCombo.blockSignals(False)

    def _training_root(self):
        derivatives = self._derivatives_root()
        if derivatives is None:
            return None
        return derivatives / "training"

    def _retrain_manifest_path(self):
        root = self._training_root()
        return None if root is None else root / "train_manifest.tsv"

    def _normalized_retrain_model_id(self):
        raw = self.retrainModelIdEdit.text.strip()
        if raw:
            return re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-") or "custom-model"
        return f"custom-{qt.QDateTime.currentDateTime().toString('yyyyMMdd-HHmmss')}"

    def _retrain_output_model_dir(self):
        models_root = self._models_dir()
        if models_root is None:
            return None
        return models_root / self._normalized_retrain_model_id()

    def _retrain_cv_folds(self):
        models_root = self._models_dir()
        if models_root is None:
            return 10
        model_id = self._selected_retrain_base_model_id()
        checkpoint_count = 0
        try:
            from motionscore.model_registry import resolve_model_dir

            model_dir, _profile = resolve_model_dir(model_root=models_root, model_id=model_id)
            checkpoint_count = len(list(Path(model_dir).glob("DNN_*.pt")))
        except Exception:
            fallback_dir = (Path(models_root) / str(model_id).strip()).resolve()
            if fallback_dir.exists():
                checkpoint_count = len(list(fallback_dir.glob("DNN_*.pt")))
        return int(max(2, checkpoint_count if checkpoint_count > 0 else 10))

    def _retrain_continue_lr(self, output_model_dir):
        metrics_path = Path(output_model_dir) / "training_metrics.json"
        default_lr = 1e-4
        if not metrics_path.exists():
            return default_lr
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return default_lr
        try:
            summary_lr = float(payload.get("lr_finetune", default_lr))
        except Exception:
            summary_lr = default_lr
        models = payload.get("models", []) if isinstance(payload, dict) else []
        for model in models:
            points = list(model.get("plot_points", [])) if isinstance(model, dict) else []
            for point in reversed(points):
                if str(point.get("stage", "")).strip() != "finetune":
                    continue
                try:
                    lr = float(point.get("lr", summary_lr))
                except Exception:
                    lr = summary_lr
                if lr > 0.0:
                    return lr
        return summary_lr if summary_lr > 0.0 else default_lr

    def _update_training_plot(self, final=False):
        model_dir = self._training_output_model_dir
        if model_dir is None:
            return
        plot_name = "training_plot.png" if final else "training_plot_live.png"
        plot_path = Path(model_dir) / plot_name
        if not plot_path.exists() and final:
            plot_path = Path(model_dir) / "training_plot_live.png"
        if not plot_path.exists():
            return
        pix = qt.QPixmap(str(plot_path))
        if pix.isNull():
            return
        width_attr = self.trainingPlotLabel.width
        current_width = int(width_attr() if callable(width_attr) else width_attr)
        target_width = int(max(320, min(TRAINING_PLOT_WIDTH, current_width)))
        scaled = pix.scaled(target_width, TRAINING_PLOT_HEIGHT, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
        self.trainingPlotLabel.setPixmap(scaled)
        self.trainingPlotLabel.setText("")

    def _update_training_summary(self):
        model_dir = self._training_output_model_dir
        if model_dir is None:
            return
        metrics_path = Path(model_dir) / "training_metrics.json"
        if not metrics_path.exists():
            return
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"[train] could not read metrics: {exc}\n")
            return
        models = payload.get("models", []) if isinstance(payload, dict) else []
        test_metrics = [m.get("test", {}) for m in models if isinstance(m, dict)]
        if not test_metrics:
            self.trainingMetricsLabel.text = "Holdout: metrics unavailable"
            return
        mean_acc = sum(float(m.get("accuracy", 0.0)) for m in test_metrics) / float(len(test_metrics))
        mean_kappa = sum(float(m.get("weighted_kappa", 0.0)) for m in test_metrics) / float(len(test_metrics))
        base_metrics = payload.get("base_model_test", {}) if isinstance(payload, dict) else {}
        improvement = payload.get("test_improvement", {}) if isinstance(payload, dict) else {}
        base_acc = float(base_metrics.get("accuracy", 0.0)) if base_metrics else 0.0
        base_kappa = float(base_metrics.get("weighted_kappa", 0.0)) if base_metrics else 0.0
        delta_acc = float(improvement.get("accuracy", mean_acc - base_acc))
        delta_kappa = float(improvement.get("weighted_kappa", mean_kappa - base_kappa))
        self.trainingMetricsLabel.text = (
            f"Holdout: models={len(test_metrics)} | base acc={base_acc:.3f} -> retrain acc={mean_acc:.3f} "
            f"(delta={delta_acc:+.3f}) | base kappa={base_kappa:.3f} -> retrain kappa={mean_kappa:.3f} "
            f"(delta={delta_kappa:+.3f})"
        )

    def _derivatives_root(self):
        dataset = self.datasetPathEdit.currentPath.strip()
        if not dataset:
            return None
        return Path(dataset) / "MotionScore"

    def _models_dir(self):
        path = str(self.modelsPathEdit.currentPath).strip()
        if not path:
            path = str(self._default_models_path())
            self.modelsPathEdit.currentPath = path
        models_dir = Path(path).resolve()
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def _has_local_models(self, models_dir=None):
        models_dir = models_dir or self._models_dir()
        if models_dir is None:
            return False
        models_dir = Path(models_dir)
        try:
            from motionscore.model_registry import list_model_profiles, resolve_model_dir

            for profile in list_model_profiles(models_dir):
                model_id = str(profile.get("model_id", "")).strip()
                if not model_id:
                    continue
                model_dir, _ = resolve_model_dir(model_root=models_dir, model_id=model_id)
                if any(Path(model_dir).glob("DNN_*.pt")) or any(Path(model_dir).glob("DNN_*.h5")):
                    return True
        except Exception:
            pass
        return (
            any(models_dir.glob("DNN_*.pt"))
            or any(models_dir.glob("DNN_*.h5"))
            or any(models_dir.glob("*/DNN_*.pt"))
            or any(models_dir.glob("*/DNN_*.h5"))
        )

    def _version_tuple(self, value):
        parts = []
        for token in re.findall(r"\d+", str(value or "")):
            try:
                parts.append(int(token))
            except Exception:
                parts.append(0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    def _core_package_ready(self):
        importlib.invalidate_caches()
        try:
            import motionscore
        except Exception as exc:
            self._last_core_import_error = str(exc)
            return False
        installed = getattr(motionscore, "__version__", "0")
        ready = self._version_tuple(installed) >= self._version_tuple(MIN_CORE_VERSION)
        if not ready:
            self._last_core_import_error = f"installed version {installed}, need >= {MIN_CORE_VERSION}"
        else:
            self._last_core_import_error = ""
        return ready

    def _core_package_ready_in_subprocess(self):
        python_exe = self._python_executable_for_setup()
        if not python_exe:
            self._last_core_import_error = "Could not find Python executable for package check."
            return False
        code = (
            "import motionscore, sys; "
            "print(getattr(motionscore, '__version__', 'unknown')); "
            "sys.exit(0)"
        )
        completed = subprocess.run(
            [python_exe, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.stdout:
            self._log(completed.stdout)
        if int(completed.returncode) != 0:
            self._last_core_import_error = (completed.stdout or "").strip() or f"exit {completed.returncode}"
            return False
        version = (completed.stdout or "").strip().splitlines()[-1] if completed.stdout else "0"
        ready = self._version_tuple(version) >= self._version_tuple(MIN_CORE_VERSION)
        if not ready:
            self._last_core_import_error = f"subprocess version {version}, need >= {MIN_CORE_VERSION}"
        else:
            self._last_core_import_error = ""
        return ready

    def _installed_core_version(self):
        importlib.invalidate_caches()
        try:
            import motionscore

            return str(getattr(motionscore, "__version__", "unknown"))
        except Exception:
            return "not installed"

    def _python_executable_for_setup(self):
        return (
            shutil.which("PythonSlicer")
            or (sys.executable if Path(sys.executable).exists() else None)
            or shutil.which("python3")
            or shutil.which("python")
        )

    def _pip_install(self, *packages, upgrade=False, extra_args=None):
        python_exe = self._python_executable_for_setup()
        if not python_exe:
            raise RuntimeError("Could not find Python executable for pip install.")
        cmd = [python_exe, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        if extra_args:
            cmd.extend(str(a) for a in extra_args if str(a).strip())
        cmd.extend(str(p) for p in packages if str(p).strip())
        if len(cmd) <= 4:
            return
        self._log(f"[setup] running: {' '.join(cmd)}\n")
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.stdout:
            self._log(completed.stdout)
        if int(completed.returncode) != 0:
            raise RuntimeError(f"pip install failed (exit {completed.returncode}).")

    def _core_pip_requirements(self):
        return [f"{CORE_PYPI_PACKAGE}>={MIN_CORE_VERSION}", *CORE_PIP_CONSTRAINTS]

    def _ensure_core_package(self):
        if self._core_package_ready():
            self._log(f"[setup] core package already installed: {CORE_PYPI_PACKAGE}\n")
            return True
        try:
            self._set_license_status("Installing MotionScore core package from PyPI...")
            self._pip_install(*self._core_pip_requirements())
            if not (self._core_package_ready() or self._core_package_ready_in_subprocess()):
                detail = getattr(self, "_last_core_import_error", "")
                raise RuntimeError(f"Installed package but 'motionscore' import still failed. {detail}".strip())
            self._log(f"[setup] core package install ok: {CORE_PYPI_PACKAGE}\n")
            return True
        except Exception as exc:
            self._set_license_status(f"Core package install failed: {exc}")
            slicer.util.errorDisplay(
                "Could not install MotionScore core package from PyPI.\n\n"
                f"Package: {CORE_PYPI_PACKAGE}\n"
                f"Error: {exc}"
            )
            return False

    def _force_reinstall_core_package(self):
        try:
            self._set_license_status("Reinstalling MotionScore core package from PyPI...")
            self._pip_install(
                *self._core_pip_requirements(),
                upgrade=True,
                extra_args=["--force-reinstall", "--no-cache-dir"],
            )
            if not (self._core_package_ready() or self._core_package_ready_in_subprocess()):
                detail = getattr(self, "_last_core_import_error", "")
                raise RuntimeError(f"Package reinstall completed but import check failed. {detail}".strip())
            self._set_license_status("Core package reinstall complete.")
            self._log(f"[setup] core package force-reinstalled: {CORE_PYPI_PACKAGE}\n")
            return True
        except Exception as exc:
            self._set_license_status(f"Core package reinstall failed: {exc}")
            slicer.util.errorDisplay(
                "Force reinstall failed.\n\n"
                f"Package: {CORE_PYPI_PACKAGE}\n"
                f"Error: {exc}"
            )
            return False
        finally:
            self._update_setup_status()

    def onForceReinstallPackage(self):
        return self._force_reinstall_core_package()

    def _update_setup_status(self):
        if not hasattr(self, "setupStatusLabel"):
            return
        install_txt = "ready" if self._core_package_ready() else f"needs {MIN_CORE_VERSION}"
        models_txt = "ready" if self._has_local_models() else "required"
        self.setupStatusLabel.setText(f"Core: {install_txt} | Models: {models_txt}")

    def _training_mode_enabled(self):
        checked_attr = self.trainingModeCheck.checked
        return bool(checked_attr() if callable(checked_attr) else checked_attr)

    def _set_buttons_enabled(self, enabled):
        self.loadDatasetButton.enabled = enabled
        self.runButton.enabled = enabled
        self.manualRunButton.enabled = enabled
        self.refreshButton.enabled = enabled
        self.exportButton.enabled = enabled
        self.importButton.enabled = enabled
        self.loadScanButton.enabled = enabled
        self.applyButton.enabled = enabled
        self.backButton.enabled = bool(enabled and self._grade_history)
        self.clearButton.enabled = enabled
        self.quickSetupButton.enabled = enabled
        self.trainingModeCheck.enabled = enabled
        self.runModeCombo.enabled = enabled
        self.modelProfileCombo.enabled = enabled
        self.retrainBaseModelCombo.enabled = enabled
        self.sliceStepSpin.enabled = enabled
        self.runScopeCombo.enabled = enabled
        self.reviewScopeCombo.enabled = enabled
        self.clearReviewerCombo.enabled = enabled
        self.autoLoadCheck.enabled = enabled
        self.prepareRetrainButton.enabled = enabled
        self.trainHeadButton.enabled = enabled
        self.trainFullButton.enabled = enabled
        self.continueTrainButton.enabled = enabled
        self.retrainModelIdEdit.enabled = enabled
        self.retrainDisplayNameEdit.enabled = enabled
        self.retrainIncludeAutoCheck.enabled = enabled
        self.retrainSliceCountSpin.enabled = enabled
        self.retrainPatienceSpin.enabled = enabled
        self.retrainSeedSpin.enabled = enabled
        self.retrainAugHFlipCheck.enabled = enabled
        self.retrainAugVFlipCheck.enabled = enabled
        self.retrainAugRotateCheck.enabled = enabled
        self.retrainAugCropCheck.enabled = enabled
        self.retrainEpochsHeadSpin.enabled = enabled
        self.retrainEpochsFineSpin.enabled = enabled
        for btn in self.quickGradeButtons.values():
            btn.enabled = enabled
        self.interruptButton.enabled = not enabled
        self.trainInterruptButton.enabled = not enabled

    def _run_cli(self, args, on_finish=None):
        try:
            self._set_buttons_enabled(False)
            self._active_task_name = args[0] if args else None
            self.logic.run_cli(
                args=args,
                on_output=self._on_process_output,
                on_finished=lambda code, status, interrupted: self._on_process_finished(code, status, on_finish, interrupted),
            )
        except Exception as exc:
            self._set_buttons_enabled(True)
            self._set_progress_idle()
            slicer.util.errorDisplay(str(exc))

    def _on_process_finished(self, code, _status, callback, interrupted=False):
        self._set_buttons_enabled(True)
        self._log(f"[process] finished with exit code {code}\n")
        task_name = self._active_task_name
        self._active_task_name = None
        if interrupted:
            self._log("[process] interrupted by user\n")
            self.progressLabel.setText("Interrupted")
            return
        if code != 0:
            self.progressLabel.setText("Failed")
            slicer.util.errorDisplay(f"motionscore command failed with exit code {code}")
            return
        if task_name == "predict":
            if self._predict_total > 0:
                done = min(self._predict_done, self._predict_total)
                self.progressBar.minimum = 0
                self.progressBar.maximum = self._predict_total
                self.progressBar.value = done
                self.progressLabel.setText(f"Completed {done}/{self._predict_total}")
            else:
                self.progressLabel.setText(f"Completed {self._predict_done} scan(s)")
            try:
                self.runBox.collapsed = True
            except Exception:
                pass
        elif task_name == "train":
            self._set_progress_idle()
            self._update_training_plot(final=True)
            self._update_training_summary()
        elif task_name == "model-register":
            self._set_progress_idle()
            self._refresh_model_profiles()
        else:
            self._set_progress_idle()
        if callback is not None:
            callback()

    def onInterrupt(self):
        if self.logic.interrupt():
            self._log("[process] interrupt requested\n")
            self.progressLabel.setText("Interrupt requested...")

    def onQuickSetup(self):
        """
        Friendly setup flow:
        1) Force reinstall/upgrade core package from PyPI
        2) Download and register the configured model catalog entry only if no local models exist
        """
        self._persist_license_settings()

        if not self._force_reinstall_core_package():
            return

        if not self._has_local_models():
            if not self.onDownloadModelBundle():
                return
            return

        self._set_license_status("Setup complete (package reinstalled and local models ready).")
        self._log("[setup] complete: package force-reinstalled and local models ready\n")
        self._refresh_model_profiles()

    def onDownloadModelBundle(self):
        if not self._ensure_core_package():
            return False
        models_dir = self._models_dir()
        if models_dir is None:
            slicer.util.errorDisplay("Could not resolve local models folder.")
            return False
        if self._has_local_models(models_dir):
            self.modelsPathEdit.currentPath = str(models_dir)
            self._refresh_model_profiles()
            self._update_setup_status()
            self._set_license_status(f"Local MotionScore models already found in {models_dir}. Download skipped.")
            self._log(f"[models] local models found in {models_dir}; download skipped\n")
            return True
        self._set_license_status("Downloading and registering MotionScore models...")

        def _finish_download():
            self.modelsPathEdit.currentPath = str(models_dir)
            self._refresh_model_profiles()
            self._update_setup_status()
            self._set_license_status(f"Models downloaded and registered in {models_dir}.")

        self._run_cli(
            [
                "model-download",
                "--model-id",
                "base-v1",
                "--model-root",
                str(models_dir),
                "--catalog",
                DEFAULT_MODEL_CATALOG_URL,
                "--overwrite",
            ],
            on_finish=_finish_download,
        )
        return True

    def onLoadDataset(self):
        self.refreshReview()
        if self._combo_count(self.scanCombo) > 0:
            self.scanCombo.setCurrentIndex(0)
            if self._auto_load_enabled():
                self.onLoadSelectedScan()

    def onRunManualOnly(self):
        self.onRunPredict(manual_only=True)

    def onRunPredict(self, manual_only=False):
        dataset = self.datasetPathEdit.currentPath.strip()
        if not dataset:
            slicer.util.errorDisplay("Please choose Dataset Root")
            return
        if not self._core_package_ready():
            self._log(f"[setup] core package missing; installing from PyPI ({CORE_PYPI_PACKAGE}).\n")
            if not self._ensure_core_package():
                return
        models_dir = self._models_dir()
        if not manual_only:
            if models_dir is None or not models_dir.exists():
                slicer.util.errorDisplay("Could not resolve local models folder.")
                return
            has_models = self._has_local_models(models_dir)
            if not has_models:
                self._log("[setup] no local model files found; running local model setup.\n")
                self.onQuickSetup()
                self._set_license_status("Model setup started. Run prediction again after the download finishes.")
                return

        selected_scope = self._combo_text(self.runScopeCombo)
        if selected_scope == self.RUN_SCOPE_ALL and self._all_scans_already_predicted(self._selected_model_id()):
            self._log("[predict] all discovered scans already predicted; refreshing review only.\n")
            self.refreshReview()
            self.progressBar.minimum = 0
            self.progressBar.maximum = 100
            self.progressBar.value = 100
            self.progressLabel.setText("Up to date (review refreshed)")
            return

        args = [
            "predict",
            dataset,
            "--confidence-threshold",
            str(int(self.confidenceSpin.value)),
            "--output-root",
            dataset,
        ]
        if manual_only:
            args.append("--manual-only")
        else:
            args.extend(
                [
                    "--model-root",
                    str(models_dir),
                    "--model-id",
                    self._selected_model_id(),
                    "--slice-step",
                    str(int(self.sliceStepSpin.value)),
                ]
            )
            selected_device = (self._combo_text(self.deviceCombo) or "auto").lower()
            if selected_device != "auto":
                args.extend(["--device", selected_device])
        if self._training_mode_enabled():
            args.append("--training-mode")
        if selected_scope and selected_scope != self.RUN_SCOPE_ALL:
            args.extend(["--scan-id", selected_scope])

        self._start_predict_progress(selected_scope)
        self._run_cli(args, on_finish=self.refreshReview)

    def onRefreshReview(self):
        derivatives = self._derivatives_root()
        if derivatives is None:
            slicer.util.errorDisplay("Please choose Dataset Root")
            return
        index_path = derivatives / "index.tsv"
        if not index_path.exists():
            # Nothing predicted yet; just refresh view state.
            self.refreshReview()
            return

        args = [
            "review-init",
            str(derivatives),
            "--confidence-threshold",
            str(int(self.confidenceSpin.value)),
        ]
        if self._training_mode_enabled():
            args.append("--training-mode")
        self._run_cli(args, on_finish=self.refreshReview)

    def refreshReview(self):
        derivatives = self._derivatives_root()
        if derivatives is None:
            slicer.util.errorDisplay("Please choose Dataset Root")
            return

        index_path = derivatives / "index.tsv"
        if not index_path.exists():
            self._all_scan_ids = []
            self._audit_rows = []
            self.scanCombo.clear()
            self.autoLabel.text = "Auto grade: - | confidence: -"
            self.agreementLabel.text = "Agreement: overlap=0 | match=- | exact=-"
            self.trainingRevealLabel.text = "Last training reveal: -"
            self.clearReviewerCombo.clear()
            self.clearReviewerCombo.addItem(self.CLEAR_ALL_OPERATORS)
            self.agreementMatrixTable.setRowCount(0)
            self.agreementMatrixTable.setColumnCount(0)
            self._clear_profile_plot()
            self._set_run_scope_items(self._discover_scan_ids_for_dataset())
            self._update_review_queue_label()
            self._log(f"[review] index not found: {index_path}\n")
            return

        self._index_rows = {}
        self._review_rows = {}
        self._audit_rows = []
        self._model_index_rows = {}
        self._model_review_rows = {}
        self._model_audit_rows = {}

        with index_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                scan_id = row.get("scan_id", "")
                if not scan_id:
                    continue
                self._index_rows[scan_id] = row
                discovered = self._discover_model_entries_for_scan(derivatives, row)
                self._model_index_rows[scan_id] = {
                    model_id: dict(payload.get("index", {}))
                    for model_id, payload in discovered.items()
                    if payload.get("index")
                }
                self._model_review_rows[scan_id] = {
                    model_id: dict(payload.get("review", {}))
                    for model_id, payload in discovered.items()
                    if payload.get("review")
                }
                self._model_audit_rows[scan_id] = {
                    model_id: list(payload.get("audits", []))
                    for model_id, payload in discovered.items()
                    if payload.get("audits")
                }

                active_model_id = str(row.get("model_id", "")).strip() or "base-v1"
                active_review_row = self._model_review_rows.get(scan_id, {}).get(active_model_id)
                if active_review_row:
                    self._review_rows[scan_id] = active_review_row
                active_audits = self._model_audit_rows.get(scan_id, {}).get(active_model_id, [])
                if active_audits:
                    self._audit_rows.extend(active_audits)

        any_training = any(self._is_training_mode_row(row) for row in self._review_rows.values())
        self.trainingModeCheck.blockSignals(True)
        self.trainingModeCheck.setChecked(bool(any_training))
        self.trainingModeCheck.blockSignals(False)
        if not any_training:
            self.trainingRevealLabel.text = "Last training reveal: -"

        self._all_scan_ids = sorted(set(self._index_rows.keys()).union(set(self._review_rows.keys())))
        pending = self._pending_scan_ids()
        self._set_run_scope_items(pending)
        self._refresh_clear_reviewer_items()
        self._rebuild_scan_combo()
        self._update_review_queue_label()

        self._log(f"[review] loaded {len(self._review_rows)} scan review row(s), pending={len(pending)}\n")
        self._update_agreement_summary()
        self._update_agreement_matrix()
        self.onScanSelectionChanged(self._combo_text(self.scanCombo))

    def onTrainingModeToggled(self, checked):
        mode = "ON" if bool(checked) else "OFF"
        self._log(f"[review] training mode toggled: {mode}\n")
        derivatives = self._derivatives_root()
        if derivatives is None:
            self.onScanSelectionChanged(self._combo_text(self.scanCombo))
            return

        index_path = derivatives / "index.tsv"
        if not index_path.exists():
            # Nothing initialized yet: only refresh visible text state.
            self.onScanSelectionChanged(self._combo_text(self.scanCombo))
            return

        # Apply the mode switch to existing review rows immediately so pending scans
        # are properly blinded/unblinded without requiring a new predict run.
        args = [
            "review-init",
            str(derivatives),
            "--confidence-threshold",
            str(int(self.confidenceSpin.value)),
        ]
        if bool(checked):
            args.append("--training-mode")
        self._run_cli(args, on_finish=self.refreshReview)

    def onReviewScopeChanged(self, _scope_text):
        self._rebuild_scan_combo(preferred_scan_id=self._combo_text(self.scanCombo))
        self._update_review_queue_label()

    def onConfidenceThresholdChanged(self, *_args):
        if self.logic.is_running():
            return
        self._rebuild_scan_combo(preferred_scan_id=self._combo_text(self.scanCombo))
        self._update_review_queue_label()

    def _update_review_queue_label(self):
        shown_count = len(self._scan_ids_for_scope())
        pending_count = len(self._pending_scan_ids())
        reviewed_count = 0
        for row in self._review_rows.values():
            manual = str(row.get("manual_grade", "")).strip()
            if manual:
                reviewed_count += 1
        total_count = len(self._all_scan_ids)
        self.reviewQueueLabel.text = (
            f"Queue: shown={shown_count} | pending={pending_count} | reviewed={reviewed_count}/{total_count}"
        )

    def _pending_scan_ids(self):
        pending = []
        for scan_id, row in self._review_rows.items():
            status = str(row.get("review_status", "")).strip()
            is_training = self._is_training_mode_row(row)
            manual_set = bool(str(row.get("manual_grade", "")).strip())
            if status == "pending":
                pending.append(scan_id)
                continue
            if status == "training_pending":
                pending.append(scan_id)
                continue
            if is_training and not manual_set:
                pending.append(scan_id)
        return sorted(set(pending))

    def _low_confidence_scan_ids(self):
        threshold = int(self.confidenceSpin.value)
        out = []
        for scan_id in self._all_scan_ids:
            row = self._review_rows.get(scan_id) or self._index_rows.get(scan_id) or {}
            raw_conf = str(row.get("automatic_confidence", "")).strip()
            try:
                conf = float(raw_conf)
            except Exception:
                # If confidence is unavailable/corrupt, keep it visible for manual review.
                out.append(scan_id)
                continue
            if conf < threshold:
                out.append(scan_id)
        return out

    def _scan_ids_for_scope(self):
        scope = self._combo_text(self.reviewScopeCombo)
        if scope == self.REVIEW_SCOPE_ALL:
            return list(self._all_scan_ids)
        if scope == self.REVIEW_SCOPE_LOW_CONFIDENCE:
            return self._low_confidence_scan_ids()
        return self._pending_scan_ids()

    def _rebuild_scan_combo(self, preferred_scan_id=None):
        scan_ids = self._scan_ids_for_scope()
        current = str(preferred_scan_id or "").strip() or self._combo_text(self.scanCombo)
        self.scanCombo.blockSignals(True)
        self.scanCombo.clear()
        self.scanCombo.addItems(scan_ids)
        if current and current in scan_ids:
            self.scanCombo.setCurrentIndex(scan_ids.index(current))
        elif scan_ids:
            self.scanCombo.setCurrentIndex(0)
        self.scanCombo.blockSignals(False)
        self.onScanSelectionChanged(self._combo_text(self.scanCombo))
        self._update_review_queue_label()

    def _refresh_clear_reviewer_items(self):
        previous = self._combo_text(self.clearReviewerCombo)
        reviewers = sorted(
            {
                str(row.get("reviewer", "")).strip()
                for row in self._review_rows.values()
                if str(row.get("reviewer", "")).strip()
            }
        )
        items = [self.CLEAR_ALL_OPERATORS] + reviewers
        self.clearReviewerCombo.blockSignals(True)
        self.clearReviewerCombo.clear()
        self.clearReviewerCombo.addItems(items)
        if previous in items:
            self.clearReviewerCombo.setCurrentIndex(items.index(previous))
        elif previous and previous != self.CLEAR_ALL_OPERATORS:
            self.clearReviewerCombo.setEditText(previous)
        else:
            self.clearReviewerCombo.setCurrentIndex(0)
        self.clearReviewerCombo.blockSignals(False)

    def onDatasetPathChanged(self, *_args):
        if self.logic.is_running():
            return
        self._grade_history = []
        self.backButton.enabled = False
        self._refresh_model_profiles()
        if self._review_rows:
            self._set_run_scope_items(self._pending_scan_ids())
        else:
            self._set_run_scope_items(self._discover_scan_ids_for_dataset())
        self._update_review_queue_label()

    def onScanSelectionChanged(self, scan_id):
        if not scan_id:
            self.autoLabel.text = "Auto grade: - | confidence: -"
            self._set_selected_manual_grade(None)
            self._refresh_profile_model_combo("")
            self._clear_profile_plot()
            return

        row = self._review_rows.get(scan_id)
        if row is None:
            self.autoLabel.text = "Auto grade: - | confidence: -"
            self._set_selected_manual_grade(None)
            self._refresh_profile_model_combo(scan_id)
            self._clear_profile_plot()
            return

        self._refresh_profile_model_combo(scan_id)

        auto_grade = row.get("automatic_grade", "-")
        auto_conf = row.get("automatic_confidence", "-")
        manual_mode = self._is_manual_mode_row(row)
        training_pending = self._is_training_mode_row(row) and not str(row.get("manual_grade", "")).strip()
        blind_active = bool(self.trainingModeCheck.checked) or training_pending
        if manual_mode:
            self.autoLabel.text = "Manual-only mode: no AI suggestion."
            self._set_selected_manual_grade(None)
            self._update_profile_plot(scan_id)
        elif blind_active:
            self.autoLabel.text = "Training mode: prediction hidden until manual grade is submitted."
            self._set_selected_manual_grade(None)
            self._show_profile_whiteout()
        else:
            self.autoLabel.text = f"Auto grade: {auto_grade} | confidence: {auto_conf}%"
            self._update_profile_plot(scan_id)
        if not blind_active and not manual_mode:
            try:
                self._set_selected_manual_grade(int(float(auto_grade)), suggested=True)
            except Exception:
                self._set_selected_manual_grade(None)

        if (
            scan_id
            and self._auto_load_enabled()
            and scan_id != self._loaded_scan_id
            and not self.logic.is_running()
            and not self._auto_loading_scan
        ):
            qt.QTimer.singleShot(0, self._auto_load_current_scan)

    def _auto_load_current_scan(self):
        if self._auto_loading_scan or not self._auto_load_enabled():
            return
        scan_id = self._combo_text(self.scanCombo)
        if not scan_id or scan_id == self._loaded_scan_id:
            return
        self._auto_loading_scan = True
        try:
            self.onLoadSelectedScan()
        finally:
            self._auto_loading_scan = False

    def onQuickSelectGrade(self, grade):
        try:
            grade_value = int(grade)
        except Exception:
            return
        if grade_value < 1 or grade_value > 5:
            return
        if not self._combo_text(self.scanCombo):
            return
        self._set_selected_manual_grade(grade_value)
        self.onApplyManual()

    def _focus_scan_for_edit(self, scan_id):
        if not scan_id:
            return False
        idx = self.scanCombo.findText(scan_id)
        self.scanCombo.blockSignals(True)
        if idx < 0:
            self.scanCombo.insertItem(0, scan_id)
            idx = 0
        self.scanCombo.setCurrentIndex(idx)
        self.scanCombo.blockSignals(False)
        self.onScanSelectionChanged(scan_id)
        self.onLoadSelectedScan()
        return True

    def onBackToPreviousScan(self):
        if not self._grade_history:
            return
        scan_id = self._grade_history.pop()
        self.backButton.enabled = bool(self._grade_history)
        if scan_id not in self._index_rows:
            self.refreshReview()
        if scan_id not in self._index_rows:
            slicer.util.errorDisplay(f"Cannot go back: scan '{scan_id}' is no longer available in this dataset.")
            return
        self._focus_scan_for_edit(scan_id)

    def onApplyManual(self):
        scan_id = self._combo_text(self.scanCombo)
        if not scan_id:
            slicer.util.errorDisplay("No pending scan selected")
            return

        reviewer = self.reviewerEdit.text.strip()
        if not reviewer:
            slicer.util.errorDisplay("Please enter Reviewer")
            return
        self._persist_reviewer_setting()

        manual_grade = int(self._selected_manual_grade or 0)
        if manual_grade < 1 or manual_grade > 5:
            slicer.util.errorDisplay("Please select a quick grade (1 to 5) before saving")
            return

        derivatives = self._derivatives_root()
        if derivatives is None:
            slicer.util.errorDisplay("Cannot resolve results root")
            return

        args = [
            "review-apply",
            str(derivatives),
            "--scan-id",
            scan_id,
            "--manual-grade",
            str(manual_grade),
            "--reviewer",
            reviewer,
        ]
        self._run_cli(args, on_finish=lambda sid=scan_id: self._on_manual_applied(sid))

    def _on_manual_applied(self, scan_id):
        if scan_id:
            self._grade_history.append(scan_id)
        self.backButton.enabled = bool(self._grade_history)
        self._refresh_and_load_next(previous_scan_id=scan_id)

    def onProfileModelChanged(self, *_args):
        scan_id = self._combo_text(self.scanCombo)
        if not scan_id:
            return
        row = self._review_rows.get(scan_id, {})
        training_pending = self._is_training_mode_row(row) and not str(row.get("manual_grade", "")).strip()
        blind_active = bool(self.trainingModeCheck.checked) or training_pending
        if blind_active:
            self._show_profile_whiteout()
            return
        self._update_profile_plot(scan_id)

    def onClearGrades(self):
        derivatives = self._derivatives_root()
        if derivatives is None:
            slicer.util.errorDisplay("Cannot resolve results root")
            return

        selected_reviewer = self._combo_text(self.clearReviewerCombo)
        args = [
            "review-clear",
            str(derivatives),
        ]
        if selected_reviewer == self.CLEAR_ALL_OPERATORS:
            args.append("--all-reviewers")
            msg = "Clear grades for ALL reviewers?"
        elif selected_reviewer:
            args.extend(["--reviewer", selected_reviewer])
            msg = f"Clear all grades for reviewer '{selected_reviewer}'?"
        else:
            slicer.util.errorDisplay("Select a reviewer to clear, or choose 'All operators'.")
            return

        if not slicer.util.confirmYesNoDisplay(msg, windowTitle="Clear Grades"):
            return
        self._run_cli(args, on_finish=self.refreshReview)

    def onExport(self):
        derivatives = self._derivatives_root()
        if derivatives is None:
            slicer.util.errorDisplay("Cannot resolve results root")
            return

        args = [
            "export",
            str(derivatives),
        ]
        self._run_cli(args)

    def onImportFinalGrades(self):
        derivatives = self._derivatives_root()
        if derivatives is None:
            slicer.util.errorDisplay("Cannot resolve results root")
            return
        file_path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Import Final Grades",
            str(derivatives),
            "Tables (*.tsv *.csv);;All Files (*)",
        )
        if isinstance(file_path, tuple):
            file_path = file_path[0]
        if not file_path:
            return
        args = [
            "import-final-grades",
            str(derivatives),
            "--input",
            str(file_path),
            "--reviewer",
            self.reviewerEdit.text.strip() or "import",
        ]
        self._run_cli(args, on_finish=self.refreshReview)

    def _prepare_retrain_then(self, callback):
        manifest_path = self._retrain_manifest_path()
        derivatives = self._derivatives_root()
        if derivatives is None or manifest_path is None:
            slicer.util.errorDisplay("Please choose Dataset Root")
            return
        args = [
            "train-prepare",
            str(derivatives),
            "--output",
            str(manifest_path),
            "--min-auto-confidence",
            str(float(int(self.confidenceSpin.value)) / 100.0),
            "--slice-count",
            str(int(self.retrainSliceCountSpin.value)),
            "--seed",
            str(int(self.retrainSeedSpin.value)),
            "--cv-folds",
            str(int(self._retrain_cv_folds())),
        ]
        auto_attr = self.retrainIncludeAutoCheck.checked
        if bool(auto_attr() if callable(auto_attr) else auto_attr):
            args.append("--include-auto-without-manual")
        self._run_cli(args, on_finish=callback)

    def onPrepareRetrainManifest(self):
        self._prepare_retrain_then(None)

    def _run_train(self, *, classifier_only=False, continue_training=False):
        manifest_path = self._retrain_manifest_path()
        output_model_dir = self._retrain_output_model_dir()
        models_root = self._models_dir()
        if manifest_path is None or output_model_dir is None or models_root is None:
            slicer.util.errorDisplay("Please choose Dataset Root")
            return
        output_model_dir.mkdir(parents=True, exist_ok=True)
        self._training_output_model_dir = output_model_dir
        args = [
            "train",
            "--manifest",
            str(manifest_path),
            "--output-model-dir",
            str(output_model_dir),
            "--device",
            (self._combo_text(self.deviceCombo) or "auto").lower(),
            "--epochs-head",
            str(0 if continue_training else int(self.retrainEpochsHeadSpin.value)),
            "--epochs-finetune",
            str(0 if classifier_only else int(self.retrainEpochsFineSpin.value)),
            "--early-stopping-patience",
            str(int(self.retrainPatienceSpin.value)),
            "--seed",
            str(int(self.retrainSeedSpin.value)),
        ]
        if continue_training and any(output_model_dir.glob("DNN_*.pt")):
            args.extend(["--init-model-dir", str(output_model_dir)])
            args.extend(["--lr-finetune", str(float(self._retrain_continue_lr(output_model_dir)))])
        else:
            args.extend(["--model-root", str(models_root), "--init-model-id", self._selected_retrain_base_model_id()])
        aug_h_attr = self.retrainAugHFlipCheck.checked
        aug_v_attr = self.retrainAugVFlipCheck.checked
        aug_rotate_attr = self.retrainAugRotateCheck.checked
        aug_crop_attr = self.retrainAugCropCheck.checked
        if not bool(aug_h_attr() if callable(aug_h_attr) else aug_h_attr):
            args.append("--no-aug-hflip")
        if not bool(aug_v_attr() if callable(aug_v_attr) else aug_v_attr):
            args.append("--no-aug-vflip")
        if bool(aug_rotate_attr() if callable(aug_rotate_attr) else aug_rotate_attr):
            args.append("--aug-rotate")
        if bool(aug_crop_attr() if callable(aug_crop_attr) else aug_crop_attr):
            args.append("--aug-crop")
        self.trainingMetricsLabel.text = "Holdout: training in progress..."
        self.trainingPlotLabel.setText("Training plot: waiting for updates...")
        self._run_cli(args, on_finish=self._register_trained_model)

    def _register_trained_model(self):
        output_model_dir = self._training_output_model_dir
        models_root = self._models_dir()
        if output_model_dir is None or models_root is None:
            return
        model_id = self._normalized_retrain_model_id()
        display_name = self.retrainDisplayNameEdit.text.strip() or model_id
        args = [
            "model-register",
            "--model-root",
            str(models_root),
            "--model-id",
            model_id,
            "--model-dir",
            str(output_model_dir),
            "--display-name",
            display_name,
            "--source-model-id",
            self._selected_retrain_base_model_id(),
            "--training-manifest",
            str(self._retrain_manifest_path() or ""),
            "--metrics-path",
            str(output_model_dir / "training_metrics.json"),
            "--make-default",
        ]
        self._run_cli(args, on_finish=self._after_model_register)

    def _after_model_register(self):
        self._refresh_model_profiles()
        self._update_training_summary()
        self._update_training_plot(final=True)

    def onTrainClassifierOnly(self):
        self._prepare_retrain_then(lambda: self._run_train(classifier_only=True, continue_training=False))

    def onTrainFullModel(self):
        self._prepare_retrain_then(lambda: self._run_train(classifier_only=False, continue_training=False))

    def onContinueTraining(self):
        self._prepare_retrain_then(lambda: self._run_train(classifier_only=False, continue_training=True))

    def onLoadSelectedScan(self):
        scan_id = self._combo_text(self.scanCombo)
        if not scan_id:
            slicer.util.errorDisplay("No pending scan selected")
            return

        index_row = self._index_rows.get(scan_id)
        if index_row is None:
            slicer.util.errorDisplay(f"Missing index row for {scan_id}")
            return

        raw_path = index_row.get("raw_image_path", "").strip()
        if not raw_path:
            slicer.util.errorDisplay(f"Missing raw image path for {scan_id}")
            return

        row = self._review_rows.get(scan_id, {})
        training_pending = self._is_training_mode_row(row) and not str(row.get("manual_grade", "")).strip()
        blind_active = bool(self.trainingModeCheck.checked) or training_pending
        if blind_active:
            self._show_profile_whiteout()
        else:
            # Always try to update the profile plot, even if volume load later fails.
            self._update_profile_plot(scan_id)

        # Avoid keeping many large scans in memory: unload prior scan before loading next.
        self._remove_loaded_scan_volume()

        primary_error = None
        loaded_node = None
        try:
            loaded_node = slicer.util.loadVolume(raw_path)
        except Exception as exc:
            primary_error = exc
            self._log(f"[load] native volume load failed for {raw_path}: {exc}\n")

        if loaded_node is None or loaded_node is False:
            try:
                loaded_node = self._load_aim_with_core(raw_path)
                self._log(f"[load] AIM fallback load succeeded: {raw_path}\n")
            except Exception as fallback_exc:
                if primary_error is not None:
                    slicer.util.errorDisplay(
                        "Could not load selected scan.\n"
                        f"Path: {raw_path}\n"
                        f"Native Slicer load error: {primary_error}\n"
                        f"AIM fallback load error: {fallback_exc}"
                    )
                else:
                    slicer.util.errorDisplay(
                        "Could not load selected scan.\n"
                        f"Path: {raw_path}\n"
                        f"AIM fallback load error: {fallback_exc}"
                    )
                return

        self._log(f"[load] loaded volume: {raw_path}\n")
        self._loaded_scan_id = scan_id
        self._loaded_volume_node = loaded_node
        self._render_profile_plot(scan_id)

    def _remove_loaded_scan_volume(self):
        node = self._loaded_volume_node
        self._loaded_volume_node = None
        self._loaded_scan_id = ""
        if node is None:
            return
        try:
            if node.GetScene() is not None:
                slicer.mrmlScene.RemoveNode(node)
                self._log(f"[load] removed previous volume: {node.GetName()}\n")
        except Exception as exc:
            self._log(f"[load] failed to remove previous volume: {exc}\n")

    def _refresh_and_load_next(self, previous_scan_id=None):
        self.refreshReview()
        self._show_training_reveal(previous_scan_id)
        count_attr = self.scanCombo.count
        count = int(count_attr() if callable(count_attr) else count_attr)
        if count <= 0:
            self._clear_profile_plot()
            return

        idx = -1
        if previous_scan_id:
            idx = self.scanCombo.findText(previous_scan_id)
            if idx >= 0 and count > 1:
                idx = min(idx + 1, count - 1)
        if idx < 0:
            idx = 0
        self.scanCombo.setCurrentIndex(idx)
        if not self._auto_load_enabled():
            self.onLoadSelectedScan()

    def _set_run_scope_items(self, scan_ids):
        previous = self._combo_text(self.runScopeCombo)
        items = [self.RUN_SCOPE_ALL] + list(scan_ids)
        self.runScopeCombo.blockSignals(True)
        self.runScopeCombo.clear()
        self.runScopeCombo.addItems(items)
        if previous in items:
            self.runScopeCombo.setCurrentIndex(items.index(previous))
        else:
            self.runScopeCombo.setCurrentIndex(0)
        self.runScopeCombo.blockSignals(False)

    def _discover_scan_ids_for_dataset(self):
        dataset = self.datasetPathEdit.currentPath.strip()
        if not dataset:
            return []
        try:
            from motionscore.config import AppConfig
            from motionscore.dataset.discovery import discover_raw_sessions
            from motionscore.utils import make_scan_id
        except Exception:
            return []

        try:
            sessions = discover_raw_sessions(root=Path(dataset), cfg=AppConfig().discovery)
            scan_ids = [
                make_scan_id(s.subject_id, s.site, s.session_id, s.raw_image_path)
                for s in sessions
            ]
            return sorted(set(scan_ids))
        except Exception as exc:
            self._log(f"[run-scope] could not discover scans for dropdown: {exc}\n")
            return []

    def _all_scans_already_predicted(self, model_id=None):
        derivatives = self._derivatives_root()
        if derivatives is None:
            return False
        index_path = derivatives / "index.tsv"
        if not index_path.exists():
            return False

        discovered_scan_ids = set(self._discover_scan_ids_for_dataset())
        if not discovered_scan_ids:
            return False

        indexed_scan_ids = set()
        selected_model_id = str(model_id or self._selected_model_id() or "").strip()
        try:
            with index_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    scan_id = str(row.get("scan_id", "")).strip()
                    row_model_id = str(row.get("model_id", "")).strip()
                    if scan_id and row_model_id == selected_model_id:
                        indexed_scan_ids.add(scan_id)
        except Exception as exc:
            self._log(f"[predict] could not read existing index for skip-check: {exc}\n")
            return False

        return discovered_scan_ids.issubset(indexed_scan_ids)

    def _on_process_output(self, text):
        self._log(text)
        if self._active_task_name == "predict":
            self._update_predict_progress_from_output(text)
        elif self._active_task_name == "train":
            self._update_training_plot(final=False)

    def _combo_count(self, combo):
        count_attr = combo.count
        return int(count_attr() if callable(count_attr) else count_attr)

    def _combo_text(self, combo):
        text_attr = combo.currentText
        text = text_attr() if callable(text_attr) else text_attr
        return str(text).strip()

    def _start_predict_progress(self, selected_scope):
        if selected_scope and selected_scope != self.RUN_SCOPE_ALL:
            total = 1
        else:
            total = max(0, self._combo_count(self.runScopeCombo) - 1)

        self._predict_total = int(total)
        self._predict_done = 0
        if self._predict_total > 0:
            self.progressBar.minimum = 0
            self.progressBar.maximum = self._predict_total
            self.progressBar.value = 0
            self.progressLabel.setText(f"Processing scan 0/{self._predict_total}")
        else:
            # Unknown total count: show indeterminate progress.
            self.progressBar.minimum = 0
            self.progressBar.maximum = 0
            self.progressBar.value = 0
            self.progressLabel.setText("Processing scans...")

    def _update_predict_progress_from_output(self, text):
        completed = len(re.findall(r"(?m)^\[predict\]\s+", text))
        if completed <= 0:
            return
        self._predict_done += completed

        if self._predict_total > 0:
            current = min(self._predict_done, self._predict_total)
            self.progressBar.minimum = 0
            self.progressBar.maximum = self._predict_total
            self.progressBar.value = current
            self.progressLabel.setText(f"Processing scan {current}/{self._predict_total}")
        else:
            self.progressBar.minimum = 0
            self.progressBar.maximum = 0
            self.progressBar.value = 0
            self.progressLabel.setText(f"Processing scans... done {self._predict_done}")

    def _set_progress_idle(self):
        self._predict_total = 0
        self._predict_done = 0
        self.progressBar.minimum = 0
        self.progressBar.maximum = 100
        self.progressBar.value = 0
        self.progressLabel.setText("Idle")

    def _load_aim_with_core(self, raw_path):
        from motionscore.io.aim import read_aim

        aim = read_aim(Path(raw_path), scaling="native")
        volume_xyz = aim.data
        volume_kji = volume_xyz.transpose(2, 1, 0)

        node_name = Path(raw_path).stem
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", node_name)
        slicer.util.updateVolumeFromArray(node, volume_kji)

        try:
            sx, sy, sz = aim.spacing
            node.SetSpacing(float(sx), float(sy), float(sz))
        except Exception:
            pass
        try:
            ox, oy, oz = aim.origin
            node.SetOrigin(float(ox), float(oy), float(oz))
        except Exception:
            pass

        slicer.util.setSliceViewerLayers(background=node)
        slicer.util.resetSliceViews()
        return node

    def _clear_profile_plot(self):
        self._profile_source_pixmap = None
        self._profile_scan_id = ""
        self.profileLabel.clear()
        self.profileLabel.setText("Slice profile plot: -")

    def _show_profile_whiteout(self):
        self._profile_source_pixmap = None
        self._profile_scan_id = ""
        self.profileLabel.clear()
        self.profileLabel.setText("Slice profile hidden until manual grade is submitted.")

    def _is_training_mode_row(self, row):
        text = str(row.get("training_mode", "")).strip().lower()
        return text in {"1", "true", "yes", "y", "on"}

    def _is_manual_mode_row(self, row):
        text = str(row.get("manual_mode", "")).strip().lower()
        return text in {"1", "true", "yes", "y", "on"}

    def _update_agreement_summary(self):
        try:
            from motionscore.review.store import compute_review_agreement
        except Exception as exc:
            self.agreementLabel.text = "Agreement: unavailable (core import failed)"
            self._log(f"[agreement] failed to import compute_review_agreement: {exc}\n")
            return

        stats = compute_review_agreement(list(self._review_rows.values()))
        n_scored = int(stats.get("n_scored", 0))
        if n_scored <= 0:
            self.agreementLabel.text = "Agreement: overlap=0 | match=- | exact=-"
            return

        exact_matches = 0
        for row in self._review_rows.values():
            auto_txt = str(row.get("automatic_grade", "")).strip()
            manual_txt = str(row.get("manual_grade", "")).strip()
            if not auto_txt or not manual_txt:
                continue
            try:
                if int(float(auto_txt)) == int(float(manual_txt)):
                    exact_matches += 1
            except Exception:
                continue
        exact = float(stats.get("agreement_exact", 0.0)) * 100.0
        self.agreementLabel.text = f"Agreement: overlap={n_scored} | match={exact_matches}/{n_scored} | exact={exact:.1f}%"

    def _update_agreement_matrix(self):
        try:
            from motionscore.review.store import compute_grade_pair_agreement
        except Exception as exc:
            self._log(f"[agreement] failed to import compute_grade_pair_agreement: {exc}\n")
            self.agreementMatrixTable.setRowCount(0)
            self.agreementMatrixTable.setColumnCount(0)
            return

        def _canon_reviewer(text):
            value = str(text or "").strip()
            if not value:
                return ""
            value = re.sub(r"\s+", " ", value)
            return value.casefold()

        ai_grade_maps = {}
        for scan_id, model_rows in self._model_review_rows.items():
            for model_id, row in model_rows.items():
                try:
                    grade = int(float(str(row.get("automatic_grade", "")).strip()))
                except Exception:
                    continue
                if 1 <= grade <= 5:
                    ai_grade_maps.setdefault(str(model_id).strip(), {})[scan_id] = grade

        # Build operator grades from current review rows first (most reliable snapshot),
        # then overlay audit history to preserve clear/apply ordering when available.
        op_scan_grade = {}
        reviewer_label_by_canon = {}
        for scan_id, model_rows in self._model_review_rows.items():
            for row in model_rows.values():
                reviewer_raw = str(row.get("reviewer", "")).strip()
                reviewer = _canon_reviewer(reviewer_raw)
                if not reviewer:
                    continue
                reviewer_label_by_canon.setdefault(reviewer, reviewer_raw)
                try:
                    grade = int(float(str(row.get("manual_grade", "")).strip()))
                except Exception:
                    continue
                if 1 <= grade <= 5:
                    op_scan_grade[(reviewer, scan_id)] = grade

        audit_rows = []
        for scan_audits in self._model_audit_rows.values():
            for model_audits in scan_audits.values():
                audit_rows.extend(model_audits)
        audit_rows = sorted(audit_rows, key=lambda row: str(row.get("timestamp", "")))
        for row in audit_rows:
            scan_id = str(row.get("scan_id", "")).strip()
            reviewer_raw = str(row.get("reviewer", "")).strip()
            reviewer = _canon_reviewer(reviewer_raw)
            event = str(row.get("event", "")).strip()
            if not scan_id or not reviewer:
                continue
            reviewer_label_by_canon.setdefault(reviewer, reviewer_raw)
            if event == "clear_manual":
                op_scan_grade.pop((reviewer, scan_id), None)
                continue
            try:
                grade = int(float(str(row.get("manual_grade", "")).strip()))
            except Exception:
                continue
            if 1 <= grade <= 5:
                op_scan_grade[(reviewer, scan_id)] = grade

        operators = sorted({op for op, _sid in op_scan_grade.keys()}, key=lambda op: reviewer_label_by_canon.get(op, op))
        op_counts = {}
        for op in operators:
            op_counts[op] = len({sid for (op_key, sid), _g in op_scan_grade.items() if op_key == op})
        ai_models = sorted(ai_grade_maps.keys(), key=lambda model_id: self._model_display_label(model_id).casefold())
        participant_ids = [f"ai::{model_id}" for model_id in ai_models] + operators
        participants = [
            f"AI {self._model_display_label(model_id)} (predicted={len(ai_grade_maps.get(model_id, {}))})"
            for model_id in ai_models
        ] + [f"{reviewer_label_by_canon.get(op, op)} (graded={op_counts.get(op, 0)})" for op in operators]
        n = len(participant_ids)
        self.agreementMatrixTable.setRowCount(n)
        self.agreementMatrixTable.setColumnCount(n)
        self.agreementMatrixTable.setVerticalHeaderLabels(participants)
        self.agreementMatrixTable.setHorizontalHeaderLabels(participants)

        for i, p_i in enumerate(participant_ids):
            for j, p_j in enumerate(participant_ids):
                if i == j:
                    item = qt.QTableWidgetItem("—")
                    item.setTextAlignment(qt.Qt.AlignCenter)
                    self.agreementMatrixTable.setItem(i, j, item)
                    continue
                if j > i:
                    # show lower-triangular matrix only (avoid duplicate mirrored entries)
                    item = qt.QTableWidgetItem("")
                    item.setTextAlignment(qt.Qt.AlignCenter)
                    self.agreementMatrixTable.setItem(i, j, item)
                    continue

                pairs = []
                if p_i.startswith("ai::"):
                    model_i = p_i.split("::", 1)[1]
                    scan_to_grade_i = ai_grade_maps.get(model_i, {})
                    if p_j.startswith("ai::"):
                        model_j = p_j.split("::", 1)[1]
                        scan_to_grade_j = ai_grade_maps.get(model_j, {})
                        common = set(scan_to_grade_i.keys()).intersection(set(scan_to_grade_j.keys()))
                        for sid in sorted(common):
                            pairs.append((scan_to_grade_i[sid], scan_to_grade_j[sid]))
                    else:
                        scan_to_grade_j = {
                            sid: g for (op, sid), g in op_scan_grade.items() if op == p_j
                        }
                        for sid, g_i in scan_to_grade_i.items():
                            g_j = scan_to_grade_j.get(sid)
                            if g_j is not None:
                                pairs.append((g_i, g_j))
                elif p_j.startswith("ai::"):
                    model_j = p_j.split("::", 1)[1]
                    scan_to_grade_j = ai_grade_maps.get(model_j, {})
                    scan_to_grade_i = {
                        sid: g for (op, sid), g in op_scan_grade.items() if op == p_i
                    }
                    for sid, g_i in scan_to_grade_i.items():
                        g_j = scan_to_grade_j.get(sid)
                        if g_j is not None:
                            pairs.append((g_i, g_j))
                elif p_i == "AI":
                    scan_to_grade_j = {
                        sid: g for (op, sid), g in op_scan_grade.items() if op == p_j
                    }
                else:
                    scan_to_grade_i = {
                        sid: g for (op, sid), g in op_scan_grade.items() if op == p_i
                    }
                    scan_to_grade_j = {
                        sid: g for (op, sid), g in op_scan_grade.items() if op == p_j
                    }
                    common = set(scan_to_grade_i.keys()).intersection(set(scan_to_grade_j.keys()))
                    for sid in sorted(common):
                        pairs.append((scan_to_grade_i[sid], scan_to_grade_j[sid]))

                stats = compute_grade_pair_agreement(pairs)
                n_scored = int(stats.get("n_scored", 0))
                if n_scored <= 0:
                    txt = "overlap=0"
                else:
                    exact_matches = sum(1 for g_i, g_j in pairs if g_i == g_j)
                    exact = float(stats.get("agreement_exact", 0.0)) * 100.0
                    txt = f"overlap={n_scored}\nmatch={exact_matches}/{n_scored}\nexact={exact:.1f}%"
                item = qt.QTableWidgetItem(txt)
                item.setTextAlignment(qt.Qt.AlignCenter)
                self.agreementMatrixTable.setItem(i, j, item)

        self.agreementMatrixTable.resizeColumnsToContents()
        self.agreementMatrixTable.resizeRowsToContents()

    def _show_training_reveal(self, scan_id):
        if not scan_id:
            return
        row = self._review_rows.get(scan_id)
        if row is None:
            return
        if not self._is_training_mode_row(row):
            return
        manual_grade = str(row.get("manual_grade", "")).strip()
        if not manual_grade:
            return
        auto_grade = str(row.get("automatic_grade", "")).strip() or "-"
        match = "match" if auto_grade == manual_grade else "mismatch"
        text = f"{scan_id}: operator={manual_grade}, AI={auto_grade} ({match})"
        self.trainingRevealLabel.text = f"Last training reveal: {text}"
        self._log(f"[training] reveal {text}\n")

    def _install_slice_observer(self):
        try:
            layout_manager = slicer.app.layoutManager()
            if layout_manager is None:
                return
            red_widget = layout_manager.sliceWidget("Red")
            if red_widget is None:
                return
            slice_node = red_widget.mrmlSliceNode()
            if slice_node is None:
                return
            if self._slice_observer_node is slice_node and self._slice_observer_tag is not None:
                return
            if self._slice_observer_node is not None and self._slice_observer_tag is not None:
                self._slice_observer_node.RemoveObserver(self._slice_observer_tag)
            self._slice_observer_node = slice_node
            self._slice_observer_tag = slice_node.AddObserver(
                vtk.vtkCommand.ModifiedEvent,
                self._on_slice_position_changed,
            )
        except Exception as exc:
            self._log(f"[profile] could not install slice observer: {exc}\n")

    def _on_slice_position_changed(self, _caller=None, _event=None):
        if not self._loaded_scan_id:
            return
        if self._profile_scan_id != self._loaded_scan_id:
            return
        self._render_profile_plot(self._loaded_scan_id)

    def _slice_step_mm(self):
        node = self._loaded_volume_node
        if node is None:
            return 1.0
        try:
            spacing = node.GetSpacing()
            if spacing is not None and len(spacing) >= 3:
                step = abs(float(spacing[2]))
                if step > 0.0:
                    return step
        except Exception:
            pass
        return 1.0

    def _scroll_loaded_slice_by(self, steps):
        if not self._loaded_scan_id or self._loaded_volume_node is None:
            return False
        try:
            steps_i = int(steps)
        except Exception:
            return False
        if steps_i == 0:
            return False
        self._install_slice_observer()
        if self._slice_observer_node is None:
            return False
        slice_idx, n_slices, ijk_h, slice_to_ras = self._current_slice_geometry()
        if slice_idx is None or n_slices is None or ijk_h is None or slice_to_ras is None:
            return False

        target_idx = max(0, min(n_slices - 1, slice_idx + steps_i))
        if target_idx == slice_idx:
            return False

        ijk_to_ras = vtk.vtkMatrix4x4()
        self._loaded_volume_node.GetIJKToRASMatrix(ijk_to_ras)
        target_ijk_h = [float(ijk_h[0]), float(ijk_h[1]), float(target_idx), 1.0]
        target_ras_h = [0.0, 0.0, 0.0, 0.0]
        ijk_to_ras.MultiplyPoint(target_ijk_h, target_ras_h)

        normal = [
            float(slice_to_ras.GetElement(0, 2)),
            float(slice_to_ras.GetElement(1, 2)),
            float(slice_to_ras.GetElement(2, 2)),
        ]
        normal_norm = math.sqrt(sum(component * component for component in normal))
        if normal_norm <= 0.0:
            return False
        target_offset = sum(
            target_ras_h[axis] * (normal[axis] / normal_norm)
            for axis in range(3)
        )
        try:
            self._slice_observer_node.SetSliceOffset(target_offset)
        except Exception:
            return False

        self._render_profile_plot(self._loaded_scan_id)
        return True

    def _wheel_steps_from_event(self, event):
        delta_y = 0
        try:
            angle = event.angleDelta()
            y_attr = angle.y
            delta_y = int(y_attr() if callable(y_attr) else y_attr)
        except Exception:
            delta_y = 0
        if delta_y == 0:
            try:
                pixel = event.pixelDelta()
                y_attr = pixel.y
                delta_y = int(y_attr() if callable(y_attr) else y_attr)
            except Exception:
                delta_y = 0
        if delta_y == 0:
            try:
                delta_y = int(event.delta())
            except Exception:
                delta_y = 0
        if delta_y == 0:
            return 0
        steps = int(delta_y / 120)
        if steps == 0:
            return 1 if delta_y > 0 else -1
        return steps

    def _is_profile_widget_or_child(self, widget):
        current = widget
        while current is not None:
            if current is self.profileLabel:
                return True
            parent_attr = getattr(current, "parentWidget", None)
            if parent_attr is None:
                break
            try:
                current = parent_attr() if callable(parent_attr) else None
            except Exception:
                break
        return False

    def _cursor_over_profile_area(self):
        try:
            app = qt.QApplication.instance()
            if app is None:
                return False
            hover = app.widgetAt(qt.QCursor.pos())
            return self._is_profile_widget_or_child(hover)
        except Exception:
            return False

    def _handle_profile_wheel_event(self, obj, event):
        try:
            if event is not None and event.type() == qt.QEvent.Wheel and self._cursor_over_profile_area():
                steps = self._wheel_steps_from_event(event)
                if steps != 0 and self._scroll_loaded_slice_by(steps):
                    try:
                        event.accept()
                    except Exception:
                        pass
                    return True
        except Exception as exc:
            self._log(f"[profile] wheel event handling failed: {exc}\n")
        return False

    def _current_slice_geometry(self):
        node = self._loaded_volume_node
        if node is None:
            return None, None, None, None
        image_data = node.GetImageData() if hasattr(node, "GetImageData") else None
        if image_data is None:
            return None, None, None, None
        dims = image_data.GetDimensions()
        if len(dims) < 3:
            return None, None, None, None
        n_slices = int(dims[2])
        if n_slices <= 0:
            return None, None, None, None

        self._install_slice_observer()
        if self._slice_observer_node is None:
            return None, n_slices, None, None

        # Slicer API compatibility: some versions fill an output matrix argument,
        # newer versions return the matrix directly with no arguments.
        slice_to_ras = vtk.vtkMatrix4x4()
        try:
            self._slice_observer_node.GetSliceToRAS(slice_to_ras)
        except TypeError:
            returned = self._slice_observer_node.GetSliceToRAS()
            if returned is not None:
                slice_to_ras.DeepCopy(returned)
        ras_h = [
            float(slice_to_ras.GetElement(0, 3)),
            float(slice_to_ras.GetElement(1, 3)),
            float(slice_to_ras.GetElement(2, 3)),
            1.0,
        ]

        ras_to_ijk = vtk.vtkMatrix4x4()
        node.GetRASToIJKMatrix(ras_to_ijk)
        ijk_h = [0.0, 0.0, 0.0, 0.0]
        ras_to_ijk.MultiplyPoint(ras_h, ijk_h)
        slice_idx = int(round(float(ijk_h[2])))
        slice_idx = max(0, min(n_slices - 1, slice_idx))
        return slice_idx, n_slices, ijk_h, slice_to_ras

    def _current_slice_cursor(self):
        slice_idx, n_slices, _ijk_h, _slice_to_ras = self._current_slice_geometry()
        return slice_idx, n_slices

    def _grade_for_scan(self, scan_id):
        if self._selected_manual_grade in {1, 2, 3, 4, 5}:
            return int(self._selected_manual_grade)
        row = self._review_rows.get(scan_id, {})
        for key in ("manual_grade", "automatic_grade"):
            try:
                value = int(float(str(row.get(key, "")).strip()))
            except Exception:
                value = 0
            if 1 <= value <= 5:
                return value
        return 3

    def _render_profile_plot(self, scan_id):
        if self._profile_source_pixmap is None:
            self._clear_profile_plot()
            return
        if not scan_id or scan_id != self._profile_scan_id:
            return

        width_attr = self.profileLabel.width
        label_width = int(width_attr() if callable(width_attr) else width_attr)
        width = max(180, min(434, label_width - 8))
        scaled = self._profile_source_pixmap.scaledToWidth(width, qt.Qt.SmoothTransformation)
        if scaled.isNull():
            self._clear_profile_plot()
            return

        out_pix = qt.QPixmap(scaled)
        if scan_id == self._loaded_scan_id:
            slice_idx, n_slices = self._current_slice_cursor()
            if slice_idx is not None and n_slices is not None and n_slices > 1:
                frac = float(slice_idx + 0.5) / float(n_slices)
                x_min = int(round(float(max(0, out_pix.width() - 1)) * PROFILE_PLOT_LEFT_FRACTION))
                x_max = int(round(float(max(0, out_pix.width() - 1)) * PROFILE_PLOT_RIGHT_FRACTION))
                if x_max <= x_min:
                    x_min = 0
                    x_max = max(0, out_pix.width() - 1)
                x = int(round(float(x_min) + frac * float(x_max - x_min)))
                x = max(0, min(max(0, out_pix.width() - 1), x))
                painter = qt.QPainter(out_pix)
                pen = qt.QPen(self._grade_plot_color(self._grade_for_scan(scan_id)))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(x, 0, x, max(0, out_pix.height() - 1))
                painter.end()

        self.profileLabel.setPixmap(out_pix)

    def _update_profile_plot(self, scan_id):
        derivatives = self._derivatives_root()
        if derivatives is None:
            self._clear_profile_plot()
            return

        selected_model_id = self._selected_profile_model_id()
        row = self._model_index_rows.get(scan_id, {}).get(selected_model_id) if selected_model_id else None
        if not row:
            row = self._index_rows.get(scan_id, {})
        rel_path = (
            row.get("slice_profile_png_path", "").strip()
            or row.get("preview_png_path", "").strip()
        )
        if not rel_path:
            self._log(f"[profile] no png path in index for {scan_id}\n")
            self._clear_profile_plot()
            return

        png_path = (derivatives / rel_path).resolve()
        if not png_path.exists():
            self._log(f"[profile] png not found on disk: {png_path}\n")
            self._clear_profile_plot()
            return

        try:
            pix = qt.QPixmap(str(png_path))
        except Exception as exc:
            self._log(f"[profile] failed to create QPixmap for {png_path}: {exc}\n")
            self._clear_profile_plot()
            return
        if pix.isNull():
            self._log(f"[profile] QPixmap is null for {png_path}\n")
            self._clear_profile_plot()
            return

        self._profile_source_pixmap = pix
        self._profile_scan_id = scan_id
        self._render_profile_plot(scan_id)
