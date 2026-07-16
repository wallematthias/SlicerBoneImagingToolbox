import importlib
import re
import shutil
import sys
from pathlib import Path

import ctk
import qt
import slicer
import vtk

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


MODULE_VERSION = "0.1.0"
CORE_REQUIREMENT = "spine-segment>=0.1.0"
TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from SlicerBoneImagingToolboxLib.slicer_update_ui import run_toolbox_update_dialog


OUTPUT_SPECS = (
    ("vertebral_level", "Vertebral levels", "vertebral-level"),
    ("process_body", "Process/body", "process-body"),
    ("cort_trab", "Cortical/trabecular", "cort-trab"),
)


def _safe_name(text):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip()).strip("._")
    return safe or "ct"


def _input_stem(path):
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def _spine_segment_output_paths(input_path, output_dir):
    stem = _input_stem(input_path)
    root = Path(output_dir)
    return {
        "vertebral_level": root / f"{stem}_vertebral-level.nii.gz",
        "process_body": root / f"{stem}_process-body.nii.gz",
        "cort_trab": root / f"{stem}_cort-trab.nii.gz",
        "centroids": root / f"{stem}_centroids.json",
    }


class SpineSegmentationCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Spine Segmentation"
        parent.categories = ["Bone Imaging.CT"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Run spine-segment vertebral CT segmentation and load vertebral-level, "
            "process/body, and cortical/trabecular outputs into Slicer. "
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = """Part of the Bone Imaging Toolbox for 3D Slicer.

Spine Segmentation is backed by the spine-segment Python package.

For vertebral localization, identification, and level segmentation, cite:
Payer C, Stern D, Bischof H, Urschler M. Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net. VISIGRAPP 2020, Volume 5: VISAPP. 2020;124-133. doi:10.5220/0008975201240133

For the process/body compartment workflow, cite:
Walle M, Matheson BE, Boyd SK. Comparing linear and nonlinear finite element models of vertebral strength across the thoracolumbar spine: a benchmark from density-calibrated computed tomography. GigaScience. 2025;14:giaf094. doi:10.1093/gigascience/giaf094"""


class SpineSegmentationCTLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._proc = None
        self._user_terminated = False

    def is_core_available(self):
        try:
            importlib.import_module("spine_segment")
            return True
        except Exception:
            return False

    def install_or_update_core(self):
        slicer.util.pip_install("spine-segment>=0.1.0")
        for name in list(sys.modules):
            if name == "spine_segment" or name.startswith("spine_segment."):
                sys.modules.pop(name, None)

    def is_running(self):
        return self._proc is not None

    def save_input_volume(self, volume_node, output_dir):
        if volume_node is None:
            raise ValueError("Select an input CT volume.")
        output_root = Path(output_dir).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        input_path = output_root / f"{_safe_name(volume_node.GetName())}.nii.gz"
        ok = slicer.util.saveNode(volume_node, str(input_path))
        if not ok:
            raise RuntimeError(f"Could not save CT volume to {input_path}")
        return input_path

    def run_cli(self, input_path, output_dir, *, device="auto", mode="full", on_output=None, on_finished=None):
        if self._proc is not None:
            raise RuntimeError("A spine segmentation process is already running.")
        self._user_terminated = False

        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.MergedChannels)

        env = qt.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        for key in ("ITK_AUTOLOAD_PATH", "SITK_AUTOLOAD_PATH"):
            if env.contains(key):
                env.remove(key)
            env.insert(key, "")
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
                on_output(text)

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
            raise RuntimeError("Could not find a Python executable for spine-segment.")

        args = [
            "-m",
            "spine_segment.cli",
            str(input_path),
            "--output",
            str(Path(output_dir).expanduser()),
            "--device",
            str(device or "auto"),
            "--overwrite",
        ]
        if mode == "localization":
            args.append("--localization-only")
        elif mode == "level":
            args.append("--level-only")
        if on_output:
            on_output(f"[process] launching: {python_exe} {' '.join(args)}\n")

        proc.start(python_exe, args)
        if not proc.waitForStarted(3000):
            raise RuntimeError("Failed to start spine-segment.")
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

    def load_segmentation_output(self, path, *, name, output_kind, reference_volume=None):
        output_path = Path(path)
        if not output_path.exists():
            raise FileNotFoundError(f"Expected spine-segment output was not created: {output_path}")
        node = slicer.util.loadSegmentation(str(output_path), {"name": name})
        if not node:
            raise RuntimeError(f"Could not load segmentation output: {output_path}")
        node.SetAttribute("BoneImaging.SpineSegment.OutputKind", str(output_kind))
        node.SetAttribute("BoneImaging.SpineSegment.SourcePath", str(output_path))
        if reference_volume is not None:
            node.SetAttribute("BoneImaging.SpineSegment.SourceVolumeID", reference_volume.GetID())
            try:
                node.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume)
            except Exception:
                pass
        try:
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            if display is not None:
                display.SetOpacity(0.5)
                display.SetVisibility2DFill(True)
                display.SetVisibility2DOutline(True)
                display.SetAllSegmentsVisibility2DFill(True)
                display.SetAllSegmentsOpacity2DFill(0.65)
        except Exception:
            pass
        return node

    def load_centroid_markers(self, centroids_path, *, name, reference_volume=None):
        import json

        path = Path(centroids_path)
        if not path.exists():
            raise FileNotFoundError(f"Expected spine-segment centroid output was not created: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        centroids = payload.get("centroids", {})
        if not isinstance(centroids, dict):
            raise RuntimeError(f"Invalid centroid JSON: {path}")

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.SetAttribute("BoneImaging.SpineSegment.OutputKind", "centroids")
        node.SetAttribute("BoneImaging.SpineSegment.SourcePath", str(path))
        node.CreateDefaultDisplayNodes()

        def _label_sort_key(item):
            text = str(item[0])
            return (0, int(text)) if text.isdigit() else (1, text)

        for raw_label, entry in sorted(centroids.items(), key=_label_sort_key):
            if not isinstance(entry, dict):
                continue
            ras = self._centroid_entry_to_ras(entry, reference_volume)
            if ras is None:
                continue
            label = f"V{entry.get('label', raw_label)}"
            self._add_fiducial(node, ras, label)

        try:
            display = node.GetDisplayNode()
            if display is not None:
                display.SetSelectedColor(1.0, 0.75, 0.05)
                display.SetGlyphScale(2.0)
                display.SetTextScale(2.0)
        except Exception:
            pass
        return node

    def _centroid_entry_to_ras(self, entry, reference_volume):
        voxel_xyz = entry.get("voxel_xyz")
        if reference_volume is not None and isinstance(voxel_xyz, (list, tuple)) and len(voxel_xyz) == 3:
            matrix = vtk.vtkMatrix4x4()
            reference_volume.GetIJKToRASMatrix(matrix)
            ras_h = matrix.MultiplyPoint(
                (float(voxel_xyz[0]), float(voxel_xyz[1]), float(voxel_xyz[2]), 1.0)
            )
            return (float(ras_h[0]), float(ras_h[1]), float(ras_h[2]))

        physical_xyz = entry.get("physical_xyz")
        if isinstance(physical_xyz, (list, tuple)) and len(physical_xyz) == 3:
            return (-float(physical_xyz[0]), -float(physical_xyz[1]), float(physical_xyz[2]))
        return None

    def _add_fiducial(self, node, ras, label):
        point = vtk.vtkVector3d(float(ras[0]), float(ras[1]), float(ras[2]))
        if hasattr(node, "AddControlPointWorld"):
            node.AddControlPointWorld(point, str(label))
        elif hasattr(node, "AddControlPoint"):
            node.AddControlPoint(point, str(label))
        else:
            node.AddFiducial(float(ras[0]), float(ras[1]), float(ras[2]), str(label))


class SpineSegmentationCTWidget(ScriptedLoadableModuleWidget):
    def _tip(self, widget, text):
        widget.toolTip = str(text)
        return widget

    def setup(self):
        super().setup()
        self.logic = SpineSegmentationCTLogic()
        self._currentInputPath = None
        self._currentOutputDir = None
        self._currentMode = "full"

        self._build_main_section()
        self.layout.addStretch(1)
        self._refresh_status()

    def cleanup(self):
        if self.logic.is_running():
            self.logic.interrupt()

    def _build_main_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Spine segmentation"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)

        self.statusLabel = qt.QLabel("Checking dependency...")
        self.installButton = qt.QPushButton("Install / Update Spine Segmentation")
        self.updateToolboxButton = qt.QPushButton("Check toolbox updates")
        self._tip(self.statusLabel, "Shows whether the spine-segment package is available in Slicer Python.")
        self._tip(self.installButton, "Install or update the spine-segment Python dependency in Slicer Python.")
        self._tip(self.updateToolboxButton, "Check whether this local Slicer toolbox checkout has upstream updates.")
        self.installButton.clicked.connect(self._install_core)
        self.updateToolboxButton.clicked.connect(self._check_toolbox_updates)
        install_row_widget = qt.QWidget()
        install_row = qt.QHBoxLayout(install_row_widget)
        install_row.setContentsMargins(0, 0, 0, 0)
        install_row.addWidget(self.installButton)
        install_row.addWidget(self.updateToolboxButton)
        form.addRow("Status", self.statusLabel)
        form.addRow(install_row_widget)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = False
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self._tip(self.volumeSelector, "Input clinical CT volume to segment.")
        form.addRow("Input CT", self.volumeSelector)

        self.deviceCombo = qt.QComboBox()
        for label, value in [
            ("Auto", "auto"),
            ("CUDA", "cuda"),
            ("CPU", "cpu"),
        ]:
            self.deviceCombo.addItem(label, value)
        self._tip(
            self.deviceCombo,
            "PyTorch device used by spine-segment. Auto chooses CUDA, then MPS only if Conv3D is supported, then CPU.",
        )
        form.addRow("Device", self.deviceCombo)

        self.modeCombo = qt.QComboBox()
        for label, value in [
            ("Full: levels + body/process + cort/trab", "full"),
            ("Vertebral levels only", "level"),
            ("Localization only: centroid markers", "localization"),
        ]:
            self.modeCombo.addItem(label, value)
        self._tip(self.modeCombo, "Choose whether to load centroid markers only, vertebral-level labels, or the full compartment output set.")
        form.addRow("Run mode", self.modeCombo)

        self.outputDirEdit = qt.QLineEdit(str(self._default_output_dir()))
        browse = qt.QPushButton("Browse...")
        browse.clicked.connect(self._browse_output_dir)
        self._tip(self.outputDirEdit, "Folder where the exported input NIfTI and spine-segment outputs are written.")
        self._tip(browse, "Select an output folder for spine-segment files.")
        output_row = qt.QHBoxLayout()
        output_row.addWidget(self.outputDirEdit)
        output_row.addWidget(browse)
        form.addRow("Output folder", output_row)

        button_row_widget = qt.QWidget()
        button_row = qt.QHBoxLayout(button_row_widget)
        button_row.setContentsMargins(0, 0, 0, 0)
        self.runButton = qt.QPushButton("Run Spine Segmentation")
        self.stopButton = qt.QPushButton("Stop")
        self.stopButton.enabled = False
        self.runButton.clicked.connect(self._run_segmentation)
        self.stopButton.clicked.connect(self._stop_segmentation)
        self._tip(self.runButton, "Export the selected CT, run spine-segment, and load the outputs for the selected run mode.")
        self._tip(self.stopButton, "Stop the active spine-segment process.")
        button_row.addWidget(self.runButton)
        button_row.addWidget(self.stopButton)
        form.addRow(button_row_widget)

        self.progressLabel = qt.QLabel("Idle")
        self.progressBar = qt.QProgressBar()
        self.progressBar.minimum = 0
        self.progressBar.maximum = 100
        self.progressBar.value = 0
        self._tip(self.progressLabel, "Current spine-segment command status.")
        self._tip(self.progressBar, "Progress for the active spine-segment command.")
        form.addRow("Progress", self.progressLabel)
        form.addRow(self.progressBar)

        self.consoleBox = ctk.ctkCollapsibleButton()
        self.consoleBox.text = "Console"
        self.consoleBox.collapsed = True
        self.layout.addWidget(self.consoleBox)
        console_layout = qt.QVBoxLayout(self.consoleBox)
        self.console = qt.QTextEdit()
        self.console.readOnly = True
        self.console.setMinimumHeight(140)
        self._tip(self.console, "spine-segment process output and loaded result paths.")
        console_layout.addWidget(self.console)

    def _default_output_dir(self):
        return Path.home() / "SlicerBoneImagingToolboxRuns" / "SpineSegmentationCT"

    def _browse_output_dir(self):
        path = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            "Select spine segmentation output folder",
            self.outputDirEdit.text,
        )
        if path:
            self.outputDirEdit.text = path

    def _refresh_status(self):
        if self.logic.is_core_available():
            self.statusLabel.text = "spine-segment installed"
        else:
            self.statusLabel.text = "spine-segment not installed"

    def _install_core(self):
        try:
            self._append_log("[setup] installing spine-segment...\n")
            self.logic.install_or_update_core()
            self._refresh_status()
            self._append_log("[setup] spine-segment dependency is installed.\n")
        except Exception as exc:
            self._error(exc)

    def _check_toolbox_updates(self):
        run_toolbox_update_dialog(__file__, log=self._append_log)

    def _run_segmentation(self):
        try:
            if self.logic.is_running():
                raise RuntimeError("A spine segmentation process is already running.")
            if not self.logic.is_core_available():
                raise RuntimeError("Install spine-segment before running this module.")
            volume_node = self.volumeSelector.currentNode()
            output_dir = Path(self.outputDirEdit.text).expanduser()
            input_path = self.logic.save_input_volume(volume_node, output_dir)
            self._currentInputPath = input_path
            self._currentOutputDir = output_dir
            self._currentMode = self.modeCombo.currentData
            self._set_running(True, f"Running spine-segment on {volume_node.GetName()}...")
            self.logic.run_cli(
                input_path,
                output_dir,
                device=self.deviceCombo.currentData,
                mode=self._currentMode,
                on_output=self._append_log,
                on_finished=self._on_process_finished,
            )
        except Exception as exc:
            self._set_running(False, "Failed")
            self._error(exc)

    def _stop_segmentation(self):
        if self.logic.interrupt():
            self.progressLabel.text = "Stop requested..."
            self._append_log("[process] stop requested\n")

    def _on_process_finished(self, exit_code, _exit_status, interrupted):
        self._set_running(False, "Interrupted" if interrupted else "Finished")
        if interrupted:
            self._append_log("[process] interrupted\n")
            return
        if int(exit_code) != 0:
            self.progressLabel.text = "Failed"
            slicer.util.errorDisplay(f"spine-segment failed with exit code {exit_code}. See the module console.")
            return
        try:
            nodes = self._load_outputs()
            self.progressLabel.text = f"Loaded {len(nodes)} output node(s)"
        except Exception as exc:
            self._error(exc)

    def _load_outputs(self):
        if self._currentInputPath is None or self._currentOutputDir is None:
            raise RuntimeError("No completed spine-segment run is available to load.")
        reference_volume = self.volumeSelector.currentNode()
        paths = _spine_segment_output_paths(self._currentInputPath, self._currentOutputDir)
        base = _input_stem(self._currentInputPath)
        nodes = []
        if self._currentMode == "localization":
            output_specs = ()
        elif self._currentMode == "level":
            output_specs = (OUTPUT_SPECS[0],)
        else:
            output_specs = OUTPUT_SPECS
        for key, title, _suffix in output_specs:
            node = self.logic.load_segmentation_output(
                paths[key],
                name=f"{base} {title}",
                output_kind=key,
                reference_volume=reference_volume,
            )
            nodes.append(node)
            self._append_log(f"[load] {title}: {paths[key]}\n")
        centroids = paths["centroids"]
        if centroids.exists():
            node = self.logic.load_centroid_markers(
                centroids,
                name=f"{base} Vertebral centroids",
                reference_volume=reference_volume,
            )
            nodes.append(node)
            self._append_log(f"[load] centroids: {centroids}\n")
        return nodes

    def _set_running(self, running, text):
        self.runButton.enabled = not bool(running)
        self.stopButton.enabled = bool(running)
        self.installButton.enabled = not bool(running)
        self.progressLabel.text = str(text)
        if running:
            self.progressBar.minimum = 0
            self.progressBar.maximum = 0
            self.progressBar.value = 0
        else:
            self.progressBar.minimum = 0
            self.progressBar.maximum = 100
            self.progressBar.value = 0 if str(text) in {"Idle", "Failed", "Interrupted"} else 100
        qt.QApplication.processEvents()

    def _append_log(self, text):
        self.console.insertPlainText(str(text))
        self.console.moveCursor(qt.QTextCursor.End)

    def _error(self, exc):
        self.progressLabel.text = f"Error: {exc}"
        self._append_log(f"[error] {exc}\n")
        slicer.util.errorDisplay(str(exc))


class SpineSegmentationCTTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("SpineSegmentationCT smoke test passed.")
