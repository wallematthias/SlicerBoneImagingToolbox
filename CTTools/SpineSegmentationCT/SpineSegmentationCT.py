import importlib
import json
import os
import platform
import re
import shutil
import subprocess
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
CONDA_RUNTIME_ENV = "spine-segment-pytorch"
TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from SlicerBoneImagingToolboxLib.slicer_update_ui import run_toolbox_update_dialog
from SlicerBoneImagingToolboxLib.spine_segmentation_batch import (
    build_spine_segmentation_batch_commands,
    discover_spine_segmentation_batch_cases,
    discovered_image_roles,
    write_spine_segmentation_manifest,
)
from SlicerBoneImagingToolboxLib.vertebra_labels import format_verse_label


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


def _default_conda_python_path():
    return Path.home() / "miniforge3" / "envs" / CONDA_RUNTIME_ENV / "bin" / "python"


def _candidate_conda_executables():
    seen = set()
    candidates = [
        Path.home() / "miniforge3" / "bin" / "conda",
        Path.home() / "miniforge3" / "condabin" / "conda",
        Path("/opt/homebrew/bin/conda"),
        Path("/usr/local/bin/conda"),
    ]
    found = shutil.which("conda")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            yield candidate


def _clean_python_subprocess_env():
    env = os.environ.copy()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "ITK_AUTOLOAD_PATH",
        "SITK_AUTOLOAD_PATH",
        "SimpleITK_AUTOLOAD_PATH",
    ):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


RUNTIME_PROBE_SCRIPT = r"""
import importlib.util
import json
import platform
import sys

payload = {
    "executable": sys.executable,
    "machine": platform.machine(),
    "python": platform.python_version(),
    "spine_segment_available": False,
    "torch_available": False,
    "mps_available": False,
    "mps_conv3d_supported": False,
}

try:
    import spine_segment
    payload["spine_segment_available"] = True
    payload["spine_segment_path"] = getattr(spine_segment, "__file__", "")
except Exception as exc:
    payload["spine_segment_error"] = repr(exc)

try:
    import torch
    payload["torch_available"] = True
    payload["torch_version"] = getattr(torch, "__version__", "")
    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )
    payload["mps_available"] = mps_available
    if mps_available:
        try:
            module = torch.nn.Conv3d(1, 1, 3, padding=1).to("mps")
            tensor = torch.zeros((1, 1, 8, 8, 8), device="mps")
            _ = module(tensor)
            torch.mps.synchronize()
            payload["mps_conv3d_supported"] = True
        except Exception as exc:
            payload["mps_conv3d_error"] = repr(exc)
except Exception as exc:
    payload["torch_error"] = repr(exc)

print(json.dumps(payload, sort_keys=True))
"""


class SpineSegmentationCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Spine Segmentation"
        parent.categories = ["Bone Imaging.Segmentation Methods"]
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

    def default_conda_python_path(self):
        return _default_conda_python_path()

    def install_or_update_conda_runtime(self, conda_python=None, on_output=None):
        python_path = Path(conda_python or self.default_conda_python_path()).expanduser()
        if not python_path.exists():
            default_path = self.default_conda_python_path()
            if python_path != default_path:
                raise RuntimeError(
                    f"Custom conda Python does not exist: {python_path}. "
                    "Create the environment first or use the default runtime path."
                )
            conda = next(_candidate_conda_executables(), None)
            if conda is None:
                raise RuntimeError(
                    "Could not find conda. Install Miniforge/conda or set the runtime Python to an existing environment."
                )
            self._run_setup_command(
                [str(conda), "create", "-y", "-n", CONDA_RUNTIME_ENV, "python=3.11"],
                on_output=on_output,
            )
        self._run_setup_command(
            [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
            on_output=on_output,
        )
        self._run_setup_command(
            [str(python_path), "-m", "pip", "install", "--upgrade", "torch", CORE_REQUIREMENT],
            on_output=on_output,
        )
        return self.probe_python_runtime(python_path)

    def _run_setup_command(self, args, *, on_output=None):
        if on_output:
            on_output(f"[setup] {' '.join(str(a) for a in args)}\n")
        completed = subprocess.run(
            [str(a) for a in args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_python_subprocess_env(),
            check=False,
        )
        if on_output and completed.stdout:
            on_output(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"Setup command failed with exit code {completed.returncode}: {' '.join(str(a) for a in args)}")

    def probe_python_runtime(self, python_executable):
        python_path = Path(str(python_executable or "")).expanduser()
        if not python_path.exists():
            return {
                "available": False,
                "executable": str(python_path),
                "error": "Python executable does not exist.",
            }
        try:
            completed = subprocess.run(
                [str(python_path), "-c", RUNTIME_PROBE_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=45,
                env=_clean_python_subprocess_env(),
                check=False,
            )
        except Exception as exc:
            return {"available": False, "executable": str(python_path), "error": repr(exc)}

        payload = None
        for line in reversed((completed.stdout or "").splitlines()):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
        if payload is None:
            payload = {}
        payload["available"] = completed.returncode == 0
        payload["executable"] = str(python_path)
        payload["probe_returncode"] = completed.returncode
        if completed.returncode != 0:
            payload["probe_output"] = completed.stdout
        return payload

    def runtime_summary(self, probe):
        if not probe or not probe.get("available"):
            return probe.get("error", "unavailable") if isinstance(probe, dict) else "unavailable"
        parts = [
            probe.get("machine") or platform.machine(),
            f"Python {probe.get('python', '?')}",
        ]
        if probe.get("spine_segment_available"):
            parts.append("spine-segment")
        else:
            parts.append("spine-segment missing")
        if probe.get("torch_available"):
            parts.append(f"torch {probe.get('torch_version', '?')}")
        else:
            parts.append("torch missing")
        if probe.get("mps_conv3d_supported"):
            parts.append("MPS Conv3D OK")
        elif probe.get("mps_available"):
            parts.append("MPS no Conv3D")
        return ", ".join(str(p) for p in parts if p)

    def is_running(self):
        return self._proc is not None

    def discover_spine_segmentation_batch_cases(self, dataset_root, **filters):
        return discover_spine_segmentation_batch_cases(dataset_root, **filters)

    def build_spine_segmentation_batch_commands(self, dataset_root, cases, **options):
        return build_spine_segmentation_batch_commands(dataset_root, cases, **options)

    def write_spine_segmentation_manifest(self, dataset_root, commands):
        return write_spine_segmentation_manifest(dataset_root, commands, module_version=MODULE_VERSION)

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

    def _python_slicer_executable(self):
        candidates = []
        found = shutil.which("PythonSlicer")
        if found:
            candidates.append(Path(found))
        try:
            app_path = Path(slicer.app.applicationFilePath())
            candidates.extend(
                [
                    app_path.parent / "PythonSlicer",
                    app_path.parent.parent / "bin" / "PythonSlicer",
                ]
            )
        except Exception:
            pass
        candidates.extend(
            [
                Path(sys.executable).with_name("PythonSlicer"),
                Path(sys.executable),
            ]
        )
        for candidate in candidates:
            try:
                if candidate.exists():
                    return str(candidate)
            except Exception:
                continue
        return None

    def select_runtime(self, runtime="auto", conda_python=None, on_output=None):
        mode = str(runtime or "auto").lower()
        conda_path = Path(conda_python or self.default_conda_python_path()).expanduser()

        if mode in {"conda", "auto"} and conda_path.exists():
            probe = self.probe_python_runtime(conda_path)
            if on_output:
                on_output(f"[runtime] Conda probe: {self.runtime_summary(probe)}\n")
            if probe.get("available") and probe.get("spine_segment_available"):
                if mode == "conda" or probe.get("mps_conv3d_supported"):
                    return str(conda_path), "Conda MPS runtime", True
            elif mode == "conda":
                raise RuntimeError(f"Conda runtime is not ready: {self.runtime_summary(probe)}")
        elif mode == "conda":
            raise RuntimeError(
                f"Conda runtime Python does not exist: {conda_path}. "
                "Click Install / Update Conda MPS Runtime or choose a valid Python path."
            )

        python_exe = self._python_slicer_executable()
        if python_exe is None:
            raise RuntimeError("Could not find PythonSlicer for the Slicer Python runtime.")
        if not self.is_core_available():
            raise RuntimeError(
                "spine-segment is not installed in Slicer Python, and no ready Conda MPS runtime was found."
            )
        if on_output:
            on_output("[runtime] Using Slicer Python runtime.\n")
        return python_exe, "Slicer Python runtime", False

    def run_cli(
        self,
        input_path,
        output_dir,
        *,
        device="auto",
        mode="full",
        runtime="auto",
        conda_python=None,
        on_output=None,
        on_finished=None,
    ):
        if self._proc is not None:
            raise RuntimeError("A spine segmentation process is already running.")
        self._user_terminated = False

        python_exe, runtime_label, clean_runtime_env = self.select_runtime(
            runtime=runtime,
            conda_python=conda_python,
            on_output=on_output,
        )

        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.MergedChannels)

        env = qt.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        keys_to_clear = ["ITK_AUTOLOAD_PATH", "SITK_AUTOLOAD_PATH", "SimpleITK_AUTOLOAD_PATH"]
        if clean_runtime_env:
            keys_to_clear.extend(["PYTHONHOME", "PYTHONPATH"])
        for key in keys_to_clear:
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
            on_output(f"[runtime] {runtime_label}: {python_exe}\n")
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
            raw_verse_label = entry.get("label", raw_label)
            label = format_verse_label(raw_verse_label)
            self._add_fiducial(node, ras, label, description=self._centroid_description(raw_verse_label, entry))

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

    def _centroid_description(self, raw_verse_label, entry):
        parts = [f"VerSe label {raw_verse_label}"]
        score = entry.get("score")
        if score is not None:
            try:
                parts.append(f"score={float(score):.3f}")
            except (TypeError, ValueError):
                pass
        return "; ".join(parts)

    def _add_fiducial(self, node, ras, label, description=None):
        point = vtk.vtkVector3d(float(ras[0]), float(ras[1]), float(ras[2]))
        if hasattr(node, "AddControlPointWorld"):
            index = node.AddControlPointWorld(point, str(label))
        elif hasattr(node, "AddControlPoint"):
            index = node.AddControlPoint(point, str(label))
        else:
            index = node.AddFiducial(float(ras[0]), float(ras[1]), float(ras[2]), str(label))
        if description and isinstance(index, int) and hasattr(node, "SetNthControlPointDescription"):
            node.SetNthControlPointDescription(index, str(description))


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
        self._spineBatchCases = []
        self._spineBatchCommands = []
        self._spineBatchCommandIndex = 0
        self._spineBatchDatasetRoot = None

        self._build_main_section()
        self.layout.addStretch(1)
        self._refresh_status()

    def cleanup(self):
        if self.logic.is_running():
            self.logic.interrupt()

    def _build_main_section(self):
        self.spineModeTabs = qt.QTabWidget()
        self.scenePage = qt.QWidget()
        self.batchPage = qt.QWidget()
        self.spineModeTabs.addTab(self.scenePage, "Scene")
        self.spineModeTabs.addTab(self.batchPage, "Batch")
        self.layout.addWidget(self.spineModeTabs)

        scene_layout = qt.QVBoxLayout(self.scenePage)
        self.runBox = ctk.ctkCollapsibleButton()
        self.runBox.text = "Run spine CT segmentation"
        scene_layout.addWidget(self.runBox)
        form = qt.QFormLayout(self.runBox)

        self.statusLabel = qt.QLabel("Checking runtime...")
        self.statusLabel.wordWrap = True
        self._tip(
            self.statusLabel,
            "Shows whether the Slicer Python and optional Conda MPS runtimes are available.",
        )
        form.addRow("Ready", self.statusLabel)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = False
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self._tip(self.volumeSelector, "Input clinical CT volume to segment.")
        form.addRow("Input CT", self.volumeSelector)

        self.modeCombo = qt.QComboBox()
        for label, value in [
            ("Full segmentation + centroids", "full"),
            ("Vertebral levels + centroids", "level"),
            ("Centroids only", "localization"),
        ]:
            self.modeCombo.addItem(label, value)
        self._tip(
            self.modeCombo,
            "Choose the output set. Body/process and cort/trab are generated together in full segmentation mode. Centroid markers are loaded for every completed run.",
        )
        form.addRow("Outputs", self.modeCombo)

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
        self._tip(self.runButton, "Export the selected CT, run spine-segment, and load centroids plus the selected segmentation outputs.")
        self._tip(self.stopButton, "Stop the active spine-segment process.")
        button_row.addWidget(self.runButton)
        button_row.addWidget(self.stopButton)
        form.addRow(button_row_widget)

        self.progressLabel = qt.QLabel("Idle")
        self.progressLabel.wordWrap = True
        self.progressBar = qt.QProgressBar()
        self.progressBar.minimum = 0
        self.progressBar.maximum = 100
        self.progressBar.value = 0
        self._tip(self.progressLabel, "Current spine-segment command status.")
        self._tip(self.progressBar, "Progress for the active spine-segment command.")
        form.addRow("Progress", self.progressLabel)
        form.addRow(self.progressBar)
        scene_layout.addStretch(1)

        self._build_batch_section(self.batchPage)

        self.runtimeBox = ctk.ctkCollapsibleButton()
        self.runtimeBox.text = "Runtime setup"
        self.runtimeBox.collapsed = True
        self.layout.addWidget(self.runtimeBox)
        runtime_form = qt.QFormLayout(self.runtimeBox)

        self.installCondaButton = qt.QPushButton("Install Conda MPS Runtime")
        self.updateToolboxButton = qt.QPushButton("Check Toolbox Updates")
        self._tip(
            self.installCondaButton,
            "Create or update the arm64 conda runtime used for faster Apple Silicon inference outside Slicer Python.",
        )
        self._tip(self.updateToolboxButton, "Check whether this local Slicer toolbox checkout has upstream updates.")
        self.installCondaButton.clicked.connect(self._install_conda_runtime)
        self.updateToolboxButton.clicked.connect(self._check_toolbox_updates)
        install_row_widget = qt.QWidget()
        install_row = qt.QHBoxLayout(install_row_widget)
        install_row.setContentsMargins(0, 0, 0, 0)
        install_row.addWidget(self.installCondaButton)
        install_row.addWidget(self.updateToolboxButton)
        runtime_form.addRow("Install", install_row_widget)

        self.runtimeCombo = qt.QComboBox()
        for label, value in [
            ("Auto: Conda MPS if available, otherwise Slicer", "auto"),
            ("Conda MPS runtime", "conda"),
            ("Slicer Python runtime", "slicer"),
        ]:
            self.runtimeCombo.addItem(label, value)
        self._tip(
            self.runtimeCombo,
            "Choose where spine-segment runs. Auto probes the conda runtime first, then falls back to Slicer Python.",
        )
        runtime_form.addRow("Runtime", self.runtimeCombo)

        self.condaPythonEdit = qt.QLineEdit(str(self.logic.default_conda_python_path()))
        self.probeRuntimeButton = qt.QPushButton("Probe runtime")
        self.probeRuntimeButton.clicked.connect(self._probe_runtime)
        self._tip(
            self.condaPythonEdit,
            "Python executable for the external arm64 conda runtime, usually ~/miniforge3/envs/spine-segment-pytorch/bin/python.",
        )
        self._tip(self.probeRuntimeButton, "Check whether this Python can import spine-segment and run PyTorch Conv3D on MPS.")
        runtime_path_row = qt.QHBoxLayout()
        runtime_path_row.addWidget(self.condaPythonEdit)
        runtime_path_row.addWidget(self.probeRuntimeButton)
        runtime_form.addRow("Conda Python", runtime_path_row)

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
        runtime_form.addRow("Device", self.deviceCombo)

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

    def _build_batch_section(self, parent):
        layout = qt.QVBoxLayout(parent)
        discovery_box = qt.QGroupBox("Discovery")
        discovery_form = qt.QFormLayout(discovery_box)
        layout.addWidget(discovery_box)

        self.batchDatasetRootSelector = ctk.ctkPathLineEdit()
        self.batchDatasetRootSelector.filters = ctk.ctkPathLineEdit.Dirs
        self._tip(
            self.batchDatasetRootSelector,
            "Dataset root to search for CT images and reusable derivatives.",
        )
        discovery_form.addRow("Dataset root", self.batchDatasetRootSelector)

        self.batchSubjectEdit = qt.QLineEdit()
        self.batchSiteEdit = qt.QLineEdit()
        self.batchSessionEdit = qt.QLineEdit()
        self.batchSiteEdit.text = "spine"
        discovery_form.addRow("Subject filter", self.batchSubjectEdit)
        discovery_form.addRow("Site filter", self.batchSiteEdit)
        discovery_form.addRow("Session filter", self.batchSessionEdit)

        self.batchDiscoverButton = qt.QPushButton("Discover")
        self.batchDiscoverButton.clicked.connect(self.discover_spine_batch)
        discovery_form.addRow("", self.batchDiscoverButton)

        self.batchTable = qt.QTableWidget()
        self.batchTable.setColumnCount(5)
        self.batchTable.setHorizontalHeaderLabels(["Run", "Subject", "Site", "Session", "Images"])
        self.batchTable.minimumHeight = 180
        try:
            self.batchTable.horizontalHeader().setStretchLastSection(True)
            self.batchTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        except Exception:
            pass
        layout.addWidget(self.batchTable)

        workflow_box = qt.QGroupBox("Workflow")
        workflow_form = qt.QFormLayout(workflow_box)
        layout.addWidget(workflow_box)

        self.batchImageRoleBox = qt.QComboBox()
        self.batchImageRoleBox.addItem("Auto")
        self.batchModeCombo = qt.QComboBox()
        for label, value in [
            ("Full segmentation + centroids", "full"),
            ("Vertebral levels + centroids", "level"),
            ("Centroids only", "localization"),
        ]:
            self.batchModeCombo.addItem(label, value)
        workflow_form.addRow("Image source", self.batchImageRoleBox)
        workflow_form.addRow("Outputs", self.batchModeCombo)

        self.batchStatusLabel = qt.QLabel(
            "Discover a dataset to prepare derivatives/SpineSegmentationCT batch cases."
        )
        self.batchStatusLabel.wordWrap = True
        layout.addWidget(self.batchStatusLabel)

        buttons = qt.QHBoxLayout()
        self.batchRunButton = qt.QPushButton("Run Batch")
        self.batchStopButton = qt.QPushButton("Stop")
        self.batchStopButton.enabled = False
        self.batchRunButton.clicked.connect(self.run_spine_batch)
        self.batchStopButton.clicked.connect(self.stop_spine_batch)
        buttons.addWidget(self.batchRunButton)
        buttons.addWidget(self.batchStopButton)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addStretch(1)

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
        slicer_status = "Slicer: spine-segment installed" if self.logic.is_core_available() else "Slicer: not installed"
        conda_text = getattr(self, "condaPythonEdit", None)
        conda_path = Path(conda_text.text).expanduser() if conda_text is not None else self.logic.default_conda_python_path()
        conda_status = "Conda: runtime found" if conda_path.exists() else "Conda: runtime not found"
        self.statusLabel.text = f"{slicer_status}; {conda_status}"

    def _install_core(self):
        try:
            self._append_log("[setup] installing spine-segment...\n")
            self.logic.install_or_update_core()
            self._refresh_status()
            self._append_log("[setup] spine-segment dependency is installed.\n")
        except Exception as exc:
            self._error(exc)

    def _install_conda_runtime(self):
        try:
            self._set_running(True, "Installing Conda MPS runtime...")
            probe = self.logic.install_or_update_conda_runtime(
                self._conda_python_path(),
                on_output=self._append_log,
            )
            self._append_log(f"[runtime] Conda probe: {self.logic.runtime_summary(probe)}\n")
            self._refresh_status()
            self._set_running(False, "Conda runtime ready")
        except Exception as exc:
            self._set_running(False, "Failed")
            self._error(exc)

    def _probe_runtime(self):
        try:
            probe = self.logic.probe_python_runtime(self._conda_python_path())
            self._append_log(f"[runtime] Conda probe: {self.logic.runtime_summary(probe)}\n")
            if probe.get("probe_output"):
                self._append_log(str(probe.get("probe_output")) + "\n")
            self._refresh_status()
        except Exception as exc:
            self._error(exc)

    def _check_toolbox_updates(self):
        run_toolbox_update_dialog(__file__, log=self._append_log)

    def _conda_python_path(self):
        return Path(self.condaPythonEdit.text).expanduser()

    def _run_segmentation(self):
        try:
            if self.logic.is_running():
                raise RuntimeError("A spine segmentation process is already running.")
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
                runtime=self.runtimeCombo.currentData,
                conda_python=self._conda_python_path(),
                on_output=self._append_log,
                on_finished=self._on_process_finished,
            )
        except Exception as exc:
            self._set_running(False, "Failed")
            self._error(exc)

    def discover_spine_batch(self):
        root = str(getattr(self.batchDatasetRootSelector, "currentPath", "") or "").strip()
        if not root:
            slicer.util.errorDisplay("Select a dataset root before discovery.")
            return
        try:
            self._spineBatchCases = self.logic.discover_spine_segmentation_batch_cases(
                root,
                subject_id=str(self.batchSubjectEdit.text or "").strip(),
                site=str(self.batchSiteEdit.text or "").strip(),
                session_id=str(self.batchSessionEdit.text or "").strip(),
            )
        except Exception as exc:
            self.batchStatusLabel.text = f"Discovery failed: {exc}"
            slicer.util.errorDisplay(str(exc))
            return
        self._populate_spine_batch_table()
        self._populate_spine_batch_image_roles()
        self._refresh_spine_batch_readiness()
        self._append_log(f"[spine batch] discovered {len(self._spineBatchCases)} case(s) from {root}\n")

    def _populate_spine_batch_table(self):
        self.batchTable.setRowCount(len(self._spineBatchCases))
        for row, case in enumerate(self._spineBatchCases):
            run_item = qt.QTableWidgetItem("")
            run_item.setFlags(run_item.flags() | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
            run_item.setCheckState(qt.Qt.Checked)
            self.batchTable.setItem(row, 0, run_item)
            values = [
                case.subject_id,
                case.site,
                case.session_id,
                ", ".join(discovered_image_roles([case])),
            ]
            for column, value in enumerate(values, start=1):
                item = qt.QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                self.batchTable.setItem(row, column, item)
        try:
            self.batchTable.resizeColumnsToContents()
        except Exception:
            pass

    def _populate_spine_batch_image_roles(self):
        try:
            blocked = self.batchImageRoleBox.blockSignals(True)
        except Exception:
            blocked = False
        self.batchImageRoleBox.clear()
        self.batchImageRoleBox.addItem("Auto")
        for role in discovered_image_roles(self._spineBatchCases):
            self.batchImageRoleBox.addItem(role)
        try:
            self.batchImageRoleBox.blockSignals(blocked)
        except Exception:
            pass

    def _selected_spine_batch_image_role(self):
        role = str(self.batchImageRoleBox.currentText or "").strip()
        return "" if role == "Auto" else role

    def _selected_spine_batch_cases(self):
        selected = []
        for row, case in enumerate(self._spineBatchCases):
            item = self.batchTable.item(row, 0)
            if item is None or item.checkState() == qt.Qt.Checked:
                selected.append(case)
        return selected

    def _refresh_spine_batch_readiness(self):
        count = len(getattr(self, "_spineBatchCases", []) or [])
        if not count:
            self.batchStatusLabel.text = "Discover a dataset to prepare derivatives/SpineSegmentationCT batch cases."
            return
        self.batchStatusLabel.text = f"{count}/{count} case(s) ready for spine segmentation batch."

    def run_spine_batch(self):
        if self.logic.is_running():
            slicer.util.errorDisplay("A spine segmentation process is already running.")
            return
        root = str(getattr(self.batchDatasetRootSelector, "currentPath", "") or "").strip()
        if not root:
            slicer.util.errorDisplay("Select a dataset root before running a batch.")
            return
        cases = self._selected_spine_batch_cases()
        if not cases:
            slicer.util.errorDisplay("Select at least one discovered spine segmentation case.")
            return
        try:
            commands = self.logic.build_spine_segmentation_batch_commands(
                root,
                cases,
                image_role=self._selected_spine_batch_image_role(),
                mode=self.batchModeCombo.currentData,
                device=self.deviceCombo.currentData,
            )
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
            return
        if not commands:
            slicer.util.errorDisplay("No selected spine segmentation cases have a usable CT image.")
            return
        self._spineBatchCommands = commands
        self._spineBatchCommandIndex = 0
        self._spineBatchDatasetRoot = Path(root).expanduser()
        self.batchRunButton.enabled = False
        self.batchStopButton.enabled = True
        self.batchStatusLabel.text = f"Running 0/{len(commands)} spine segmentation case(s)..."
        self._append_log(f"[spine batch] starting {len(commands)} case(s)\n")
        self._run_next_spine_batch_case()

    def _run_next_spine_batch_case(self):
        if self._spineBatchCommandIndex >= len(self._spineBatchCommands):
            self.batchRunButton.enabled = True
            self.batchStopButton.enabled = False
            total = len(self._spineBatchCommands)
            self.batchStatusLabel.text = f"Finished {total}/{total} spine segmentation case(s)."
            self._append_log("[spine batch] finished\n")
            return
        command = self._spineBatchCommands[self._spineBatchCommandIndex]
        case_number = self._spineBatchCommandIndex + 1
        total = len(self._spineBatchCommands)
        self.batchStatusLabel.text = (
            f"Running {case_number}/{total}: sub-{command.case.subject_id} ses-{command.case.session_id}"
        )
        self._append_log(f"[spine batch] running {case_number}/{total}: {' '.join(command.cli_args)}\n")
        self.logic.run_cli(
            command.input_path,
            command.output_dir,
            device=command.device,
            mode=command.mode,
            runtime=self.runtimeCombo.currentData,
            conda_python=self._conda_python_path(),
            on_output=self._append_log,
            on_finished=self._on_spine_batch_case_finished,
        )

    def _on_spine_batch_case_finished(self, exit_code, _exit_status, interrupted):
        if interrupted:
            self.batchRunButton.enabled = True
            self.batchStopButton.enabled = False
            self.batchStatusLabel.text = "Spine segmentation batch stopped."
            self._append_log("[spine batch] stopped\n")
            return
        if int(exit_code) != 0:
            self.batchRunButton.enabled = True
            self.batchStopButton.enabled = False
            self.batchStatusLabel.text = f"Spine segmentation batch failed at case {self._spineBatchCommandIndex + 1}."
            self._append_log(f"[spine batch] failed with exit code {exit_code}\n")
            return
        self._write_completed_spine_batch_manifest()
        self._spineBatchCommandIndex += 1
        self._run_next_spine_batch_case()

    def stop_spine_batch(self):
        self.logic.interrupt()

    def _write_completed_spine_batch_manifest(self):
        root = self._spineBatchDatasetRoot
        if root is None:
            return
        command = self._spineBatchCommands[self._spineBatchCommandIndex]
        try:
            manifest_path = self.logic.write_spine_segmentation_manifest(root, [command])
            self._append_log(f"[spine batch] derivative manifest: {manifest_path}\n")
        except Exception as exc:
            self._append_log(f"[spine batch] could not write derivative manifest: {exc}\n")

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
        if not centroids.exists():
            raise FileNotFoundError(
                "Centroid markers are expected for every completed spine segmentation run: "
                f"{centroids}"
            )
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
        self.installCondaButton.enabled = not bool(running)
        self.probeRuntimeButton.enabled = not bool(running)
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
