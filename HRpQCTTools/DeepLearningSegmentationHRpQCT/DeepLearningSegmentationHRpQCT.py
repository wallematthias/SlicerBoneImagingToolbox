from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

import ctk
import numpy as np
import qt
import SimpleITK as sitk
import sitkUtils
import slicer
import vtk

_TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TOOLBOX_ROOT))
import SlicerBoneImagingToolboxLib as _toolbox_lib
_LOCAL_LIB = _TOOLBOX_ROOT/"SlicerBoneImagingToolboxLib"
if str(_LOCAL_LIB) not in _toolbox_lib.__path__:
    _toolbox_lib.__path__.insert(0, str(_LOCAL_LIB))
from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import (
    build_scene_command, local_core_source, DEFAULT_MODEL_LABEL, validate_deep_learning_compartments,
    model_cache_directory,
    scene_worker_args,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule, ScriptedLoadableModuleWidget, ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest,
)

MODULE_VERSION = "0.1.0"
INPUT_REFERENCE = "DeepLearningSegmentationHRpQCT.InputVolume"
SEGMENTS = {"full": ("Full mask", (.2,.8,.25)), "trab": ("Trabecular mask", (0,.75,1)),
            "cort": ("Cortical mask", (1,.55,.1))}


def reference_geometry(node):
    if node is None or node.GetScene() is not slicer.mrmlScene or node.GetImageData() is None:
        raise ValueError("The selected scan is no longer available in the scene.")
    if node.GetParentTransformNode() is not None:
        raise ValueError("Use an untransformed BMD volume for deep-learning inference.")
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return (tuple(node.GetImageData().GetDimensions()),
            tuple(matrix.GetElement(r,c) for r in range(4) for c in range(4)))


def voxel_digest(node):
    return hashlib.blake2b(np.ascontiguousarray(slicer.util.arrayFromVolume(node)).tobytes(), digest_size=16).hexdigest()


def slicer_python():
    app = Path(slicer.app.applicationDirPath())
    for candidate in (app/"PythonSlicer", app.parent/"bin"/"PythonSlicer"):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("PythonSlicer") or sys.executable


class DeepLearningSegmentationHRpQCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Deep Learning Segmentation"
        parent.categories = ["Bone Imaging.Microstructural Analysis"]
        parent.index = 65
        parent.dependencies = ["ScancoIO"]
        parent.contributors = [
            "Nathan J. Neeteson (published method)",
            "Bryce A. Besler (published method)",
            "Danielle E. Whittier (published method)",
            "Steven K. Boyd (published method)",
        ]
        parent.helpText = (
            "Published radius/tibia U-Net with fixed defaults. Input must be calibrated BMD. "
            "Headless batch uses the same core.<br><br>"
            "The segmentation method and published trained model are from Nathan J. Neeteson, "
            "Bryce A. Besler, Danielle E. Whittier and Steven K. Boyd, "
            "Bone Imaging Laboratory, University of Calgary. "
            "Please cite the original paper when using this method."
        )
        parent.acknowledgementText = (
            "Neeteson NJ, Besler BA, Whittier DE, Boyd SK. Automatic segmentation of trabecular "
            "and cortical compartments in HR-pQCT images using an embedding-predicting U-Net "
            "and morphological post-processing. Scientific Reports. 2023;13:252. "
            '<a href="https://doi.org/10.1038/s41598-022-27350-0">Original paper</a>.<br>'
            'Original code: <a href="https://github.com/Bonelab/HR-pQCT-Segmentation">'
            "Bonelab/HR-pQCT-Segmentation</a>. "
            'Published weights: <a href="https://doi.org/10.5281/zenodo.14755838">Zenodo</a>. '
            "Scientific code and published weights retain their upstream GPL-3.0 license "
            "in the bone-contouring U-Net backend."
        )


class DeepLearningSegmentationHRpQCTLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._proc = None
        self._workspace = None
        self.context = None
        self._on_output = self._on_finished = None
        self._cancelled = False

    def is_running(self):
        return self._proc is not None

    def detach_callbacks(self):
        self._on_output = self._on_finished = None

    def _cleanup_workspace(self):
        if self._workspace:
            self._workspace.cleanup()
        self._workspace = None
        self.context = None

    def run_segmentation(self, reference, device="auto", *, on_output=None, on_finished=None):
        if self.is_running():
            raise RuntimeError("A segmentation is already running.")
        geometry = reference_geometry(reference)
        scaling = str(reference.GetAttribute("HRpQCT.AIMScaling") or "").lower()
        if scaling not in ("density", "bmd"):
            raise ValueError("Load the scan through Scanco I/O using Density/BMD scaling; native/HU/unknown units are not accepted.")
        self._workspace = tempfile.TemporaryDirectory(prefix="hrpqct-unet-", dir=slicer.app.temporaryPath)
        directory = Path(self._workspace.name)
        snapshot, output = directory/"density.npz", directory/"masks.npz"
        try:
            np.savez(snapshot, density=np.array(slicer.util.arrayFromVolume(reference), copy=True))
            self.context = dict(reference=reference, geometry=geometry, digest=voxel_digest(reference), output=output)
            command = build_scene_command(slicer_python(), snapshot, output, device)
            self._start(command, on_output, on_finished)
        except Exception:
            self._cleanup_workspace()
            raise

    def _start(self, command, on_output=None, on_finished=None):
        self._cancelled = False
        self._on_output, self._on_finished = on_output, on_finished
        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.MergedChannels)
        proc.setWorkingDirectory(str(command.cwd))
        environment = qt.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        if not environment.value("HRPQCT_SEGMENTATION_MODEL_DIR"):
            environment.insert("HRPQCT_SEGMENTATION_MODEL_DIR", str(model_cache_directory(
                qt.QStandardPaths.writableLocation(qt.QStandardPaths.AppDataLocation))))
        for key in ("ITK_AUTOLOAD_PATH", "SITK_AUTOLOAD_PATH"):
            environment.insert(key, "")
        local = local_core_source(_TOOLBOX_ROOT)
        if local:
            existing = str(environment.value("PYTHONPATH") or "")
            environment.insert("PYTHONPATH", str(local)+(os.pathsep+existing if existing else ""))
        proc.setProcessEnvironment(environment)

        def read_output():
            raw = proc.readAll()
            try:
                data = bytes(raw)
            except Exception:
                data = raw.data()
                if isinstance(data, str):
                    data = data.encode()
            if data and self._on_output:
                self._on_output(data.decode("utf-8", errors="replace"))

        def finished(*args):
            if self._proc is not proc:
                return
            read_output()
            self._proc = None
            callback = self._on_finished
            self.detach_callbacks()
            completed_workspace, completed_context = self._workspace, self.context
            code = int(args[0]) if args else int(proc.exitCode())
            status = args[1] if len(args)>1 else proc.exitStatus()
            try:
                if callback:
                    callback(code, status, self._cancelled)
            finally:
                proc.deleteLater()
                if completed_workspace is not None:
                    completed_workspace.cleanup()
                if self._workspace is completed_workspace:
                    self._workspace = None
                if self.context is completed_context:
                    self.context = None

        proc.readyRead.connect(read_output)
        proc.finished.connect(finished)
        self._proc = proc
        proc.start(command.program, scene_worker_args(command.args, local))
        if not proc.waitForStarted(3000):
            message = proc.errorString()
            self._proc = None
            self.detach_callbacks()
            proc.deleteLater()
            raise RuntimeError(f"Could not start segmentation worker: {message}")

    def interrupt(self):
        proc = self._proc
        if proc is None:
            return False
        self._cancelled = True
        proc.terminate()
        def kill_if_needed():
            if self._proc is proc:
                proc.kill()
        qt.QTimer.singleShot(1500, kill_if_needed)
        return True

    def import_outputs(self, reference, output, *, expected_geometry=None, expected_digest=None):
        geometry = reference_geometry(reference)
        if expected_geometry is not None and geometry != expected_geometry:
            raise ValueError("Selected scan geometry changed during inference; outputs were not imported.")
        if expected_digest is not None and voxel_digest(reference) != expected_digest:
            raise ValueError("Selected scan voxels changed during inference; outputs were not imported.")
        image = sitkUtils.PullVolumeFromSlicer(reference)
        with np.load(output, allow_pickle=False) as data:
            arrays = {role: data[role] for role in SEGMENTS}
        images = []
        for role in ("cort", "trab"):
            mask = sitk.GetImageFromArray(arrays[role])
            if mask.GetSize() != image.GetSize():
                raise ValueError("Predicted masks do not match the selected scan dimensions.")
            mask.CopyInformation(image)
            images.append(mask)
        masks = validate_deep_learning_compartments(*images, image)
        if not np.array_equal(arrays["full"], sitk.GetArrayFromImage(masks["full"])):
            raise ValueError("Full mask does not equal the cortical/trabecular union.")
        existing = next((node for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
                         if node.GetNodeReferenceID(INPUT_REFERENCE) == reference.GetID()), None)
        if existing is not None and existing.GetParentTransformNode() is not None:
            raise ValueError("The previous U-Net result has a parent transform. Remove/harden that transform or remove the result before regenerating.")
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{reference.GetName()}_UNet_compartments")
        try:
            node.CreateDefaultDisplayNodes()
            node.SetReferenceImageGeometryParameterFromVolumeNode(reference)
            for role, (name,color) in SEGMENTS.items():
                segment_id = node.GetSegmentation().AddEmptySegment(name,name,color)
                node.GetSegmentation().GetSegment(segment_id).SetTag("HRpQCT.Role", role)
                slicer.util.updateSegmentBinaryLabelmapFromArray(sitk.GetArrayFromImage(masks[role]), node, segment_id, reference)
            if existing:
                existing.GetSegmentation().DeepCopy(node.GetSegmentation())
                slicer.mrmlScene.RemoveNode(node)
                node = existing
            node.SetNodeReferenceID(INPUT_REFERENCE, reference.GetID())
            node.SetAttribute("BoneImaging.MaskRoles", "full,trab,cort")
            node.SetAttribute("DeepLearningSegmentationHRpQCT.Model", DEFAULT_MODEL_LABEL)
            node.SetAttribute("DeepLearningSegmentationHRpQCT.Method", "Published UNet defaults; frozen scene BMD voxels; full=trab|cort; no extra toolbox peel")
            display = node.GetDisplayNode()
            display.SetOpacity2DFill(.35)
            display.SetSegmentVisibility(node.GetSegmentation().GetSegmentIdBySegmentName("Full mask"), False)
            hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            hierarchy.SetItemParent(hierarchy.GetItemByDataNode(node), hierarchy.GetItemParent(hierarchy.GetItemByDataNode(reference)))
            return node
        except Exception:
            if node is not existing:
                slicer.mrmlScene.RemoveNode(node)
            raise


class DeepLearningSegmentationHRpQCTWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = DeepLearningSegmentationHRpQCTLogic()
        self._closed = False
        box = ctk.ctkCollapsibleButton()
        box.text = "Scene Segmentation"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.inputVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelector.addEnabled = self.inputVolumeSelector.removeEnabled = False
        self.inputVolumeSelector.noneEnabled = True
        self.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("BMD volume", self.inputVolumeSelector)
        self.deviceCombo = qt.QComboBox()
        for label,value in (("Automatic","auto"),("CPU","cpu"),("CUDA","cuda"),("Apple MPS","mps")):
            self.deviceCombo.addItem(label,value)
        form.addRow("Device", self.deviceCombo)
        buttons = qt.QHBoxLayout()
        self.runButton, self.cancelButton = qt.QPushButton("Generate"), qt.QPushButton("Cancel")
        self.cancelButton.enabled = False
        buttons.addWidget(self.runButton)
        buttons.addWidget(self.cancelButton)
        form.addRow(buttons)
        self.statusLabel = qt.QLabel("Published 61 µm radius/tibia weights. XCT-I and knee use are unvalidated. Returns compartments, not a bone-tissue SEG.")
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)
        self.outputText = qt.QPlainTextEdit()
        self.outputText.readOnly = True
        self.outputText.maximumBlockCount = 1000
        self.layout.addWidget(self.outputText)
        self.layout.addStretch(1)
        self.runButton.clicked.connect(self._on_run)
        self.cancelButton.clicked.connect(self._on_cancel)

    def cleanup(self):
        self._closed = True
        self.logic.detach_callbacks()
        self.logic.interrupt()

    def _append_output(self, text):
        if not self._closed:
            self.outputText.appendPlainText(str(text).rstrip())

    def _on_run(self):
        try:
            value = self.deviceCombo.currentData
            device = str(value() if callable(value) else value)
            self.runButton.enabled, self.cancelButton.enabled = False, True
            self.outputText.clear()
            self.statusLabel.text = "Running published U-Net defaults..."
            self.logic.run_segmentation(self.inputVolumeSelector.currentNode(), device,
                                        on_output=self._append_output, on_finished=self._on_finished)
        except Exception as exc:
            self.runButton.enabled, self.cancelButton.enabled = True, False
            self.statusLabel.text = f"Could not start: {exc}"

    def _on_cancel(self):
        if self.logic.interrupt():
            self.statusLabel.text = "Cancelling..."

    def _on_finished(self, code, status, cancelled=False):
        if self._closed:
            return
        self.runButton.enabled, self.cancelButton.enabled = True, False
        if cancelled:
            self.statusLabel.text = "Cancelled."
        elif code != 0 or status != qt.QProcess.NormalExit:
            self.statusLabel.text = "Inference failed. See the output log; install/update the core through Toolbox Setup."
        else:
            try:
                context = self.logic.context
                node = self.logic.import_outputs(context["reference"], context["output"],
                    expected_geometry=context["geometry"], expected_digest=context["digest"])
                self.statusLabel.text = f"Loaded {node.GetName()}: full, trabecular and cortical compartments."
            except Exception as exc:
                self.statusLabel.text = f"Output validation/import failed: {exc}"


class DeepLearningSegmentationHRpQCTTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.assertFalse(DeepLearningSegmentationHRpQCTLogic().is_running())
