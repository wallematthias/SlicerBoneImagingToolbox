from __future__ import annotations

import json
from importlib import metadata
import os
from pathlib import Path
import sys
import tempfile

import ctk
import qt
import slicer

TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))
VOIDSPACE_LOCAL_SRC = TOOLBOX_ROOT.parent / "voidspace" / "src"
if VOIDSPACE_LOCAL_SRC.exists() and str(VOIDSPACE_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(VOIDSPACE_LOCAL_SRC))

from SlicerBoneImagingToolboxLib.slicer_pip import slicer_pip_install, slicer_python_executable  # noqa: E402

from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


MODULE_VERSION = "0.1.0"
VOIDSPACE_MINIMUM_VERSION = "0.1.3"
VOIDSPACE_CITATION = (
    "Whittier DE, Burt LA, Boyd SK. A new approach for quantifying localized bone loss by measuring void spaces. "
    "Bone. 2021 Feb;143:115785. doi: 10.1016/j.bone.2020.115785."
)
DYNAMIC_VOIDSPACE_CITATION = (
    "Whittier DE, Walle M, Atkins PR, Collins CJ, Zumstein MA, Christen P, Lippuner K, Müller R. "
    "Structural alterations during fracture healing lead to void spaces developing in surrounding bone microarchitecture. "
    "Journal of Bone and Mineral Research. 2025 Jun;40(6):791-798. doi: 10.1093/jbmr/zjaf046."
)

_VOIDSPACE_SCENE_PROCESS_SCRIPT = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(job_json_path: str) -> int:
    job = json.loads(Path(job_json_path).read_text(encoding="utf-8"))
    local_src = str(job.get("local_src") or "")
    if local_src and local_src not in sys.path:
        sys.path.insert(0, local_src)

    from voidspace import compare, run_case

    mode = str(job.get("mode") or "single")
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "longitudinal":
        baseline_dir = output_dir / "baseline"
        followup_dir = output_dir / "followup"
        change_dir = output_dir / "change"
        print("[voidspace] baseline", flush=True)
        baseline = run_case(
            segmentation_path=job["baseline_segmentation_path"],
            mask_path=job.get("baseline_mask_path") or None,
            output_dir=baseline_dir,
            force=True,
        )
        print("[voidspace] follow-up", flush=True)
        followup = run_case(
            segmentation_path=job["followup_segmentation_path"],
            mask_path=job.get("followup_mask_path") or None,
            output_dir=followup_dir,
            force=True,
        )
        print("[voidspace] compare", flush=True)
        change = compare(
            baseline_void_path=baseline.large_mask_path,
            followup_void_path=followup.large_mask_path,
            output_dir=change_dir,
            force=True,
        )
        outputs = {
            "baseline_all": str(baseline.all_mask_path),
            "baseline_large": str(baseline.large_mask_path),
            "followup_all": str(followup.all_mask_path),
            "followup_large": str(followup.large_mask_path),
            "expanded": str(change.expanded_path),
            "contracted": str(change.contracted_path),
            "measurements": str(change.measurements_path),
        }
    else:
        print("[voidspace] run case", flush=True)
        result = run_case(
            segmentation_path=job["segmentation_path"],
            mask_path=job.get("mask_path") or None,
            output_dir=output_dir,
            force=True,
        )
        outputs = {
            "all": str(result.all_mask_path),
            "large": str(result.large_mask_path),
            "measurements": str(result.measurements_path),
        }
    Path(job["outputs_json_path"]).write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print("[voidspace] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
"""


class VoidspaceHRpQCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Voidspace"
        parent.categories = ["Bone Imaging.Microstructural Analysis"]
        parent.index = 115
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Compute cross-sectional and longitudinal voidspace maps from segmentation masks.\n"
            f"Module version: {MODULE_VERSION}\n\n"
            f"Citation: {VOIDSPACE_CITATION}\n\n"
            f"Dynamic voidspace: {DYNAMIC_VOIDSPACE_CITATION}"
        )
        parent.acknowledgementText = (
            "Author: Matthias Walle. This module wraps the separate voidspace core package. "
            f"Please cite: {VOIDSPACE_CITATION}"
        )


class VoidspaceHRpQCTLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._proc = None

    @staticmethod
    def core_runtime_status():
        try:
            version = metadata.version("voidspace")
            import voidspace  # noqa: F401
        except Exception as exc:
            return False, f"voidspace is not available in Slicer Python: {exc}"
        return True, f"voidspace {version} is available."

    @staticmethod
    def install_or_update_core():
        return slicer_pip_install(f"--upgrade --prefer-binary voidspace>={VOIDSPACE_MINIMUM_VERSION}")

    @staticmethod
    def scene_cli_command(job_json_path: str) -> list[str]:
        return ["-c", _VOIDSPACE_SCENE_PROCESS_SCRIPT, str(job_json_path)]

    @staticmethod
    def _decode_qbytearray(raw) -> str:
        if isinstance(raw, (bytes, bytearray)):
            data = bytes(raw)
        else:
            try:
                data = raw.data()
                data = data.encode("utf-8", errors="replace") if isinstance(data, str) else bytes(data)
            except Exception:
                data = str(raw).encode("utf-8", errors="replace")
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _export_node(node, path: Path) -> Path:
        if node is None:
            raise ValueError("Select an input node.")
        if not slicer.util.saveNode(node, str(path)):
            raise RuntimeError(f"Could not save {node.GetName()} to {path}.")
        return path

    def prepare_single_job(self, *, segmentation_node, mask_node=None, output_dir=None) -> dict:
        job_dir = Path(output_dir or tempfile.mkdtemp(prefix="slicer-voidspace-"))
        input_dir = job_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "mode": "single",
            "local_src": str(VOIDSPACE_LOCAL_SRC) if VOIDSPACE_LOCAL_SRC.exists() else "",
            "output_dir": str(job_dir / "outputs"),
            "outputs_json_path": str(job_dir / "outputs.json"),
            "segmentation_path": str(self._export_node(segmentation_node, input_dir / "segmentation.nrrd")),
            "mask_path": "",
        }
        if mask_node is not None:
            job["mask_path"] = str(self._export_node(mask_node, input_dir / "mask.nrrd"))
        job_path = job_dir / "job.json"
        job["job_json_path"] = str(job_path)
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        return job

    def prepare_longitudinal_job(
        self,
        *,
        baseline_segmentation_node,
        baseline_mask_node=None,
        followup_segmentation_node,
        followup_mask_node=None,
        output_dir=None,
    ) -> dict:
        job_dir = Path(output_dir or tempfile.mkdtemp(prefix="slicer-voidspace-longitudinal-"))
        input_dir = job_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "mode": "longitudinal",
            "local_src": str(VOIDSPACE_LOCAL_SRC) if VOIDSPACE_LOCAL_SRC.exists() else "",
            "output_dir": str(job_dir / "outputs"),
            "outputs_json_path": str(job_dir / "outputs.json"),
            "baseline_segmentation_path": str(self._export_node(baseline_segmentation_node, input_dir / "baseline_segmentation.nrrd")),
            "baseline_mask_path": "",
            "followup_segmentation_path": str(self._export_node(followup_segmentation_node, input_dir / "followup_segmentation.nrrd")),
            "followup_mask_path": "",
        }
        if baseline_mask_node is not None:
            job["baseline_mask_path"] = str(self._export_node(baseline_mask_node, input_dir / "baseline_mask.nrrd"))
        if followup_mask_node is not None:
            job["followup_mask_path"] = str(self._export_node(followup_mask_node, input_dir / "followup_mask.nrrd"))
        job_path = job_dir / "job.json"
        job["job_json_path"] = str(job_path)
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        return job

    def run_scene_job(self, job: dict, *, on_output=None, on_finished=None):
        proc = qt.QProcess()
        proc.setProcessChannelMode(qt.QProcess.MergedChannels)
        if hasattr(qt, "QProcessEnvironment") and hasattr(proc, "setProcessEnvironment"):
            env = qt.QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONUNBUFFERED", "1")
            pythonpath = os.pathsep.join(path for path in (str(TOOLBOX_ROOT), str(VOIDSPACE_LOCAL_SRC)) if path)
            env.insert("PYTHONPATH", pythonpath)
            proc.setProcessEnvironment(env)

        def _read_output():
            text = self._decode_qbytearray(proc.readAll())
            if on_output and text:
                on_output(text)

        def _finished(*signal_args):
            exit_code = int(signal_args[0]) if signal_args else int(proc.exitCode())
            exit_status = signal_args[1] if len(signal_args) > 1 else proc.exitStatus()
            if on_finished:
                on_finished(exit_code, exit_status)

        proc.readyRead.connect(_read_output)
        proc.finished.connect(_finished)
        self._proc = proc
        proc.start(slicer_python_executable(slicer.app.applicationFilePath()), self.scene_cli_command(job["job_json_path"]))
        return proc

    @staticmethod
    def load_scene_outputs(job: dict) -> list:
        outputs_path = Path(job["outputs_json_path"])
        if not outputs_path.exists():
            raise FileNotFoundError(outputs_path)
        outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
        nodes = []
        for role, path_text in outputs.items():
            path = Path(path_text)
            if not path.exists():
                continue
            if role == "measurements":
                node = slicer.util.loadTable(str(path))
            else:
                node = slicer.util.loadLabelVolume(str(path))
            if node is not None:
                try:
                    node.SetName(f"Voidspace {role.replace('_', ' ')}")
                except Exception:
                    pass
                nodes.append(node)
        return nodes


class VoidspaceHRpQCTWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = VoidspaceHRpQCTLogic()
        self._currentJob = None

        scene_box = ctk.ctkCollapsibleButton()
        scene_box.text = "Scene"
        scene_box.collapsed = False
        self.layout.addWidget(scene_box)
        form = qt.QFormLayout(scene_box)

        self.longitudinalCheck = qt.QCheckBox()
        self.longitudinalCheck.toggled.connect(self._update_mode_visibility)
        form.addRow("Longitudinal voidspace", self.longitudinalCheck)

        self.segmentationSelector = self._volume_selector("Segmentation")
        self.maskSelector = self._volume_selector("Mask")
        self.baselineSegmentationSelector = self._volume_selector("Baseline segmentation")
        self.baselineMaskSelector = self._volume_selector("Baseline mask")
        self.followupSegmentationSelector = self._volume_selector("Follow-up segmentation")
        self.followupMaskSelector = self._volume_selector("Follow-up mask")

        form.addRow("Segmentation", self.segmentationSelector)
        form.addRow("Mask", self.maskSelector)
        form.addRow("Baseline segmentation", self.baselineSegmentationSelector)
        form.addRow("Baseline mask", self.baselineMaskSelector)
        form.addRow("Follow-up segmentation", self.followupSegmentationSelector)
        form.addRow("Follow-up mask", self.followupMaskSelector)

        self.runButton = qt.QPushButton("Run")
        self.runButton.clicked.connect(self._run)
        form.addRow("", self.runButton)

        self.statusLabel = qt.QLabel()
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)

        self.log = qt.QTextEdit()
        self.log.readOnly = True
        self.log.minimumHeight = 120
        self.layout.addWidget(self.log)
        self.layout.addStretch(1)
        self._update_mode_visibility(False)

    @staticmethod
    def _volume_selector(label: str):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode"]
        selector.selectNodeUponCreation = True
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = True
        selector.showHidden = False
        selector.showChildNodeTypes = False
        selector.setMRMLScene(slicer.mrmlScene)
        selector.toolTip = f"Select {label.lower()} volume."
        return selector

    def _update_mode_visibility(self, checked):
        single = not bool(checked)
        for selector in (self.segmentationSelector, self.maskSelector):
            selector.visible = single
        for selector in (
            self.baselineSegmentationSelector,
            self.baselineMaskSelector,
            self.followupSegmentationSelector,
            self.followupMaskSelector,
        ):
            selector.visible = not single

    def _append_log(self, text: str):
        self.log.insertPlainText(str(text))
        self.log.ensureCursorVisible()

    def _run(self):
        self.runButton.enabled = False
        try:
            if self.longitudinalCheck.checked:
                job = self.logic.prepare_longitudinal_job(
                    baseline_segmentation_node=self.baselineSegmentationSelector.currentNode(),
                    baseline_mask_node=self.baselineMaskSelector.currentNode(),
                    followup_segmentation_node=self.followupSegmentationSelector.currentNode(),
                    followup_mask_node=self.followupMaskSelector.currentNode(),
                )
            else:
                job = self.logic.prepare_single_job(
                    segmentation_node=self.segmentationSelector.currentNode(),
                    mask_node=self.maskSelector.currentNode(),
                )
            self._currentJob = job
        except Exception as exc:
            self.statusLabel.text = str(exc)
            self.runButton.enabled = True
            return

        self.statusLabel.text = "Running voidspace..."
        self.logic.run_scene_job(job, on_output=self._append_log, on_finished=self._on_finished)

    def _on_finished(self, exit_code, _exit_status):
        self.runButton.enabled = True
        if int(exit_code) != 0:
            self.statusLabel.text = f"Voidspace failed with exit code {exit_code}."
            return
        try:
            nodes = self.logic.load_scene_outputs(self._currentJob)
        except Exception as exc:
            self.statusLabel.text = f"Voidspace completed, but loading outputs failed: {exc}"
            return
        self.statusLabel.text = f"Voidspace complete. Loaded {len(nodes)} output node(s)."


class VoidspaceHRpQCTTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("VoidspaceHRpQCT module loaded.")
