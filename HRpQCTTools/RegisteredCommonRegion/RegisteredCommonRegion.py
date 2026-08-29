from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import qt
import slicer
import SimpleITK as sitk

TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))
TIMELAPSED_LOCAL_SRC = TOOLBOX_ROOT.parent / "TimelapsedHRpQCT" / "src"
if TIMELAPSED_LOCAL_SRC.exists() and str(TIMELAPSED_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(TIMELAPSED_LOCAL_SRC))

from SlicerBoneImagingToolboxLib.common_region import CommonRegionSession, build_common_scan_region  # noqa: E402
from SlicerBoneImagingToolboxLib.derivatives import DerivativeManifest, DerivativeRecord, write_manifest  # noqa: E402
from SlicerBoneImagingToolboxLib.image_io import read_image, read_mask, write_mask  # noqa: E402
from SlicerBoneImagingToolboxLib.registration import register_image_pair  # noqa: E402
from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


MODULE_VERSION = "0.1.0"
DERIVATIVE_NAME = "CommonRegion"


class RegisteredCommonRegion(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Registered Common Region"
        parent.categories = ["Bone Imaging.HR-pQCT"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Build longitudinal registered scan/FOV common-region masks for reuse by analysis modules.\n"
            f"Module version: {MODULE_VERSION}"
        )


class RegisteredCommonRegionLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._proc = None

    def derivatives_root(self, dataset_root, derivatives_root=""):
        if derivatives_root:
            return Path(str(derivatives_root)).expanduser().resolve()
        return Path(str(dataset_root)).expanduser().resolve() / "derivatives" / "CommonRegion"

    def write_common_region_manifest(self, path, *, dataset_root, records):
        manifest = DerivativeManifest(
            workflow=DERIVATIVE_NAME,
            version=MODULE_VERSION,
            dataset_root=str(dataset_root),
            records=list(records),
            metadata={"module": "RegisteredCommonRegion"},
        )
        return write_manifest(path, manifest)

    def discover_batch_series(self, dataset_root):
        from timelapsedhrpqct.dataset.discovery import discover_raw_sessions
        from timelapsedhrpqct.utils.session_ids import session_sort_key

        sessions = discover_raw_sessions(Path(str(dataset_root)).expanduser().resolve())
        rows = []
        for session in sessions:
            masks = dict(getattr(session, "raw_mask_paths", {}) or {})
            rows.append(
                {
                    "subject_id": str(session.subject_id),
                    "site": str(session.site),
                    "session_id": str(session.session_id),
                    "stack_index": int(getattr(session, "stack_index", 1) or 1),
                    "image_path": str(session.image_path),
                    "registration_mask_path": str(masks.get("full") or masks.get("regmask") or ""),
                }
            )
        return sorted(rows, key=lambda row: (row["subject_id"], row["site"], int(row["stack_index"]), session_sort_key(row["session_id"])))

    def run_batch_common_region(self, job, *, progress_callback=None):
        from timelapsedhrpqct.utils.session_ids import session_sort_key
        from timelapsedhrpqct.processing.transform_chain import (
            PairwiseTransform,
            compose_sequential_to_baseline,
            flatten_transform,
        )

        dataset_root = Path(job["dataset_root"]).expanduser().resolve()
        root = self.derivatives_root(dataset_root, job.get("derivatives_root", ""))
        rows = list(job.get("rows", []))
        groups = {}
        for row in rows:
            groups.setdefault((row["subject_id"], row["site"], int(row.get("stack_index", 1))), []).append(row)

        records = []
        for (_subject_id, _site, _stack), group_rows in groups.items():
            ordered = sorted(group_rows, key=lambda row: session_sort_key(row["session_id"]))
            if len(ordered) < 2:
                continue
            baseline = ordered[0]
            baseline_image = read_image(baseline["image_path"])
            transforms = {baseline["session_id"]: sitk.Transform(3, sitk.sitkIdentity)}
            pairwise = []
            previous = baseline
            previous_image = baseline_image
            previous_mask = read_mask(previous["registration_mask_path"])
            for row in ordered[1:]:
                self._progress(
                    progress_callback,
                    f"[common-region] registering sub-{row['subject_id']} ses-{row['session_id']} to ses-{previous['session_id']}",
                )
                image = read_image(row["image_path"])
                mask = read_mask(row["registration_mask_path"])
                result = register_image_pair(
                    fixed_image=previous_image,
                    moving_image=image,
                    fixed_mask=previous_mask,
                    moving_mask=mask,
                )
                pairwise_transform = flatten_transform(result.transform)
                pairwise.append(PairwiseTransform(session_id=row["session_id"], transform=pairwise_transform))
                transform_path = self._pairwise_transform_path(root, row, previous)
                transform_path.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteTransform(pairwise_transform, str(transform_path))
                records.append(self._record("Registration", "transform_pairwise", row, transform_path, "native", {"fixed_session_id": previous["session_id"]}))
                previous = row
                previous_image = image
                previous_mask = mask

            composed = compose_sequential_to_baseline(
                pairwise_transforms=pairwise,
                baseline_session_id=baseline["session_id"],
                dimension=3,
            )
            transforms.update({item.session_id: item.transform for item in composed})
            for row in ordered:
                transform_path = self._composed_transform_path(root, row, baseline)
                transform_path.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteTransform(flatten_transform(transforms[row["session_id"]]), str(transform_path))
                records.append(self._record("Registration", "transform_composed", row, transform_path, "reference", {"reference_session_id": baseline["session_id"]}))

            common_sessions = [
                CommonRegionSession(
                    subject_id=row["subject_id"],
                    site=row["site"],
                    session_id=row["session_id"],
                    stack_index=int(row.get("stack_index", 1)),
                    image=read_image(row["image_path"]),
                    transform_to_reference=transforms[row["session_id"]],
                )
                for row in ordered
            ]
            result = build_common_scan_region(common_sessions, reference_session_id=baseline["session_id"])
            common_path = self._common_mask_path(root, baseline)
            write_mask(common_path, result.common_mask)
            records.append(self._record(DERIVATIVE_NAME, "scan_region_common", baseline, common_path, "reference", {"reference_session_id": baseline["session_id"]}))
            for row in ordered:
                native_path = self._native_common_mask_path(root, row)
                write_mask(native_path, result.native_masks[row["session_id"]])
                records.append(self._record(DERIVATIVE_NAME, "scan_region_native_common", row, native_path, "native", {"reference_session_id": baseline["session_id"]}))

        manifest_path = root / "manifest.json"
        self.write_common_region_manifest(manifest_path, dataset_root=dataset_root, records=records)
        return {"manifest": str(manifest_path), "records": [record.__dict__ for record in records]}

    def run_batch_job(self, job, *, on_output=None, on_finished=None):
        proc = qt.QProcess()
        env = qt.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        self._proc = proc

        def _read_stdout():
            if on_output:
                on_output(bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace"))

        def _read_stderr():
            if on_output:
                on_output(bytes(proc.readAllStandardError()).decode("utf-8", errors="replace"))

        def _finished(exit_code, exit_status):
            self._proc = None
            if on_finished:
                on_finished(exit_code, exit_status)

        proc.readyReadStandardOutput.connect(_read_stdout)
        proc.readyReadStandardError.connect(_read_stderr)
        proc.finished.connect(_finished)
        job_path = Path(job["job_json_path"])
        python_exe = slicer.app.applicationFilePath()
        proc.start(python_exe, [str(Path(__file__).resolve()), "--registered-common-region-job", str(job_path)])
        return proc

    def _write_job(self, job, root):
        job_dir = Path(root) / "slicer_run_configs"
        job_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="registered_common_region_", suffix=".json", dir=str(job_dir))
        Path(path).write_text(json.dumps(job, indent=2), encoding="utf-8")
        return Path(path)

    def prepare_batch_job(self, dataset_root, derivatives_root, rows):
        root = self.derivatives_root(dataset_root, derivatives_root)
        job = {"dataset_root": str(dataset_root), "derivatives_root": str(root), "rows": list(rows)}
        job["job_json_path"] = str(self._write_job(job, root))
        return job

    @staticmethod
    def _progress(callback, message):
        if callback:
            callback(str(message))

    @staticmethod
    def _record(derivative, role, row, path, space, metadata):
        return DerivativeRecord(
            derivative=derivative,
            role=role,
            subject_id=str(row["subject_id"]),
            site=str(row["site"]),
            session_id=str(row["session_id"]),
            stack_index=int(row.get("stack_index", 1)),
            space=space,
            path=str(path),
            source="generated",
            metadata=dict(metadata),
        )

    def _subject_site_dir(self, root, row):
        return Path(root) / f"sub-{row['subject_id']}" / f"site-{row['site']}" / f"stack-{int(row.get('stack_index', 1)):02d}"

    def _pairwise_transform_path(self, root, moving_row, fixed_row):
        return self._subject_site_dir(root, moving_row) / "registration" / "pairwise" / (
            f"sub-{moving_row['subject_id']}_site-{moving_row['site']}_stack-{int(moving_row.get('stack_index', 1)):02d}_"
            f"from-ses-{moving_row['session_id']}_to-ses-{fixed_row['session_id']}.tfm"
        )

    def _composed_transform_path(self, root, row, baseline_row):
        return self._subject_site_dir(root, row) / "registration" / "composed" / (
            f"sub-{row['subject_id']}_site-{row['site']}_stack-{int(row.get('stack_index', 1)):02d}_"
            f"from-ses-{row['session_id']}_to-ses-{baseline_row['session_id']}.tfm"
        )

    def _common_mask_path(self, root, row):
        return self._subject_site_dir(root, row) / "common_space" / "masks" / (
            f"sub-{row['subject_id']}_site-{row['site']}_stack-{int(row.get('stack_index', 1)):02d}_mask-scan-region_common.nii.gz"
        )

    def _native_common_mask_path(self, root, row):
        return self._subject_site_dir(root, row) / "native_space" / f"ses-{row['session_id']}" / "masks" / (
            f"sub-{row['subject_id']}_ses-{row['session_id']}_site-{row['site']}_mask-scan-region_native_common.nii.gz"
        )


class RegisteredCommonRegionWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = RegisteredCommonRegionLogic()
        self.tabs = qt.QTabWidget()
        self.layout.addWidget(self.tabs)
        scene_tab = qt.QWidget()
        batch_tab = qt.QWidget()
        self.tabs.addTab(scene_tab, "Scene")
        self.tabs.addTab(batch_tab, "Batch")
        self._setup_scene_tab(scene_tab)
        self._setup_batch_tab(batch_tab)
        self.layout.addStretch(1)

    def _setup_scene_tab(self, parent):
        layout = qt.QFormLayout(parent)
        self.baselineVolumeSelector = slicer.qMRMLNodeComboBox()
        self.baselineVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.baselineVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.followupVolumeSelector = slicer.qMRMLNodeComboBox()
        self.followupVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.followupVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.baselineMaskSelector = slicer.qMRMLNodeComboBox()
        self.baselineMaskSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLSegmentationNode"]
        self.baselineMaskSelector.setMRMLScene(slicer.mrmlScene)
        self.followupMaskSelector = slicer.qMRMLNodeComboBox()
        self.followupMaskSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLSegmentationNode"]
        self.followupMaskSelector.setMRMLScene(slicer.mrmlScene)
        layout.addRow("Baseline volume", self.baselineVolumeSelector)
        layout.addRow("Follow-up volume", self.followupVolumeSelector)
        layout.addRow("Baseline registration mask", self.baselineMaskSelector)
        layout.addRow("Follow-up registration mask", self.followupMaskSelector)
        self.sceneRunButton = qt.QPushButton("Run")
        layout.addRow(self.sceneRunButton)

    def _setup_batch_tab(self, parent):
        layout = qt.QFormLayout(parent)
        self.datasetRootEdit = qt.QLineEdit()
        self.derivativesRootEdit = qt.QLineEdit()
        self.discoverButton = qt.QPushButton("Discover")
        self.runButton = qt.QPushButton("Run")
        self.statusLog = qt.QPlainTextEdit()
        self.statusLog.readOnly = True
        layout.addRow("Dataset root", self.datasetRootEdit)
        layout.addRow("Derivatives root", self.derivativesRootEdit)
        layout.addRow(self.discoverButton)
        layout.addRow(self.runButton)
        layout.addRow(self.statusLog)


def run_registered_common_region_job(job_path):
    logic = RegisteredCommonRegionLogic()
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))

    def print_progress(message):
        print(message, flush=True)

    result = logic.run_batch_common_region(job, progress_callback=print_progress)
    result_path = Path(job_path).with_suffix(".result.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--registered-common-region-job":
        run_registered_common_region_job(argv[1])
        return 0
    return 0


class RegisteredCommonRegionTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.test_RegisteredCommonRegion1()

    def test_RegisteredCommonRegion1(self):
        logic = RegisteredCommonRegionLogic()
        self.assertTrue(logic.derivatives_root("/tmp").name == "CommonRegion")


if __name__ == "__main__":
    raise SystemExit(_main())
