"""Headless Slicer integration checks; set HRPQCT_TEST_DENSITY to an NPZ.

Without that optional real BMD fixture, only worker lifecycle and import checks
run. With it, three middle slices run the actual published weights in Slicer's
Python. This is an integration check, not cohort/model accuracy validation.
"""
from pathlib import Path
import os
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import numpy as np
import qt
import SimpleITK as sitk
import sitkUtils
import slicer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"HRpQCTTools"/"DeepLearningSegmentationHRpQCT"))
sys.path.insert(0, str(ROOT/"IOTools"/"ScancoIO"))
sys.path.insert(0, str(ROOT/"IOTools"/"BatchProcessor"))


def wait(logic, seconds=180):
    deadline = time.monotonic()+seconds
    while logic.is_running() and time.monotonic()<deadline:
        slicer.app.processEvents()
        time.sleep(.01)
    assert not logic.is_running(), "Worker timed out"


def main():
    from DeepLearningSegmentationHRpQCT import (
        DeepLearningSegmentationHRpQCTLogic, DeepLearningSegmentationHRpQCTWidget,
        reference_geometry, voxel_digest, slicer_python,
    )
    from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import SceneCommand
    logic = DeepLearningSegmentationHRpQCTLogic()
    with tempfile.TemporaryDirectory(prefix="unet-slicer-check-") as temporary:
        directory = Path(temporary)
        records, messages = [], []
        def run(code, cancel=False, program=None):
            records.clear()
            logic._start(SceneCommand(program or slicer_python(), ["-c",code], directory),
                         messages.append, lambda *args: records.append(args))
            if cancel:
                assert logic.interrupt()
            wait(logic, 20)
            assert len(records)==1
            return records[0]
        assert run("print('final output', flush=True)")[0]==0
        assert "final output" in "".join(messages)
        assert run("raise SystemExit(7)")[0]==7
        crashed = run("import os,signal; os.kill(os.getpid(),signal.SIGKILL)")
        assert crashed[0]!=0 or crashed[1]==qt.QProcess.CrashExit
        assert run("import time; time.sleep(30)", cancel=True)[2]
        try:
            run("pass", program=str(directory/"missing-python"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Missing Python must fail startup")
        assert not logic.is_running()
        assert run("print('retry')")[0]==0

        # A completion callback may chain another job on the same logic object.
        chained=[]
        logic._workspace=tempfile.TemporaryDirectory(prefix="unet-first-")
        logic.context={"first":True}
        def start_second(*args):
            logic._workspace=tempfile.TemporaryDirectory(prefix="unet-second-")
            second_path=Path(logic._workspace.name)
            second_context={"second":True}
            logic.context=second_context
            def check_second(*args):
                assert second_path.exists() and logic.context is second_context
                chained.append(True)
            logic._start(SceneCommand(slicer_python(),["-c","import time; time.sleep(.05)"],directory),
                         on_finished=check_second)
        logic._start(SceneCommand(slicer_python(),["-c","pass"],directory),on_finished=start_second)
        wait(logic,20)
        assert chained==[True] and logic._workspace is None and logic.context is None

        full = np.zeros((3,7,9), dtype=np.uint8)
        full[:,1:6,1:8]=1
        trab = np.zeros_like(full)
        trab[:,2:5,2:7]=1
        cort=full-trab
        image=sitk.GetImageFromArray(full.astype(np.float32))
        image.SetSpacing((.061,)*3)
        image.SetOrigin((.061*20,-.061*40,.061*60))
        reference=sitkUtils.PushVolumeToSlicer(image, name="UNet import fixture")
        output=directory/"masks.npz"
        np.savez(output, full=full,trab=trab,cort=cort)
        count=slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSegmentationNode")
        node=logic.import_outputs(reference,output)
        assert node.GetAttribute("BoneImaging.MaskRoles") == "full,trab,cort"
        assert logic.import_outputs(reference,output) is node
        assert slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSegmentationNode")==count+1
        transform=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        node.SetAndObserveTransformNodeID(transform.GetID())
        try:
            logic.import_outputs(reference,output)
        except ValueError as error:
            assert "parent transform" in str(error)
        else:
            raise AssertionError("Transformed prior result accepted")
        node.SetAndObserveTransformNodeID(None)
        slicer.mrmlScene.RemoveNode(transform)
        for name,expected in (("Full mask",full),("Trabecular mask",trab),("Cortical mask",cort)):
            segment=node.GetSegmentation().GetSegmentIdBySegmentName(name)
            np.testing.assert_array_equal(slicer.util.arrayFromSegmentBinaryLabelmap(node,segment,reference),expected)
        geometry, digest=reference_geometry(reference),voxel_digest(reference)
        origin=reference.GetOrigin()
        reference.SetOrigin(8,9,10)
        try:
            logic.import_outputs(reference,output,expected_geometry=geometry)
        except ValueError as error:
            assert "geometry changed" in str(error)
        else:
            raise AssertionError("Changed geometry accepted")
        reference.SetOrigin(origin)
        slicer.util.arrayFromVolume(reference)[0,0,0]=42
        try:
            logic.import_outputs(reference,output,expected_digest=digest)
        except ValueError as error:
            assert "voxels changed" in str(error)
        else:
            raise AssertionError("Changed voxels accepted")
        np.savez(output,full=full,trab=trab,cort=trab)
        try:
            logic.import_outputs(reference,output)
        except ValueError:
            pass
        else:
            raise AssertionError("Overlapping masks accepted")
        assert slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSegmentationNode")==count+1
        # Rejection happens before scene mutation, preserving previous results.
        np.testing.assert_array_equal(slicer.util.arrayFromSegmentBinaryLabelmap(
            node,node.GetSegmentation().GetSegmentIdBySegmentName("Cortical mask"),reference),cort)

        # Exercise the real Batch Processor native AIM loader, not only NPZ import.
        # Slicer may have already discovered main's Batch Processor at startup.
        import importlib.util
        spec=importlib.util.spec_from_file_location("BatchProcessorFeatureSmoke",ROOT/"IOTools"/"BatchProcessor"/"BatchProcessor.py")
        batch=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=batch
        spec.loader.exec_module(batch)
        from ScancoIOLib import aim_io
        native_paths=[]
        for tag,array in (("full",full),("trab",trab),("cort",cort)):
            mask=sitk.GetImageFromArray(array)
            mask.CopyInformation(image)
            path=directory/f"fixture_desc-{tag}_mask.AIM"
            aim_io.write_aim(mask,path,mask=True)
            native_paths.append(str(path))
        shell=SimpleNamespace(_ensure_loaded_source_volume=lambda path,**kwargs: reference,
                              _append_log=messages.append)
        batch.BatchProcessorWidget._load_deep_learning_outputs(shell,{"image_path":"fixture.AIM"},native_paths)
        assert slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSegmentationNode")==count+1

        fixture=os.environ.get("HRPQCT_TEST_DENSITY")
        if fixture:
            with np.load(fixture,allow_pickle=False) as data:
                density=data["density"]
            middle=density.shape[0]//2
            actual=sitk.GetImageFromArray(density[middle:middle+3].astype(np.float32))
            actual.SetSpacing((.061,)*3)
            volume=sitkUtils.PushVolumeToSlicer(actual,name="Real radius UNet test")
            volume.SetAttribute("HRpQCT.AIMScaling","density")
            slicer.modules.deeplearningsegmentationhrpqct=SimpleNamespace(
                path=str(ROOT/"HRpQCTTools"/"DeepLearningSegmentationHRpQCT"/"DeepLearningSegmentationHRpQCT.py"))
            widget=DeepLearningSegmentationHRpQCTWidget()
            widget.inputVolumeSelector.setCurrentNode(volume)
            widget.deviceCombo.setCurrentIndex(1)
            widget._on_run()
            wait(widget.logic)
            assert str(widget.statusLabel.text).startswith("Loaded "), str(widget.statusLabel.text)+"\n"+str(widget.outputText.toPlainText())
            assert widget.logic.context is None and widget.logic._workspace is None
            widget.cleanup()
            print("ACTUAL_PUBLISHED_WEIGHTS_SLICER_CPU_PASSED",flush=True)
    print("DEEP_LEARNING_SLICER_SMOKE_PASSED",flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    slicer.app.exit(1)
else:
    slicer.app.exit(0)
