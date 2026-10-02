"""Headless Slicer integration checks; set HRPQCT_TEST_DENSITY to an NPZ.

Without that optional real BMD fixture, only worker lifecycle and import checks
run. With it, three middle slices run the actual published weights in Slicer's
Python. This is an integration check, not cohort/model accuracy validation.
"""
from pathlib import Path
import json
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
sys.path.insert(0, str(ROOT/"HRpQCTTools"/"SegmentationHRpQCT"))
sys.path.insert(0, str(ROOT/"IOTools"/"ScancoIO"))
sys.path.insert(0, str(ROOT/"IOTools"/"BatchProcessor"))


def wait(logic, seconds=180):
    deadline = time.monotonic()+seconds
    while logic.is_running() and time.monotonic()<deadline:
        slicer.app.processEvents()
        time.sleep(.01)
    assert not logic.is_running(), "Worker timed out"


def main():
    from SlicerBoneImagingToolboxLib.deep_learning_contouring import (
        DeepLearningSegmentationHRpQCTLogic,
        reference_geometry, voxel_digest, slicer_python,
    )
    from SegmentationHRpQCT import SegmentationHRpQCTWidget
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

        # One Contouring UI, with independent tissue SEG and no hidden contour knobs.
        slicer.modules.segmentationhrpqct = SimpleNamespace(
            path=str(ROOT/"HRpQCTTools"/"SegmentationHRpQCT"/"SegmentationHRpQCT.py"))
        widget = SegmentationHRpQCTWidget()
        widget._set_combo_by_data(widget.contourProfileCombo, "xct2")
        widget.contourSettingsButton.collapsed = False
        widget.segmentationSettingsButton.collapsed = False
        widget.contourBackendCombo.setCurrentIndex(widget.contourBackendCombo.findData("unet"))
        assert widget._uses_unet() and widget._expertSections["Endosteal contour"].isHidden()
        assert widget._expertSections["Periosteal contour"].isHidden()
        assert not widget._expertSections["Bone segmentation"].isHidden() and not widget.unetDeviceCombo.isHidden(), (
            widget._expertSections["Bone segmentation"].isHidden(), widget.unetDeviceCombo.isHidden())
        widget.contourBackendCombo.setCurrentIndex(0)
        assert widget.unetDeviceCombo.isHidden() and not widget._expertSections["Endosteal contour"].isHidden()
        widget.contourBackendCombo.setCurrentIndex(widget.contourBackendCombo.findData("unet"))
        widget.volumeSelector.setCurrentNode(reference)
        reference.SetAttribute("HRpQCT.AIMScaling", "density")
        reference.SetAttribute("HRpQCT.AIMMetadata", json.dumps(
            {"element_size": [0.0607] * 3, "processing_log": {"Site": 20}}))
        real_run = widget._unet_logic.run_segmentation
        captured = []
        def fake_run(selected, device, **kwargs):
            captured.append(kwargs)
            np.savez(output, full=full, trab=trab, cort=cort,
                     **({"seg": cort} if kwargs["segmentation"] is not None else {}))
            widget._unet_logic.context = dict(reference=selected, output=output,
                geometry=reference_geometry(selected), digest=voxel_digest(selected))
            qt.QTimer.singleShot(0, lambda: kwargs["on_finished"](0, qt.QProcess.NormalExit, False))
        widget._unet_logic.run_segmentation = fake_run
        for method, roles in (("none", "full,trab,cort"), ("seg_gauss", "full,trab,cort,seg")):
            widget.segmentationMethodCombo.setCurrentIndex(widget._combo_value_index(widget.segmentationMethodCombo, method))
            widget._create_segmentation()
            assert not widget.createButton.enabled and widget.unetCancelButton.enabled
            slicer.app.processEvents()
            assert str(widget.messageLabel.text).startswith("Loaded "), str(widget.messageLabel.text)
            assert node.GetAttribute("BoneImaging.MaskRoles") == roles
            assert widget.createButton.enabled and not widget.unetCancelButton.enabled
        assert captured[0]["segmentation"] is None
        assert captured[1]["segmentation"]["method"] == "gauss"
        assert captured[1]["tissue_image"].GetSpacing() == reference.GetSpacing()
        # Malformed optional tissue masks are rejected before replacing valid results.
        np.savez(output, full=full, trab=trab, cort=cort, seg=np.ones_like(full))
        try:
            logic.import_outputs(reference, output)
        except ValueError as error:
            assert "Bone-tissue SEG" in str(error)
        else:
            raise AssertionError("Out-of-compartment tissue SEG accepted")
        assert node.GetAttribute("BoneImaging.MaskRoles") == "full,trab,cort,seg"
        widget._unet_logic.run_segmentation = real_run
        print("UNIFIED_CONTOURING_UI_PASSED", flush=True)

        fixture=os.environ.get("HRPQCT_TEST_DENSITY")
        if fixture:
            with np.load(fixture,allow_pickle=False) as data:
                density=data["density"]
            middle=density.shape[0]//2
            actual=sitk.GetImageFromArray(density[middle:middle+3].astype(np.float32))
            actual.SetSpacing((.061,)*3)
            volume=sitkUtils.PushVolumeToSlicer(actual,name="Real radius UNet test")
            volume.SetAttribute("HRpQCT.AIMScaling","density")
            widget.volumeSelector.setCurrentNode(volume)
            widget.unetDeviceCombo.setCurrentIndex(1)
            widget._create_segmentation()
            wait(widget._unet_logic)
            assert str(widget.messageLabel.text).startswith("Loaded "), str(widget.messageLabel.text)+"\n"+str(widget.unetOutputText.toPlainText())
            assert widget._unet_logic.context is None and widget._unet_logic._workspace is None
            result=next(n for n in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
                        if n.GetNodeReferenceID("DeepLearningSegmentationHRpQCT.InputVolume") == volume.GetID())
            assert result.GetAttribute("BoneImaging.MaskRoles") == "full,trab,cort,seg"
            print("ACTUAL_PUBLISHED_WEIGHTS_SLICER_CPU_PASSED",flush=True)
        widget.cleanup()
    print("DEEP_LEARNING_SLICER_SMOKE_PASSED",flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    slicer.app.exit(1)
else:
    slicer.app.exit(0)
