# Deep Learning Segmentation

Runs the published embedding U-Net with fixed scientific defaults. The GPL-licensed
model and morphology live in `bone-contouring[unet]`; this Toolbox
only owns execution and scene import. No Bonelab or vtkbone is required.

## Setup

Install **Bone Contouring** through Toolbox Setup, which includes the U-Net extra.
Requires `bone-contouring>=0.3.0` and `bone-imaging-derivatives>=0.1.6` from PyPI.
Setup can also use sibling local checkouts for development.
Restart Slicer after dependency installation.

Runtime dependencies are PyTorch, NumPy, scikit-image and aimio-py. The first run
downloads the pinned `radius_tibia_final` weights from Zenodo and checks their SHA256;
later runs reuse the cache. Scene and local batch share Slicer's application-data
`HRpQCTSegmentation/models/radius_tibia_final.pth`, alongside MotionScore's model
folder, outside the extension and datasets. Extension/package updates do not remove
it. Set `HRPQCT_SEGMENTATION_MODEL_DIR` before launching Slicer to use another
persistent folder. For offline use, set `HRPQCT_SEGMENTATION_WEIGHTS` to the
same published `.pth` file before launching Slicer; its checksum must match.

## Scene mode

1. Load the scan through Scanco I/O with **Density/BMD** scaling.
2. Open **Deep Learning Segmentation**, select the BMD volume and device.
3. Click **Generate**, then review the full, trabecular and cortical compartments.

`auto` tries CUDA, Apple MPS, then CPU. Explicit CUDA/MPS requests fail clearly
when unavailable; they never silently fall back. MPS requires an Apple Silicon
Python/PyTorch runtime; Intel Slicer workers cannot use it even on an Apple Silicon
computer. There are no model, threshold,
smoothing, Python executable or repository selectors.

The worker uses a frozen snapshot of the loaded scene voxels, not the original file.
Transformed, HU, native-scaled and unknown-unit volumes are rejected. If the selected
volume is removed, moved or edited during inference, results are not imported.
Execution is asynchronous with progress and cancellation. Temporary snapshots and
outputs are cleaned up. Repeating a run replaces this scan's existing U-Net result
after output validation, instead of adding duplicate segmentation nodes.

The full mask is the exact disjoint union of trabecular and cortical masks and is
initially hidden. Empty, overlapping or wrong-size masks are rejected. These are
**compartment masks**, not a thresholded bone-tissue `SEG`. No extra Toolbox peel or
smoothing is applied; the published post-processing includes its own 8-voxel shell.

## Batch and SSH

In **Batch Processor**, choose **U-Net Bone Contours**, a normalized dataset
root and device. Only raw AIM images are selected. Native AIM outputs go under
`derivatives/BoneContours/sub-<id>/ses-<id>/xct/`, using the standard
`_desc-full_mask.AIM`, `_desc-trab_mask.AIM`, `_desc-cort_mask.AIM` names, individual
sidecars, and the shared portable BoneContours manifest. Model, resolved device,
weights checksum and scientific defaults remain in the provenance. Load imports
all three roles into one segmentation node, as in scene mode.

Existing outputs are never overwritten. A complete published contour set is
loadable; partial files block U-Net reruns to prevent mixed results. Complete cases
whose manifest publication failed offer **Publish**, which retries only the
manifest without inference or mask rewriting. Incomplete cases are withheld from
downstream discovery until all masks, sidecars and the completion marker exist.
Virtual stack views must first be exported as physical AIM stacks. Downstream tools
discover these compartment roles through the usual BoneContours contract; imported
and IPL contours keep their existing priority. Run **Bone Contouring** afterwards
to generate missing tissue SEG and FEA material labels while reusing the U-Net
compartments, without recomputing or overwriting their files (unless explicitly
using the other tool's force option).

The same core runs without Slicer on another computer, including over SSH:

```bash
python -m pip install --upgrade 'bone-contouring[unet]>=0.3.0'
bone-contouring unet /data/raw --output /data/unet-results --device auto
ssh host 'bone-contouring unet /data/raw --output /data/unet-results --device cuda'
```

The remote host needs the core installed and access to scans and weights; a local
installation does not transfer those automatically. No live remote host was tested.
Batch Processor's Server/SLURM submission is not integrated for this tool yet;
use the core CLI over SSH instead. The server CLI defaults to its per-user cache,
or uses `HRPQCT_SEGMENTATION_MODEL_DIR` for a pre-provisioned model folder.
The CLI accepts files or directories, preserves relative directory structure and
reuses one loaded model per batch. Outputs use the `_desc-<role>_mask.AIM` names,
individual sidecars and a `_UNET.json` completion/provenance marker. AIM mask values
are native 0/127; dimensions/position/offset are preserved, with the AIM writer's
nanometre spacing precision. Slicer checks the integer lattice and permits only
this sub-nanometre spacing quantization, without voxel interpolation.
Standalone CLI results have per-case provenance; the Toolbox additionally writes
its shared dataset manifest. Scene results are temporary until saved through Slicer.

To install weights in advance rather than during the first inference, run this in
Slicer's Python console (after installing the core):

```python
import os, qt
from SlicerBoneImagingToolboxLib.deep_learning_segmentation_scene import model_cache_directory
from bone_contouring.unet.weights import weights_path
os.environ.setdefault("HRPQCT_SEGMENTATION_MODEL_DIR", str(model_cache_directory(
    qt.QStandardPaths.writableLocation(qt.QStandardPaths.AppDataLocation))))
print(weights_path())
```

The same published file can be copied to an offline machine. Its pinned checksum
is verified on reuse, and PyTorch loads it with `weights_only=True`.

## Applicability and developer checks

The pinned model targets 61 µm radius/tibia HR-pQCT. XCT-I and knee use are not
validated. Running successfully does not establish accuracy or equivalence to IPL.
The original duplicated upper neighbor in the five-slice input stack is retained
for compatibility with the published implementation and weights.

Run `python -m pytest tests/test_deep_learning_segmentation_scene.py -q` for wrapper
contracts. Run `Slicer --no-splash --no-main-window --python-script
scripts/test_deep_learning_scene_in_slicer.py` for real worker lifecycle and MRML
import checks. Optionally set `HRPQCT_TEST_DENSITY` to an NPZ with a calibrated
`density` array in ZYX order to also test published-weight inference on three slices.
This integration check is not model-accuracy validation.

## Attribution

The original segmentation method and published trained model are the work of
Nathan J. Neeteson, Bryce A. Besler, Danielle E. Whittier and Steven K. Boyd
(Bone Imaging Laboratory, University of Calgary). This module provides a Slicer
wrapper and minimal inference integration for their method and model.
Please cite the original publication when using this module:

Neeteson NJ, Besler BA, Whittier DE, Boyd SK. Automatic segmentation of trabecular and
cortical compartments in HR-pQCT images using an embedding-predicting U-Net and
morphological post-processing. *Scientific Reports*. 2023;13:252.
doi: [10.1038/s41598-022-27350-0](https://doi.org/10.1038/s41598-022-27350-0).
Weights: [Zenodo record](https://zenodo.org/records/14755838).
Original code: [Bonelab/HR-pQCT-Segmentation](https://github.com/Bonelab/HR-pQCT-Segmentation).
The scientific backend and published weights retain their upstream GPL-3.0 license;
the Toolbox wrapper does not replace that license.
