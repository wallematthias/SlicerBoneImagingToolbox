# Motion Scoring

`Motion Scoring` is the Slicer front end for HR-pQCT motion grading. It runs MotionScore predictions, supports manual review, and exports review tables.

## When To Use

Use this tool when you want to:

- predict HR-pQCT scan motion grades,
- resume interrupted prediction runs,
- skip scans that already have predictions,
- review and correct motion grades in Slicer,
- export a review table for downstream analysis.

## Setup

Motion Scoring requires PyTorch for model inference.

1. Install the `PyTorch` extension from Slicer's Extension Manager.
2. Restart Slicer.
3. Open `Bone Imaging > HR-pQCT > Motion Scoring`.
4. Use the module's install/update controls to install the MotionScore runtime if needed.

The Slicer module is the GUI layer. Core model and inference logic lives in:

https://github.com/wallematthias/MotionScoreHRpQCT

## Model Bundle Setup

The module no longer depends on a hosted license request service. The recommended model distribution is a downloadable model bundle.

The automatic `Install / Download Models` button downloads the default model from the MotionScoreHRpQCT release catalog. If that automatic download is unavailable, install the same model manually:

1. Download the default MotionScore base model bundle:
   - Windows-friendly zip: https://github.com/wallematthias/MotionScoreHRpQCT/releases/download/v2.5.4/motionscore-base-v1.zip
   - macOS/Linux tarball: https://github.com/wallematthias/MotionScoreHRpQCT/releases/download/v2.5.4/motionscore-base-v1.tar.gz
2. In Slicer, open `Motion Scoring` and note the path shown in `Local models folder`.
3. Extract the bundle into that folder.
4. Confirm the extracted files include model weights named like `DNN_*.pt` or `DNN_*.h5`.
5. Restart Slicer or reopen the module.
6. Select `Base v1 (base-v1)` from `Model Profile`.

Recommended layout:

```text
<Local models folder>/
  model_registry.json
  base-v1/
    DNN_*.pt
```

The module also accepts a flat fallback layout:

```text
<Local models folder>/
  model_registry.json
  DNN_*.pt
```

The `base-v1` subfolder layout is preferred because it keeps model versions separated.

## Basic Workflow

The module has two modes:

- `Scene` scores or initializes review for one scalar volume already loaded in Slicer.
- `Batch` discovers AIM scans from a dataset folder, runs/resumes prediction, and manages cohort review.

## Scene Mode

Use `Scene` when a scan is already loaded in Slicer. Select the scalar volume, enter scan/subject/site/session labels, choose a model profile or manual-review mode, and run. The module exports the volume to a scoped scene-run handoff file and runs MotionScore in a background process.

## Batch Mode

1. Select the folder or table of scans to score.
2. Select the model profile.
3. Run prediction.
4. Review predicted scores in the review table.
5. Correct scores manually where needed.
6. Export the review table.

When valid model weights are already present in `Local models folder`, the module uses those local files and skips the GitHub download. This is the preferred setup for managed workstations or offline installations.

Current default model catalog:

https://github.com/wallematthias/MotionScoreHRpQCT/releases/latest/download/model_catalog.json

## Attribution

For Motion Scoring, cite:

Walle M, Eggemann D, Atkins PR, Kendall JJ, Stock K, Müller R, Collins CJ. Motion grading of high-resolution quantitative computed tomography supported by deep convolutional neural networks. *Bone*. 2023;166:116607. doi: 10.1016/j.bone.2022.116607.
