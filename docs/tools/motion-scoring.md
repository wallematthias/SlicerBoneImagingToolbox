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
3. Open `Bone Imaging > Setup > Toolbox Setup`.
4. Install or update the MotionScore runtime if needed.
5. Open `Bone Imaging > Microstructural Analysis > Motion Scoring`.

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

The module focuses on prediction and review:

- loaded scan review for a single image,
- batch prediction from a dataset folder,
- manual correction of predicted grades,
- export of reviewed results.

## Review Workflow

Motion Scoring is intentionally kept as its own review-focused module rather than being folded into the central Batch Processor. It manages scan loading, prediction review, manual correction, model retraining, and export of final grades.

The review scope controls which scans are shown to the reader after prediction. Common scopes are all scans, only pending scans, or low-confidence scans. The confidence threshold defines which model predictions are considered low confidence and should be surfaced for manual review.

During manual review, selecting a grade records the reader's decision and advances to the next scan in the active review scope. The reviewer tag is stored with the manual grade so several readers can review the same dataset independently.

Optional blind review hides the model grade and prediction plot until the reader submits a grade. This is useful when model-assisted triage should not bias the reader.

The retraining controls use previously reviewed datasets to prepare a retrain manifest, sample slices per scan, and fine-tune either the classifier or the full model depending on the selected training settings.

## Scene Mode

Use loaded scan review when a scan is already open in Slicer. Select the scalar volume, run the model, and review the generated motion-grading image and grade.

## Batch Mode

1. Select the dataset root or scan table.
2. Select the model profile.
3. Run prediction.
4. Review predicted scores in the review table.
5. Correct scores manually where needed.
6. Export the review table.

When valid model weights are already present in `Local models folder`, the module uses those local files and skips the GitHub download. This is the preferred setup for managed workstations or offline installations.

Motion Scoring can run on a normalized dataset, but it should ignore derivative artifacts and score only source scan images. Results are written under `derivatives/MotionScore/` so prediction, review, and export records stay with the dataset.

Current default model catalog:

https://github.com/wallematthias/MotionScoreHRpQCT/releases/latest/download/model_catalog.json

## Attribution

For Motion Scoring, cite:

Walle M, Eggemann D, Atkins PR, Kendall JJ, Stock K, Müller R, Collins CJ. Motion grading of high-resolution quantitative computed tomography supported by deep convolutional neural networks. *Bone*. 2023;166:116607. doi: 10.1016/j.bone.2022.116607.

## Screenshot To Add

Add one generic screenshot showing the review table and one motion-grading preview image without identifying overlays.
