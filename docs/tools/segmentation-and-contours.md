# Contouring

Contouring creates bone segmentations, periosteal and endosteal contours, trabecular/cortical ROI masks, and material label maps. It is the preferred place to prepare masks before Timelapsed Remodelling, Microarchitecture, Plate/Rod Morphometry, and ParOsol-FEA.

!!! video "Tutorial video"
    Watch the [contouring tutorial](https://www.youtube.com/watch?v=bJsD-42t7hk), or browse [all tutorial videos](../tutorials/index.md).

Core contouring logic lives in:

https://github.com/wallematthias/bone-contouring

## When To Use

Use this tool when you need to:

- create a binary bone segmentation,
- create full, trabecular, and cortical ROI masks,
- generate material labels for FEA,
- derive one compartment mask from two existing masks,
- export reusable contouring profiles.

## Inputs And Outputs

| Input | Output |
| --- | --- |
| XCT image | bone segmentation |
| XCT image plus profile | full/trab/cort ROI masks |
| bone segmentation plus ROI masks | FEA material label map |
| any two of full/trab/cort | missing compartment mask |

## Scene Workflow

1. Select the input image.
2. Select a shipped or custom profile.
3. Adjust segmentation, periosteal contour, or endosteal contour settings if needed.
4. Export a custom profile if the settings should be reused.
5. Click `Generate`.
6. Review the loaded segmentation and masks in Slicer.

Expert settings are grouped by algorithm. Gaussian settings appear for Gaussian segmentation, Laplace-Hamming settings appear for Laplace-Hamming segmentation, and geodesic settings appear only when the geodesic periosteal contour is selected.

## Batch Workflow

Use `Bone Imaging > I/O > Batch Processor` for cohort contouring. Each row corresponds to one image. Batch contouring writes generated masks under `derivatives/BoneContours/` and records how they were generated in sidecars and manifests.

## Profiles

Shipped profiles provide scanner/site defaults. Custom profiles can be exported from the scene UI and reused later.

Common profile families include:

- XtremeCT I profiles,
- XtremeCT II profiles,
- optional geodesic periosteal-contour profiles,
- user-defined custom profiles.

## Advanced Settings

Advanced settings expose the algorithm choices that are normally fixed by a profile. Changing these settings is useful for method development, protocol validation, or creating a custom profile.

| Setting group | What it controls |
| --- | --- |
| Gaussian segmentation | Gaussian smoothing and threshold settings commonly used by XtremeCT II style workflows. |
| Laplace-Hamming segmentation | Laplace-Hamming filtering and threshold settings commonly used by XtremeCT I style workflows. |
| adaptive local thresholding | Local thresholding behavior for datasets where a single global threshold is not appropriate. |
| Periosteal contour | Outer contour extraction. The standard contour is profile controlled; geodesic contour settings appear when a geodesic contour profile is selected. |
| Endosteal contour | Inner contour extraction and trabecular/cortical separation behavior. |
| Mask and material outputs | Generation of full, trabecular, cortical, and FEA material label outputs. |

To reuse edited settings, enter a workflow display name and export a custom profile. Custom profiles are stored outside the shipped package profiles and are discovered by the Batch Processor.

## Output Roles

| Role | Meaning |
| --- | --- |
| `seg` | binary bone segmentation |
| `full` | periosteal/full ROI mask |
| `trab` | trabecular ROI mask |
| `cort` | cortical ROI mask |
| `fea-materials` | material label map for ParOsol-FEA |

## Attribution

The contouring workflow builds on profile definitions and processing conventions developed with the Galateia Kazakia lab.

Credit Bone Imaging Toolbox and `bone-contouring` for generated masks. Cite study-specific segmentation or contouring definitions required by the analysis protocol or target journal.
