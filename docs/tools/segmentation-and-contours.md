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
2. Choose **Profile**: XCTI, XCTII, Custom, or a saved profile. There is no separate scanner selector and no resolution-based scanner detection. Leave **Site preset** on Auto (AIM header), or choose the site explicitly.
3. Choose **Contouring**: dual threshold (standard), geodesic with standard inner contour, published U-Net (Neeteson et al.), or None.
4. Independently choose **Tissue segmentation**: Gaussian, Laplace-Hamming, adaptive, or None. Selecting XCTI initializes Laplace-Hamming; selecting XCTII initializes Gaussian. You can override the method afterward. Custom profiles retain the current method, and saved profiles restore their saved method.
5. Adjust **Contouring advanced settings** directly below Contouring, or **Tissue segmentation advanced settings** directly below Tissue segmentation. Load/save named profiles under **Custom profiles**.
6. Click `Generate`.
7. Review the loaded segmentation and masks in Slicer.

Scanner selection is always manual and never changes with voxel spacing or a new
input volume. Auto site detection prefers the AIM header stored by Scanco I/O,
then uses recognized volume/file names. If Auto cannot resolve the site, Generate
asks you to select it explicitly. **Custom** retains edited settings and initially
sets Site to **None (custom settings)**, without applying scanner/site calibration.
For example, micro-CT users can enter thresholds in their image's units and save
their own named profile for scene or batch use; there is no built-in micro-CT profile.
Selecting **None (custom settings)** under Site likewise retains edited values.
**Contouring: None** generates only tissue SEG, without inventing compartment masks.
**Tissue segmentation: None** generates contours only; both cannot be None.

For U-Net, select XCTII with radius/tibia, a Density/BMD volume, and Device under
**Contouring advanced settings**; Device is its only runtime control. Published contour defaults are
fixed, with no extra Toolbox smoothing or peel. Tissue settings
control only the independent SEG; select None for compartments only.
Both inference and tissue SEG run asynchronously with progress and cancellation,
and all outputs load into one segmentation node. See
[Published U-Net Contouring](deep-learning-segmentation.md) for weights, applicability,
batch/SSH use and the required Neeteson et al. citation. The separate DL module
has been folded into Contouring.

Expert settings are grouped by algorithm. Gaussian settings appear for Gaussian segmentation, Laplace-Hamming settings appear for Laplace-Hamming segmentation, and geodesic settings appear only when the geodesic periosteal contour is selected.

Gaussian tissue segmentation defaults to **sigma 1.2 voxels**, **trabecular
threshold 320** and **cortical threshold 450 mg HA/cm³**. Sigma is scaled by the
smallest voxel spacing. This is one filter of the original density image;
contour prefilters and boundary smoothing do not feed a second smoothed image
into tissue segmentation. These defaults apply in scene mode (including optional
tissue SEG after U-Net) and standard batch profiles. Saved custom values remain
unchanged, except the obsolete `keep_largest_component` tissue setting is
ignored. Tissue SEG retains disconnected bone; its largest-component control
has been removed. The separate minimum-size noise filter remains. FEA can
still select its largest connected component during model preparation without
changing the reusable tissue segmentation or its material labelmap.

The current toolbox requires `bone-contouring[unet]>=0.3.5`. Update the toolbox
and the Bone Contouring runtime package through Setup, then restart Slicer.

Standard uses the same topology-first compartment contouring for XCTI and XCTII,
including radius, tibia, and knee. Standard XCTI uses periosteal threshold
250 for radius/tibia and 150 for knee in both scene and batch defaults (calibrated
mg HA/cm³, not the native-gray Laplace–Hamming tissue threshold). Saved custom
profiles retain their explicit values. XCTII radius/tibia use outer threshold 320 and
inner threshold 380 in calibrated mg HA/cm³; other presets retain their existing
density settings. All envelopes are independent of tissue-segmentation thresholds.
The fragile pre-fill opening is removed. The standard outer contour dilates in XY,
fills the dilated shell, then erodes with the same radius. Filling **before erosion**
prevents a narrow sealed bridge from reopening before the interior is filled.
There is one shared Standard/Dual Threshold implementation; the package's
`stable_3d` name is an alias, not another standard method. Modest 3D signed-distance smoothing uses XYZ sigmas
`(0.03, 0.03, 0.06)` mm. This is not full-3D morphological closing or a literal
Buie/native IPL reproduction. Native-mask comparisons currently cover XCTII
radius/tibia only. Largest-component selection still assumes one target bone;
this is not a multi-bone knee segmentation method.
Before selecting the marrow component, the standard inner method restricts its
low-density seed to full eroded in XY by `Peel` (default 3 voxels). This excludes
the peripheral low-density layer, which can otherwise connect through cortical
pores and swallow dense cortex during dilation and filling.
Final axial filling after smoothing prevents reintroduced enclosed envelope
holes; the final trabecular ROI is constrained to full eroded in XY by `Peel`
(default 3 voxels for all standard profiles) before cortical subtraction.
The same minimum cortical compartment rim is reapplied after smoothing/filling and
does not peel Z end slices. It is not measured cortical bone thickness or an
original Buie requirement; an explicit peel of 0 disables the constraint.

Ignored legacy controls are hidden for this standard; the effective Peel control
is visible in Endosteal expert settings. Physical smoothing and
the endosteal footprint are recorded in exported recipes (`(31, 31, 1)` dimensions
for XCTII radius/tibia; existing Buie defaults `(10, 10, 1)` for XCTI/knee),
and custom recipe overrides are preserved. Algorithm revision, effective
parameters, and advisory contour QA are saved on the generated segmentation
node. Existing saved masks are not automatically changed. Inspect each result:
the method cannot guarantee anatomical correctness or voxel-identical Scanco
contours.

## Batch Workflow

Use `Bone Imaging > I/O > Batch Processor` for cohort contouring. Each row corresponds to one image. Batch contouring writes generated masks under `derivatives/BoneContours/` and records how they were generated in sidecars and manifests. Select **Bone Contouring → U-Net (Neeteson et al.)** for published compartment masks. Existing standard and custom batch profiles remain available; batch does not duplicate the scene's method selectors.

## Profiles

Scene scanner/site presets and algorithm choices are independent, avoiding a long
list of combinations. Saved custom scene profiles retain edited numerical settings
and automatic site selection. Existing saved profiles still load.

Batch retains its shipped scanner profiles and standard custom profiles. U-Net and
segmentation-only scene recipes are scene-only; batch U-Net uses its fixed profile.

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

To reuse edited settings, enter a workflow display name and export a custom profile. Custom profiles are stored outside the shipped package profiles; standard contour recipes are also discovered by the Batch Processor.

## Output Roles

| Role | Meaning |
| --- | --- |
| `seg` | binary bone segmentation |
| `full` | periosteal/full ROI mask |
| `trab` | trabecular ROI mask |
| `cort` | cortical ROI mask |
| `fea-input` | material label map for ParOsol-FEA |

## Attribution

The contouring workflow builds on profile definitions and processing conventions developed with the Galateia Kazakia lab.

Credit Bone Imaging Toolbox and `bone-contouring` for generated masks. Cite study-specific segmentation or contouring definitions required by the analysis protocol or target journal.

For the published U-Net, credit Nathan J. Neeteson, Bryce A. Besler, Danielle E.
Whittier and Steven K. Boyd, and cite their
[Scientific Reports paper (2023)](https://doi.org/10.1038/s41598-022-27350-0).
Their embedding-predicting U-Net is not nnU-Net; its scientific code and weights
retain GPL-3.0 terms in the core package, independently of the MIT Slicer wrapper.
