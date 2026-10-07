# Microarchitecture

Microarchitecture computes bone structure and density measurements from images, segmentations, and ROI masks. The Slicer module handles node selection, table display, map loading, and scene review. The `bone-microarchitecture` core package owns the measurement logic.

!!! video "Tutorial video"
    Watch the [microarchitecture tutorial](https://www.youtube.com/watch?v=wtnzl54njQM), or browse [all tutorial videos](../tutorials/index.md).

https://github.com/wallematthias/bone-microarchitecture

For prerequisites and step-by-step native or registered Functional Bone analysis,
start with the [Functional Bone guide](../tutorials/functional-bone.md).

## Required Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| XCT/BMD image | yes | grayscale or calibrated density image |
| Bone segmentation | yes | binary bone mask |
| Analysis ROI masks | yes | one or more reporting regions such as full, trabecular, or cortical |
| Common region | registered profiles only | native-space scan/FOV common region from Timelapsed |
| Large voidspace mask | both Functional Bone profiles | large-void map to exclude from the reporting region |

If a Slicer segmentation node contains multiple labels, select the exact segment in the adjacent segment dropdown.

## Scene Mode

Use scene mode for one loaded image and loaded masks.

1. Select the grayscale or BMD image.
2. Select the bone segmentation.
3. Select full, trabecular, cortical, or custom ROI masks.
4. Optionally select a common scan-region mask.
5. Run the analysis.
6. Review the loaded measurement table and maps.

## Batch Mode

Use `Bone Imaging > I/O > Batch Processor`.

Four profiles are exposed:

| Profile | Behavior |
| --- | --- |
| XtremeCT II | measures each session in native space |
| XtremeCT II - registered | applies each session's native common region during measurement |
| Native Functional Bone | measures each session's full scan after excluding large voidspace; no common region required |
| Registered Functional Bone | restricts each session to its native common region, then excludes large voidspace |

Native maps are reusable. If native maps already exist, the registered profile can reuse them and only recompute the common-region-restricted measurement table.

### IPL-aligned standard measurements

The updated core uses `ipl-aligned-v1`: cortical porosity counts cleaned,
slice-seeded intracortical pores rather than all cortical non-bone voxels;
cortical thickness is measured on the cortical compartment including its pores;
and trabecular number is **inverse mean ridge spacing**, not mean inverse spacing.
Spacing calculations retain native bone-phase context. Incomplete image-boundary
spheres are excluded without deleting a fixed number of end slices.

Scene analysis also computes native maps before applying an optional analysis
mask. The saved `Ct.Po.Mask` retains pore selection independently of the diameter
map, so reporting restrictions do not rerun the five-voxel cleanup. CSV JSON
sidecars and manifests identify the scientific method. Older method tables no
longer mark a batch row complete: rerun analysis to regenerate current results.
Ratios such as `Ct.Po` remain fractions; multiply by 100 when comparing with IPL
percentages. `Tb.N`'s other CSV distribution columns describe local inverse
spacing, whereas its Mean is inverse mean spacing.

These are closer definitions, **not validated numerical IPL equivalence**.
Native pore connectivity/percentage conventions, GOBJ boundaries and sphere
assignment still require identical-input IPL map comparisons. See the
[core method notes](https://github.com/wallematthias/bone-microarchitecture#ipl-aligned-measurement-definitions).

### Functional Bone reporting regions

Both Functional Bone profiles follow the same map reuse model: they do not create functional-bone maps or recompute thickness on clipped regions. They reuse or create native maps, then summarize over `full mask AND NOT large voidspace` (native) or `full mask AND common region AND NOT large voidspace` (registered), intersected with each reporting compartment.
For review/debugging, both profiles also write and load their effective analysis-region masks with distinct mode names.

Native Functional Bone requires the full-domain **Voidspace** large-void map for that session. It does not accept a Registered voidspace map because that map may exclude voids outside the common region. It runs one row per session.

Registered Functional Bone prefers the native Voidspace large-void map and falls back to the Registered voidspace map for that session. It additionally requires the matching native common region and groups sessions for one subject/VOI. Registration and common-region generation remain upstream steps.

Existing saved `functional-bone` jobs retain registered behavior and the existing `functional_bone_measurements` directory. The new native profile uses `functional-bone-native` and a separate directory; neither mode treats the other's measurements as completed outputs.

Imported compartment masks may be cropped differently from the grayscale image.
Batch processing places masks on the grayscale grid with nearest-neighbor
resampling using origin, spacing, and direction, rather than assuming equal array
dimensions. This reconciles grids in the same physical space; it does not register
different scans. Geometry-free arrays must already match the image dimensions.
The mask-grid policy is included in cache compatibility, invalidating older maps
computed under the equal-array assumption.

## Outputs

Outputs are written under `derivatives/Microarchitecture/`:

```text
derivatives/
  Microarchitecture/
    sub-001/
      ses-001/
        xct/
          maps/
          measurements/
          registered_measurements/
          functional_bone_measurements/
          functional_bone_native_measurements/
```

The Slicer load action should load:

- measurement table,
- available scalar maps,
- common-region segmentation for registered outputs.
- functional-bone analysis-region segmentation for functional bone outputs.

## Reported Measures

Common outputs include:

| Measure | Meaning |
| --- | --- |
| `Tt.BMD` | BMD over the total selected ROI |
| `Tb.BMD` | trabecular BMD |
| `Ct.BMD` | cortical BMD |
| `Tb.BV/TV` | trabecular bone volume fraction |
| `Tb.Th` | trabecular thickness |
| `Tb.Sp` | trabecular separation |
| `Tb.N` | trabecular number estimate |
| `Ct.Th` | cortical thickness |
| `Ct.Po` | cortical porosity fraction |
| `Ct.Po.Dm` | cortical pore diameter map summary |

Fractions are reported as fractions, not percentages.

`Ct.Po` is selected intracortical pore volume divided by cortical compartment
volume; it is no longer all cortical non-bone volume. `Ct.Po.Dm` summarizes a
thickness map of those selected pores. An IPL workflow that measures a separately
defined `PORE.AIM` mask may still select a different pore population. Matching compartment
masks alone does not establish equivalence; compare tissue segmentation, pore
definitions, reporting regions and fraction/percent units before pooling results.

## Citation

Credit Bone Imaging Toolbox and `bone-microarchitecture`. Cite field-specific HR-pQCT reporting guidelines or study-specific analysis definitions where required.
