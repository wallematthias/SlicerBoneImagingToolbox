# Microarchitecture

Microarchitecture computes bone structure and density measurements from images, segmentations, and ROI masks. The Slicer module handles node selection, table display, map loading, and scene review. The `bone-microarchitecture` core package owns the measurement logic.

!!! video "Tutorial video"
    Watch the [microarchitecture tutorial](https://www.youtube.com/watch?v=wtnzl54njQM), or browse [all tutorial videos](../tutorials/index.md).

https://github.com/wallematthias/bone-microarchitecture

## Required Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| XCT/BMD image | yes | grayscale or calibrated density image |
| Bone segmentation | yes | binary bone mask |
| Analysis ROI masks | yes | one or more reporting regions such as full, trabecular, or cortical |
| Common region | registered profile only | native-space scan/FOV common region from Timelapsed |
| Large voidspace mask | functional bone profile only | native-space registered voidspace mask to exclude from the common region |

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

Three common profiles are exposed:

| Profile | Behavior |
| --- | --- |
| Microarchitecture | measures each session in native space |
| Registered Microarchitecture | applies each session's native common region during measurement |
| Functional bone | applies each session's native common region after removing the registered large voidspace mask |

Native maps are reusable. If native maps already exist, the registered profile can reuse them and only recompute the common-region-restricted measurement table.
Functional bone follows the same map reuse model: it does not create functional-bone maps. It reuses or creates the native maps, then writes measurements over `common region AND NOT large voidspace`.
For review/debugging, the functional bone profile also writes and loads the effective analysis-region mask.

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

## Citation

Credit Bone Imaging Toolbox and `bone-microarchitecture`. Cite field-specific HR-pQCT reporting guidelines or study-specific analysis definitions where required.
