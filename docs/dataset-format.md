# Dataset Format

Batch workflows are most stable when data are normalized into the toolbox dataset layout before analysis.

## Recommended Layout

```text
dataset-root/
  sub-001/
    ses-001/
      xct/
        sub-001_ses-001_voi-radiusleft_xct.AIM
        sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM
        sub-001_ses-001_voi-radiusleft_desc-trab_mask.AIM
        sub-001_ses-001_voi-radiusleft_desc-cort_mask.AIM
    ses-002/
      xct/
        sub-001_ses-002_voi-radiusleft_xct.AIM
  derivatives/
    ImportedContours/
    BoneContours/
    Registration/
    CommonRegion/
    Timelapse/
    Microarchitecture/
    PlateRodMorphometry/
    FEA/
    Mechanoregulation/
```

The `voi-*` token identifies the volume of interest. Examples include `radiusleft`, `radiusright`, `tibialeft`, `tibiaright`, `kneeleft`, and `kneeright`.

## Discovery Rules

The Dataset Naming Helper attempts to recover subject, session, VOI, stack, and mask role information from:

- normalized filenames,
- common Scanco/IPL filename conventions,
- AIM header metadata when available,
- user edits in the review table.

If automatic discovery is wrong, edit the review table before renaming or exporting a plan.

## Imported Contours

Scanco/IPL/manual masks that arrive with a dataset should be stored as `ImportedContours` derivatives. Analysis tools prefer `ImportedContours` over generated `BoneContours` when both are available because imported masks are often manually reviewed or corrected.

## Generated Outputs

Generated outputs are written under `derivatives/` and recorded in manifest files using portable relative paths where possible. Avoid moving individual derivative files by hand; move the dataset root as a whole.
