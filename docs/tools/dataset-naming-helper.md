# Dataset Naming Helper

Dataset Naming Helper reviews raw Scanco-style datasets and prepares the normalized layout used by the Batch Processor. Use it when filenames are inconsistent, contain vendor suffixes such as `;1`, or do not clearly encode subject, session, VOI, stack, and mask role.

## What It Produces

The helper can:

- analyze files without modifying them,
- let you correct parsed fields in a review table,
- export a rename plan,
- rename files into the toolbox dataset layout,
- write JSON sidecars with non-identifying metadata,
- keep a manifest that can undo a rename.

The helper should not expose private subject names in shareable outputs. Identifying source metadata should either be stripped or kept in a private sidecar that is not part of an exported shareable dataset.

## Basic Workflow

1. Select a dataset root.
2. Click `Analyze`.
3. Review subject, session, VOI, stack, modality, and file role.
4. Edit any uncertain rows directly in the table.
5. Export the plan if you want to review it outside Slicer.
6. Click `Rename` when the plan is correct.
7. Use `Undo rename` if you need to restore the original file names from the manifest.

## Before And After

A loosely named input set:

```text
sample_0001_RL_Y00.AIM
sample_0001_RL_Y00_TRAB_MASK.AIM
sample_0001_RL_Y00_CORT_MASK.AIM
sample_0001_RL_Y04.AIM;1
```

becomes:

```text
sub-001/
  ses-001/
    xct/
      sub-001_ses-001_voi-radiusleft_xct.AIM
  ses-002/
    xct/
      sub-001_ses-002_voi-radiusleft_xct.AIM
derivatives/
  ImportedContours/
    sub-001/
      ses-001/
        xct/
          sub-001_ses-001_voi-radiusleft_desc-trab_mask.AIM
          sub-001_ses-001_voi-radiusleft_desc-cort_mask.AIM
```

The exact session labels are assigned in chronological or parsed order. If the automatic session order is wrong, edit it before renaming.

## VOI Names

The `voi-*` token should preserve laterality and anatomical region. Prefer compact names such as:

- `radiusleft`
- `radiusright`
- `tibialeft`
- `tibiaright`
- `kneeleft`
- `kneeright`
- `tibiaproxleft`

Do not collapse left and right scans into a single `radius` or `tibia` VOI when both sides may appear in a dataset.

## Imported Contours

Existing Scanco/IPL/manual masks are stored under `derivatives/ImportedContours/`. These are preferred over generated `BoneContours` because they may have been manually reviewed.

Common recognized roles include:

| Source token | Normalized role |
| --- | --- |
| `TRAB_MASK`, `TRAB` | `trab` |
| `CORT_MASK`, `CORT`, `CRTX` | `cort` |
| `FULL_MASK`, `BLCK` | `full` |
| `REGMASK` | `regmask` |
| `ROI1`, `ROI2`, `MASK1` | generic analysis ROI |

## Screenshot To Add

Add one screenshot of the review table after `Analyze`, with only generic sample IDs visible.

## Citation

Credit Bone Imaging Toolbox when the helper is used for dataset preparation.
