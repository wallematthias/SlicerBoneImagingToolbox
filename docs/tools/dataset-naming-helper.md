# Dataset Naming Helper

Dataset Naming Helper reviews raw Scanco-style datasets and prepares the normalized layout used by the Batch Processor. It is meant for lab datasets with mixed naming conventions, vendor version suffixes, or incomplete identifiers.

## Workflow

1. Select a dataset root.
2. Analyze the discovered files.
3. Review subject, session, VOI, stack, modality, and mask-role assignments.
4. Edit uncertain rows in the review table.
5. Export a plan, rename files, or undo a previous rename from the generated manifest.

The helper keeps a manifest of changes so renames can be reversed. Shareable exports can omit private identity sidecars.

## Dataset Layout

Normalized raw files live under `sub-*/ses-*/xct/` and use `voi-*` tokens for scan regions such as `radiusleft`, `radiusright`, `tibialeft`, `tibiaright`, or `kneeleft`.

Imported Scanco/IPL masks are placed in `derivatives/ImportedContours/`. Generated toolbox contours are placed in `derivatives/BoneContours/`.

## Citation

Credit Bone Imaging Toolbox when the helper is used for dataset preparation.
