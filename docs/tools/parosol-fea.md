# ParOsol-FEA

ParOsol-FEA prepares and runs finite-element workflows from material label maps. It supports XtremeCT I/II-style profiles and load-history profiles through `parosol-py`.

## Inputs

- A material label map, usually generated from bone segmentation and trabecular/cortical ROI masks.
- A selected FEA profile.

Batch mode is available through the Batch Processor. Each row corresponds to a discovered material label map and profile-specific output family.

## Outputs

The workflow writes FEA derivatives including mechanical summaries, SED fields, and load-history scale factors when relevant. Loading results brings the SED field and selected summary values back into Slicer.

## Citation

Credit Bone Imaging Toolbox and `parosol-py`. Cite the study-specific FEA method and validation source used for the selected profile.
