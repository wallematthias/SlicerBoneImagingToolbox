# Batch Processor

The Batch Processor runs cohort jobs from a normalized dataset root. It is the preferred batch entry point for reusable workflows because it uses the shared derivative discovery contract and writes outputs into `derivatives/`.

## Workflow

1. Normalize raw data with the Dataset Naming Helper when filenames are inconsistent.
2. Select the dataset root.
3. Select a tool and profile.
4. Review the discovered rows and missing prerequisites.
5. Use row-level `Run` / `Load` actions or `Run all`.

Rows are tool-specific. Single-session tools run one session per row. Registered or longitudinal tools group the sessions needed for one subject and VOI.

## Outputs

Outputs are written as derivative artifacts with portable relative paths and manifest records. Tools should reuse compatible existing artifacts when `Skip existing` is enabled.

## Citation

Cite the analysis workflow selected in the Batch Processor. For Timelapsed remodelling and mechanoregulation, cite [Walle et al., Bone 2023](https://doi.org/10.1016/j.bone.2023.116780). For multistack registration, cite [Whittier et al., Bone 2023](https://doi.org/10.1016/j.bone.2023.116893).
