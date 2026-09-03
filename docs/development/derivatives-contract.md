# Derivatives Contract

Derivative-producing tools should write outputs under `derivatives/<Family>/` and record produced artifacts in `manifest.json`.

## Core Principles

- Do not write generated outputs directly into raw data folders.
- Prefer manifest records over filename inference.
- Store portable relative paths when possible.
- Keep derivative families separate so tools can reuse each other's outputs without folder-name guessing.
- Treat a common region as scan/FOV support, not as a biological mask.
- Prefer imported Scanco/IPL/manual contours over generated contours when both exist.

## Common Families

- `ImportedContours`
- `BoneContours`
- `Registration`
- `CommonRegion`
- `Timelapse`
- `Microarchitecture`
- `PlateRodMorphometry`
- `FEA`
- `Mechanoregulation`
- `MotionScore`

Consumers should report missing prerequisites before launching a batch job.
