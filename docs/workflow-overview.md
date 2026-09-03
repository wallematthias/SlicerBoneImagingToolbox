# Workflow Overview

Most workflows follow the same pattern:

```text
raw scans
  -> dataset naming
  -> imported or generated contours
  -> registration and common region when longitudinal analysis is needed
  -> analysis maps and measurements
  -> downstream combined analyses
```

## Common Processing Order

| Step | Tool | Main output |
| --- | --- | --- |
| 1 | Dataset Naming Helper | normalized `sub-* / ses-* / xct/` dataset |
| 2 | Scanco I/O | loaded AIM/ISQ/SCV/GOBJ nodes when working in scene mode |
| 3 | Contouring | bone segmentation, full/trab/cort ROIs, material labels |
| 4 | Timelapsed Remodelling | transforms, common region, remodelling maps |
| 5 | Microarchitecture | thickness, spacing, BMD, volume, and porosity measurements |
| 6 | Plate/Rod Morphometry | plate/rod maps and network measurements |
| 7 | ParOsol-FEA | SED fields and mechanical summary |
| 8 | Mechanoregulation | remodelling-mechanics association tables and curves |

Motion Scoring is usually run near the beginning as a quality-control step.

## Scene Or Batch

| Mode | Use when | Output behavior |
| --- | --- | --- |
| Scene | you are reviewing one loaded scan or one loaded longitudinal series | loads outputs back into the current Slicer scene |
| Batch Processor | you are processing a cohort from disk | writes derivative artifacts and loads selected completed results |

Batch Processor rows should report missing prerequisites before a job is launched. If a row is missing contours, run Contouring first. If a row is missing a common region, run Timelapsed Remodelling first.

## Reuse Between Tools

The toolbox is designed so downstream tools reuse upstream derivatives:

- Timelapsed Remodelling can consume `ImportedContours` or `BoneContours`.
- Microarchitecture and Plate/Rod Morphometry can reuse native maps and apply common regions only during measurement.
- Mechanoregulation consumes Timelapsed remodelling maps and ParOsol-FEA SED fields.

Manifests in `derivatives/<Family>/manifest.json` are the preferred source of truth for reusable outputs.
