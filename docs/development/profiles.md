# Profiles

Profiles define reproducible parameter sets for core workflows.

## Rules

- Shipped profiles live with the core package that owns the algorithm.
- Custom profiles should be exported by the Slicer tool and stored in a user-controlled profile directory.
- Batch Processor profile lists should be tool-specific.
- The selected profile provides defaults; explicit user edits in the UI should become the settings used for that run.
- Profiles should record enough metadata to explain what was run in derivative sidecars and manifests.

## Examples

- Contouring profiles define scanner/site segmentation and contour parameters.
- Timelapsed profiles define registration and remodelling-analysis defaults.
- ParOsol-FEA profiles define material conversion, solver, and load-history settings.
- Mechanoregulation currently uses a compact standard workflow and selected analysis settings.
