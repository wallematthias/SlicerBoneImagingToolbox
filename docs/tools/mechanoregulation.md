# Mechanoregulation

Mechanoregulation combines Timelapsed remodelling maps with ParOsol FEA SED fields. It reports formation, resorption, activity, net-volume metrics, classification thresholds, and odds ratios.

## Inputs

- A Timelapsed remodelling map with formation and resorption labels.
- A matching baseline-space SED field from ParOsol-FEA.
- Optional analysis ROI masks.

Scene mode works from loaded Slicer nodes. Batch mode is available through the Batch Processor and discovers compatible Timelapsed and FEA derivatives.

## Outputs

The module writes summary tables and curve figures into `derivatives/Mechanoregulation/`. SED values are normalized to a 0-100 axis for mechanoregulation curves and OR-F/OR-R are reported per one normalized SED percentage point.

## Citation

For mechanoregulation outputs, cite [Walle et al., Bone 2023](https://doi.org/10.1016/j.bone.2023.116780).
