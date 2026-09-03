# Mechanoregulation

Mechanoregulation combines Timelapsed remodelling maps with ParOsol-FEA strain-energy-density fields. It reports how formation and resorption relate to local mechanical stimulus.

Core analysis logic lives in:

https://github.com/wallematthias/BoneMechanoregulation

## Required Inputs

| Input | Meaning |
| --- | --- |
| Remodelling map | labelled formation and resorption events from Timelapsed |
| Baseline SED field | FEA field for the baseline scan of the comparison |
| Analysis mask | optional mask restricting the analysis region |

If no mask is selected in scene mode, the analysis uses the whole remodelling image.

## Scene Mode

Use scene mode when the remodelling map and SED field are already loaded.

1. Select the remodelling map.
2. Map formation, resorption, and optional quiescent labels if the image is labelled differently from the toolbox default.
3. Select the SED field.
4. Optionally select an analysis mask and segment.
5. Set bootstrap iterations.
6. Run.
7. Load the compact summary table and curve figure.

Scene mode resamples selected inputs onto the SED/remodelling analysis grid when needed.

## Batch Mode

Use `Bone Imaging > I/O > Batch Processor`.

Each row corresponds to one remodelling comparison with a matching baseline SED field. If the SED is missing, run ParOsol-FEA first.

## Outputs

Outputs are written under `derivatives/Mechanoregulation/`:

- summary CSV,
- curve PNGs,
- formation/resorption event segmentation for scene review.

The loaded Slicer table focuses on:

| Metric | Units |
| --- | --- |
| `CCR` | fraction |
| `Lazy min` | normalized SED units |
| `Lazy max` | normalized SED units |
| `ORR` | odds ratio per normalized SED unit |
| `ORF` | odds ratio per normalized SED unit |

## Screenshot To Add

Add one generic screenshot showing the selected remodelling map, SED field, and compact summary table.

## Citation

For mechanoregulation outputs, cite:

Walle M, Whittier DE, Schenk D, Atkins PR, Blauth M, Zysset P, Lippuner K, Müller R, Collins CJ. Precision of bone mechanoregulation assessment in humans using longitudinal high-resolution peripheral quantitative computed tomography in vivo. *Bone*. 2023;172:116780. doi: [10.1016/j.bone.2023.116780](https://doi.org/10.1016/j.bone.2023.116780).
