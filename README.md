<p align="center">
  <img src="resources/SlicerBoneImagingToolbox.png" alt="Bone Imaging Toolbox logo" width="360">
</p>

# Bone Imaging Toolbox for 3D Slicer

Bone Imaging Toolbox is a 3D Slicer extension for bone-imaging workflows. It provides Slicer interfaces for Scanco image import/export, dataset normalization, batch processing, contouring, longitudinal remodelling, microarchitecture, plate/rod morphometry, finite element analysis, mechanoregulation, motion grading, and CT spine segmentation.

The toolbox appears in Slicer under the `Bone Imaging` category. Reusable scientific logic lives in focused Python packages, while this repository owns the Slicer UI, scene integration, batch orchestration, setup page, and user documentation.

## Documentation

Full documentation is built with MkDocs and intended for Read the Docs Community:

https://slicerboneimagingtoolbox.readthedocs.io/

Until the hosted site is enabled, browse the source docs in [`docs/`](docs/index.md).

Key pages:

- [Installation](docs/installation.md)
- [Dataset Format](docs/dataset-format.md)
- [Batch Processor](docs/tools/batch-processor.md)
- [Derivative Workflow Contract](docs/derivatives.md)
- [Adding A Tool](docs/development/adding-a-tool.md)

## Install From A Clone

Clone this repository and link the module folders in Slicer:

```bash
git clone https://github.com/wallematthias/SlicerBoneImagingToolbox.git
```

In Slicer's Python Interactor:

```python
script = "/path/to/SlicerBoneImagingToolbox/scripts/link_local_toolbox_modules.py"
exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
```

Restart Slicer, then open `Bone Imaging > Setup > Toolbox Setup` to install or update runtime packages.

## Included Workflows

- Scanco I/O
- Dataset Naming Helper
- Batch Processor
- Motion Scoring
- Contouring
- Mask and Label Algebra
- Timelapsed Remodelling
- Mechanoregulation
- Microarchitecture
- Plate/Rod Morphometry
- ParOsol-FEA
- Spine Segmentation

## Core Packages

The Slicer modules wrap these focused packages where possible:

- [`bone-imaging-derivatives`](https://github.com/wallematthias/bone-imaging-derivatives)
- [`bone-contouring`](https://github.com/wallematthias/bone-contouring)
- [`timelapsed-hrpqct`](https://github.com/wallematthias/TimelapsedHRpQCT)
- [`bone-microarchitecture`](https://github.com/wallematthias/bone-microarchitecture)
- [`plate-rod-thinning`](https://github.com/wallematthias/bone-plate-rod-thinning)
- [`parosol-py`](https://github.com/wallematthias/parosol-py)
- [`bone-mechanoregulation`](https://github.com/wallematthias/BoneMechanoregulation)
- [`MotionScoreHRpQCT`](https://github.com/wallematthias/MotionScoreHRpQCT)
- [`aimio-py`](https://github.com/wallematthias/aimio-py)
- [`spine-segment`](https://github.com/wallematthias/spine-segment)

## Citation

Use the citation that matches the workflow and results you report. Tool-specific citation guidance is maintained in the documentation.

- Timelapsed remodelling and mechanoregulation: Walle M et al. *Bone*. 2023;172:116780. doi: [10.1016/j.bone.2023.116780](https://doi.org/10.1016/j.bone.2023.116780).
- Multistack registration: Whittier DE et al. *Bone*. 2023;176:116893. doi: [10.1016/j.bone.2023.116893](https://doi.org/10.1016/j.bone.2023.116893).
- Motion grading: Walle M et al. *Bone*. 2023;166:116607. doi: [10.1016/j.bone.2022.116607](https://doi.org/10.1016/j.bone.2022.116607).
- Plate/rod network morphometry: Walle M et al. *Front Bioeng Biotechnol*. 2024;12:1384280. doi: [10.3389/fbioe.2024.1384280](https://doi.org/10.3389/fbioe.2024.1384280).
- Spine vertebral localization: Payer C et al. VISAPP 2020. doi: [10.5220/0008975201240133](https://doi.org/10.5220/0008975201240133).
- Spine compartment workflow: Walle M and Matheson BE et al. *GigaScience*. 2025;14:giaf094. doi: [10.1093/gigascience/giaf094](https://doi.org/10.1093/gigascience/giaf094).

## Authorship

The Bone Imaging Toolbox Slicer modules were developed by Matthias Walle. Core packages and third-party methods may have their own authorship, license, and citation requirements.

## License

This extension is distributed under the MIT License. Core packages installed from PyPI or separate repositories are governed by their own license terms.
