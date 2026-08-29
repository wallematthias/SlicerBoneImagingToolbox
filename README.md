<p align="center">
  <img src="resources/SlicerBoneImagingToolbox.png" alt="Bone Imaging Toolbox logo" width="360">
</p>

# Bone Imaging Toolbox for 3D Slicer

Bone Imaging Toolbox is a 3D Slicer extension for bone-imaging workflows. It groups focused tools for longitudinal HR-pQCT analysis, motion grading, Scanco image import/export, CT spine segmentation, HR-pQCT segmentation, microarchitecture, and plate/rod morphometry into one installable Slicer extension.

The extension appears in Slicer as:

```text
Bone Imaging
  Setup
    Toolbox Setup
  I/O
    Scanco I/O
  HR-pQCT
    Timelapsed HR-pQCT
    Motion Scoring
    Segmentation and Contours
    Bone Microarchitecture
    Plate/Rod Morphometry
  CT
    Spine Segmentation
```

## Included Tools

| Tool | Slicer Category | What It Does | Guide |
| --- | --- | --- | --- |
| Toolbox Setup | `Bone Imaging > Setup` | Checks and updates the toolbox plus Slicer Python runtime package versions from one dashboard. | Runtime Dependencies |
| Timelapsed HR-pQCT | `Bone Imaging > HR-pQCT` | Runs longitudinal HR-pQCT processing, registration, remodelling analysis, and review. | [Timelapsed HR-pQCT](docs/tools/timelapsed-hrpqct.md) |
| Motion Scoring | `Bone Imaging > HR-pQCT` | Runs and reviews HR-pQCT motion grading using MotionScore models. | [Motion Scoring](docs/tools/motion-scoring.md) |
| Segmentation and Contours | `Bone Imaging > HR-pQCT` | Creates HR-pQCT full, trabecular, cortical, binary, and material labelmaps. | [Segmentation and Contours](docs/tools/segmentation-and-contours.md) |
| Bone Microarchitecture | `Bone Imaging > HR-pQCT` | Computes trabecular microarchitecture, BMD, and compartment measures from masks. | [Bone Microarchitecture](docs/tools/microarchitecture.md) |
| Plate/Rod Morphometry | `Bone Imaging > HR-pQCT` | Runs topology-preserving plate/rod thinning, ports labels back to full thickness, and summarizes plate, rod, and junction measures. | [Plate/Rod Morphometry](docs/tools/plate-rod-morphometry.md) |
| Scanco I/O | `Bone Imaging > I/O` | Imports Scanco AIM, ISQ, SCV, and GOBJ images, and exports AIM images, masks, and metadata. | [Scanco I/O](docs/tools/scanco-io.md) |
| Spine Segmentation | `Bone Imaging > CT` | Runs PyTorch spine CT segmentation and loads vertebral-level, process/body, and cortical/trabecular outputs. | [Spine Segmentation](docs/tools/spine-segmentation-ct.md) |

The Slicer modules are wrappers around core Python packages where possible:

| Core package | Used By | Source |
| --- | --- | --- |
| `timelapsed-hrpqct` | Timelapsed HR-pQCT, registered microarchitecture discovery and common-region setup | https://github.com/wallematthias/TimelapsedHRpQCT |
| `MotionScoreHRpQCT` | Motion Scoring | https://github.com/wallematthias/MotionScoreHRpQCT |
| `bone-microarchitecture` | Bone Microarchitecture | https://github.com/wallematthias/bone-microarchitecture |
| `plate-rod-thinning` | Plate/Rod Morphometry | https://github.com/wallematthias/bone-plate-rod-thinning |
| `aimio-py` / `py_aimio` | Scanco I/O and AIM-backed BMD calibration | https://github.com/wallematthias/aimio-py |
| `spine-segment` | Spine Segmentation | https://github.com/wallematthias/spine-segment |

Each tool guide contains its own focused workflow instructions and attribution/citation notes.

## Runtime Dependencies

Some tools need Slicer-side dependencies in addition to this toolbox:

- **Toolbox Setup** is the recommended place to check for toolbox updates and install or update runtime packages after the toolbox itself is installed.
- **Motion Scoring** requires the `PyTorch` extension from Slicer's Extension Manager. Install `PyTorch`, restart Slicer, then run Motion Scoring.
- **Spine Segmentation** can run in Slicer Python with the `PyTorch` extension, or in an external arm64 conda runtime for Apple Silicon MPS acceleration. The module includes buttons to install/update either runtime and probes MPS `Conv3D` support before using the conda path.
- **Timelapsed HR-pQCT** installs/updates the `timelapsed-hrpqct` Python runtime from inside the module.
- **Bone Microarchitecture** uses the lightweight `bone-microarchitecture` Python core for trabecular BV/TV, thickness, separation, cortical thickness, BMD, compartment volumes, Tb.N, and cortical porosity.
- **Plate/Rod Morphometry** installs the published `plate-rod-thinning` wheel from PyPI and verifies that the compiled backend is importable in Slicer Python.
- **Scanco I/O** installs/updates the lightweight `aimio-py` / `py_aimio` Scanco image reader-writer stack from inside the module.

Open `Bone Imaging > Setup > Toolbox Setup` to see installed and latest PyPI versions for the main runtime packages. Update buttons appear next to missing or out-of-date packages. Package changes are only made after confirmation.

## Installation

### Extension Manager

When the toolbox is listed for your Slicer version:

1. Open 3D Slicer.
2. Install `Bone Imaging Toolbox` from Extension Manager.
3. Restart 3D Slicer.
4. Open modules from the `Bone Imaging` category.

### Manual Install From A Clone

Until the toolbox is available for your Slicer version, clone this repository and add the module folders to Slicer:

```bash
git clone https://github.com/wallematthias/SlicerBoneImagingToolbox.git
```

Then in Slicer:

1. Open `View -> Python Interactor`.
2. Run the helper script below, replacing `<repo>` with the folder containing this README:

```python
script = "<repo>/scripts/link_local_toolbox_modules.py"
exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
```

3. Restart Slicer.

Example macOS path:

```python
script = "/Users/<you>/Documents/14_GitHub/active/SlicerBoneImagingToolbox/scripts/link_local_toolbox_modules.py"
exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
```

### Manual Slicer Settings Alternative

Instead of using the helper script, add the module folders in `Edit -> Application Settings -> Modules -> Additional module paths`, then restart Slicer:

- `<repo>/HRpQCTTools/TimelapsedHRpQCT`
- `<repo>/HRpQCTTools/MotionScoreHRpQCT`
- `<repo>/HRpQCTTools/SegmentationHRpQCT`
- `<repo>/HRpQCTTools/BoneMicroarchitecture`
- `<repo>/HRpQCTTools/PlateRodMorphometryHRpQCT`
- `<repo>/IOTools/ScancoIO`
- `<repo>/CTTools/SpineSegmentationCT`
- `<repo>/Setup/BoneImagingToolboxSetup`

Do not add only the top-level repository folder. Slicer needs each module folder above. The helper script also discovers any vendored scripted modules under `ExternalModules/`.

## Tool Documentation

- [Timelapsed HR-pQCT](docs/tools/timelapsed-hrpqct.md): longitudinal HR-pQCT analysis, input naming, results layout, remodelling review, and citations.
- [Motion Scoring](docs/tools/motion-scoring.md): PyTorch setup, model bundle setup, prediction/review workflow, and motion-grading citation.
- [Segmentation and Contours](docs/tools/segmentation-and-contours.md): segmentation presets, Laplace-Hamming notes, mask utilities, and attribution.
- [Bone Microarchitecture](docs/tools/microarchitecture.md): trabecular BV/TV, Tb.Th, Tb.Sp, cortical thickness, BMD, and compartment measures from Slicer masks.
- [Plate/Rod Morphometry](docs/tools/plate-rod-morphometry.md): plate/rod thinning, full-thickness label propagation, individual element labels, junction measures, visualization, and citation.
- [Scanco I/O](docs/tools/scanco-io.md): AIM/ISQ/SCV/GOBJ import, AIM export, metadata handling, and attribution.
- [Spine Segmentation](docs/tools/spine-segmentation-ct.md): CT input, PyTorch setup, `spine-segment` outputs, and attribution.

## Attribution And Citation

Use the citation that matches the workflow and results you report. Core packages and third-party methods may also have their own license and citation requirements.

| Scope | Cite / Credit | When To Cite |
| --- | --- | --- |
| Timelapsed HR-pQCT mechanoregulation and remodelling analysis | Walle M, Whittier DE, Schenk D, Atkins PR, Blauth M, Zysset P, Lippuner K, Müller R, Collins CJ. Precision of bone mechanoregulation assessment in humans using longitudinal high-resolution peripheral quantitative computed tomography in vivo. *Bone*. 2023;172:116780. doi: 10.1016/j.bone.2023.116780. | Longitudinal HR-pQCT remodelling or mechanoregulation outputs. |
| Multistack registration | Whittier DE, Walle M, Schenk D, Atkins PR, Collins CJ, Zysset P, Lippuner K, Müller R. A multi-stack registration technique to improve measurement accuracy and precision across longitudinal HR-pQCT scans. *Bone*. 2023;176:116893. doi: 10.1016/j.bone.2023.116893. | Multistack registration or workflows that depend on it. |
| Motion Scoring | Walle M, Eggemann D, Atkins PR, Kendall JJ, Stock K, Müller R, Collins CJ. Motion grading of high-resolution quantitative computed tomography supported by deep convolutional neural networks. *Bone*. 2023;166:116607. doi: 10.1016/j.bone.2022.116607. | Automated or reviewed HR-pQCT motion grades from MotionScore. |
| Plate/Rod network morphometry | Walle M, Yeritsyan D, Abbasian M, Oftadeh R, Müller R, Nazarian A. A graph model to describe the network connectivity of trabecular plates and rods. *Front Bioeng Biotechnol*. 2024;12:1384280. doi: 10.3389/fbioe.2024.1384280. PMID: 38770275; PMCID: PMC11103010. | Plate/rod labels, graph connectivity, or plate/rod summary measures. |
| Spine Segmentation vertebral localization and levels | Payer C, Stern D, Bischof H, Urschler M. Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net. In: VISIGRAPP 2020, Volume 5: VISAPP. 2020;124-133. doi:10.5220/0008975201240133. | Vertebral centroids or vertebral-level labels generated by the spine segmentation workflow. |
| Spine process/body compartment workflow | Walle M, Matheson BE, Boyd SK. Comparing linear and nonlinear finite element models of vertebral strength across the thoracolumbar spine: a benchmark from density-calibrated computed tomography. *GigaScience*. 2025;14:giaf094. doi:10.1093/gigascience/giaf094. | Process/body and cortical/trabecular spine compartment outputs. |
| Segmentation, contours, microarchitecture, and Scanco I/O | Credit this toolbox and the specific core packages used in the analysis record. Cite method papers for any study-specific segmentation, contour, or microarchitecture definitions required by the target field or journal. | Mask generation, derived compartment measurements, or Scanco import/export support. |

## Adding External Modules

External modules can be checked into `ExternalModules/` as maintained forks, git subtrees, or git submodules. A direct scripted module folder must contain `CMakeLists.txt` and a same-named `.py` file, for example:

```text
ExternalModules/
  SlicerParOSol/
    ParOSolFEA/
      CMakeLists.txt
      ParOSolFEA.py
```

The top-level `CMakeLists.txt` discovers these folders during ExtensionIndex builds. The local `scripts/link_local_toolbox_modules.py` helper also adds them to Slicer's `Modules/AdditionalPaths`.

For a module to feel native inside the toolbox, set its Slicer category to a dot-separated submenu such as `Bone Imaging.FEA`, `Bone Imaging.I/O`, or `Bone Imaging.HR-pQCT` in the vendored fork.

## Repository Layout

```text
HRpQCTTools/
  TimelapsedHRpQCT/
  MotionScoreHRpQCT/
  SegmentationHRpQCT/
  BoneMicroarchitecture/
  PlateRodMorphometryHRpQCT/
IOTools/
  ScancoIO/
CTTools/
  SpineSegmentationCT/
ExternalModules/
resources/
docs/tools/
```

## License

This extension is distributed under the MIT License. Core pipeline packages installed from PyPI or separate repositories are governed by their own license terms.
