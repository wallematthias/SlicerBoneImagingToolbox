# Bone Microarchitecture

`Bone Microarchitecture` computes microarchitecture measurements from Slicer masks using the lightweight `bone-microarchitecture` Python core. The Slicer module handles node selection, AIM calibration, segmentation export, table display, and map loading; the core package handles the measurements.

## Inputs

- `Grayscale/BMD volume`: optional grayscale or BMD-calibrated image used for `Tb.BMD` and `Ct.BMD`. If the node was loaded with Scanco I/O and has an AIM source path, the module prefers re-reading and calibrating that source through `aimio-py`.
- `Bone segmentation`: optional binary bone segmentation. Use this with `Cortical mask` to derive Tb.N, compartment volumes, and cortical porosity.
- `Full/periosteal mask`: binary mask defining the total analysis region, or segmentation node with a `Full mask` segment.
- `Trabecular mask`: binary trabecular compartment labelmap, scalar mask, or segmentation node with a `Trabecular mask` segment.
- `Cortical mask`: optional cortical compartment mask, or segmentation node with a `Cortical mask` segment.

The masks should be in the same image space. The module reads labelmaps directly. When a Slicer segmentation node is selected, use the adjacent `Segment` dropdown to choose the exact segment, or leave it on `Auto` for generated names such as `Full mask`, `Trabecular mask`, `Cortical mask`, and `Bone segmentation`.

## Outputs

- A Slicer summary table with one row per parameter and columns for `Mean`, `Median`, `SD`, `P5`, `P25`, `P75`, `P95`, `Min`, `Max`, and `Units`. Map-backed parameters such as `Tb.Th`, `Tb.Sp`, `Tb.N`, `Ct.Th`, `Ct.Po.Dm`, `Tb.BMD`, and `Ct.BMD` include distribution statistics; scalar parameters such as `Tb.BV/TV`, `Ct.Po`, and compartment volumes report the scalar value in `Mean`. `Tb.BV/TV` and `Ct.Po` are reported as unitless fractions.
- With bone segmentation plus cortical mask: `Tb.N`, `Tb.1/N.SD`, `Tb.BV`, `Tb.TV`, `Ct.BV`, `Ct.TV`, `Ct.Po.V`, `Ct.Po`, and `Ct.Po.Dm`.
- With cortical mask: cortical thickness statistics, reported as `Ct.Th`, `Ct.Th SD`, `Ct.Th Min`, and `Ct.Th Max`.
- With grayscale/BMD volume: masked BMD outputs `Tb.BMD`, `Tb.BMD SD`, `Ct.BMD`, and `Ct.BMD SD`.
- Scalar map volumes are loaded automatically for available local fields: `Tb.Th`, `Tb.Sp`, `Tb.N`, `Ct.Th`, `Ct.Po.Dm`, `Tb.BMD`, and `Ct.BMD` when the required masks/images are selected.
- The measurement table is shown in Slicer's `Tables` module after a successful run.
- The `Export measurements CSV` button writes the most recent table with the same summary columns shown in Slicer.
- Exact sphere fitting uses a platform-aware backend default: native Metal on macOS when available, OpenCL on Windows/Linux when `pyopencl` and a GPU runtime are available, and CPU otherwise. The backend recorded in the output table metadata is the resolved backend actually used.

The trabecular parameters are not independent. In particular, `Tb.N` is a local-map estimate derived from the spatial relationship between trabecular bone ridges and the trabecular domain, and it should not be interpreted as statistically independent from `Tb.BV/TV`, `Tb.Th`, and `Tb.Sp`.

## Workflow

### Single Scan

1. Generate or load a bone segmentation plus full/trabecular/cortical masks.
2. Open `Bone Imaging > HR-pQCT > Bone Microarchitecture`.
3. Select the grayscale/BMD volume if BMD should be reported.
4. Select the segmentation node or labelmaps for the bone, full/periosteal, trabecular, and cortical masks. If using a segmentation node, choose the matching segment in the adjacent `Segment` dropdown when needed.
5. Run `Run microarchitecture`. The Slicer table opens automatically and available maps are loaded into the scene.
6. Use `Export measurements CSV` to save the displayed measurements.

### Registered Series

The `Registered Series` tab is for longitudinal datasets where registration is used to define common full, trabecular, and cortical regions, but each timepoint is measured in its native image space. Dataset discovery uses the Timelapsed HR-pQCT filename/header logic.

1. Select the dataset root.
2. Use the default output root or choose a folder. The default is `derivatives/RegisteredMicroarchitecture` under the dataset root.
3. Run `Discover series`.
4. Use the `Subject` and `Site` dropdowns to show all discovered sessions or a selected subset.
5. Review the session table.
6. Run `Prepare registered workspace` to write the `RegisteredMicroarchitecture` folder structure and manifest, including sequential adjacent registration pairs.
7. Run `Run series measurements` to compute complete sessions that already have native image, bone segmentation, full, trabecular, and cortical mask files. Per-session results are written under `native_space/ses-*/microarchitecture`, and the combined table is written as `microarchitecture_long.csv`.

If the core package is not available, install it from `Bone Imaging > Setup > Toolbox Setup` or the module's `Install / update microarchitecture core` button.
