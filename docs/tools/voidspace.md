# Voidspace

Voidspace computes void masks and measurements from binary bone segmentations. The Slicer module handles node selection, result loading, and cohort orchestration. The `voidspace` core package owns the algorithm, command-line interface, file I/O, and numerical behavior.

https://github.com/wallematthias/voidspace

## Required Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| Bone segmentation | yes | binary bone mask used to define the solid structure |
| Mask | optional in native scene mode, required for common-region batch profiles | analysis-domain mask already prepared by the upstream workflow |
| Native full mask | registered batch profile | full bone mask in the same native space as the segmentation |
| Registered segmentation | dynamic batch profile | segmentation already transformed by Timelapsed Remodelling |
| Registered full mask | dynamic batch profile | full bone mask in the same registered space as the segmentation |
| Common region | registered and dynamic batch profiles | scan/FOV common region from Timelapsed Remodelling |

Voidspace does not own registration. The Registered voidspace profile uses native segmentations with the matching native-space common region. The Dynamic voidspace profile expects registered inputs that already exist, usually under `derivatives/Timelapse/sub-*/ses-*/xct/transformed/`.

## Scene Mode

Use `Bone Imaging > Microstructural Analysis > Voidspace` for one loaded case or one loaded baseline/follow-up pair.

For a cross-sectional run:

1. Select a segmentation volume.
2. Optionally select one mask volume.
3. Run the analysis.
4. Review the loaded all-void and large-void masks and measurement table.

For a longitudinal run:

1. Enable `Longitudinal voidspace`.
2. Select baseline segmentation and optional baseline mask.
3. Select follow-up segmentation and optional follow-up mask.
4. Run the analysis.
5. Review baseline/follow-up voidspace outputs and the change maps.

Scene mode writes temporary files for the core package, then loads the result nodes back into the active scene.

## Batch Mode

Use `Bone Imaging > I/O > Batch Processor`.

Three profiles are exposed:

| Profile | Behavior |
| --- | --- |
| Voidspace | runs native voidspace from each native segmentation |
| Registered voidspace | reuses or creates native voidspace maps, then constrains measurement and output masks to `native full mask & native common region` |
| Dynamic voidspace | runs pair-local registered-space voidspace for adjacent Timelapsed pairs, then compares baseline and follow-up large-void masks |

The native profile writes full-image voidspace maps. The registered profile stays in each scan's native segmentation space and uses the matching native common region for common-region-restricted reporting. The dynamic profile discovers adjacent Timelapsed pairs such as `001-002` and `002-003` when those pairs are available.

Dynamic voidspace uses the registered Timelapsed segmentations as inputs, but it writes its baseline and follow-up intermediate voidspace maps inside the dynamic pair folder. Running dynamic voidspace should not cause standalone Registered voidspace rows to switch to `Load`.

For native plus common-region reporting, combine masks upstream and provide the combined mask as the single Voidspace mask. The Voidspace batch interface intentionally exposes one mask input.

## Loaded Outputs

Batch loading creates one Slicer segmentation node per row or grouped pair.

Native rows load:

- source bone segmentation in gray,
- all voidspace,
- large voidspace,
- measurement table.

Registered rows load one segmentation node per timepoint because each timepoint remains in its own native segmentation space. Each node contains:

- source bone segmentation in gray,
- analysis mask when available,
- all voidspace,
- large voidspace,
- measurement table.

Dynamic rows load:

- baseline bone segmentation in gray,
- follow-up bone segmentation in gray,
- baseline analysis mask when available,
- follow-up analysis mask when available,
- quiescent voidspace,
- expanded voidspace,
- contracted voidspace,
- change measurement table.

Registered analysis masks are `native full mask & native common region` for each timepoint. Dynamic analysis masks are `registered full mask & registered common region` for each compared timepoint. The common region itself is a scan/FOV overlap mask; it is not a biological mask and is not interpreted as voidspace.

## Outputs

Outputs are written under `derivatives/Voidspace/`:

```text
derivatives/
  Voidspace/
    sub-001/
      ses-001/
        xct/
          native/
            voi-radiusleft/
          registered/
            voi-radiusleft/
      ses-001-002/
        xct/
          dynamic/
            voi-radiusleft/
              baseline/
              followup/
```

Common files include:

| File | Meaning |
| --- | --- |
| `voidspace_all_mask.*` | all detected voidspace |
| `voidspace_large_mask.*` | size-filtered large voidspace |
| `voidspace_analysis_mask.*` | mask used to constrain registered analysis |
| `voidspace_expanded_mask.*` | voidspace present at follow-up but absent at baseline |
| `voidspace_contracted_mask.*` | voidspace present at baseline but absent at follow-up |
| `voidspace_quiescent_mask.*` | voidspace present at both timepoints |
| `voidspace_measurements.csv` | cross-sectional measurements |
| `voidspace_change_measurements.csv` | dynamic change measurements |

## Citation

For cross-sectional voidspace, cite:

Whittier DE, Burt LA, Boyd SK. A new approach for quantifying localized bone loss by measuring void spaces. *Bone*. 2021 Feb;143:115785. doi: [10.1016/j.bone.2020.115785](https://doi.org/10.1016/j.bone.2020.115785).

For dynamic voidspace, cite:

Whittier DE, Walle M, Atkins PR, Collins CJ, Zumstein MA, Christen P, Lippuner K, Müller R. Structural alterations during fracture healing lead to void spaces developing in surrounding bone microarchitecture. *Journal of Bone and Mineral Research*. 2025 Jun;40(6):791-798. doi: [10.1093/jbmr/zjaf046](https://doi.org/10.1093/jbmr/zjaf046).
