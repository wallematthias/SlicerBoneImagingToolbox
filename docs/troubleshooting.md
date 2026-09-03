# Troubleshooting

## Module Does Not Appear In Slicer

Run the local link helper again and restart Slicer:

```python
script = "/path/to/SlicerBoneImagingToolbox/scripts/link_local_toolbox_modules.py"
exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
```

Do not add only the repository root in Slicer settings. Slicer needs the individual module folders.

## Package Is Installed But Not Ready

Open `Bone Imaging > Setup > Toolbox Setup` and check the package row. The Setup page is the canonical place to install or update public runtime packages.

Common causes:

- Slicer Python has an older PyPI package than the module expects.
- A compiled backend is missing for the current platform.
- Slicer needs to be restarted after a package update.

## Masks Do Not Overlap Images

Check how the image and mask were loaded:

- AIM images and AIM masks should be loaded through Scanco I/O when possible.
- AIM masks must be read as masks, not as density images.
- Segmentations loaded without a reference image may need slice views centered on the segmentation bounds.
- Scene-mode tools resample selected inputs to the analysis grid when a workflow requires matching arrays.

## Batch Row Says Inputs Are Missing

The Batch Processor only launches rows that have the inputs required by the selected tool and profile.

Typical fixes:

- Run Dataset Naming Helper if the dataset is not normalized.
- Run Contouring if segmentation or ROI masks are missing.
- Run Timelapsed Remodelling if registered/common-region outputs are missing.
- Check that left and right scans have distinct `voi-*` names such as `radiusleft` and `radiusright`.

## Files With `;1` Suffixes

Some systems preserve versioned AIM names such as `scan.AIM;1`. Dataset Naming Helper strips that suffix during normalization and records the change in the rename manifest.

## Scene Feels Sluggish

Large loaded cohorts can make Slicer slow over time. Prefer loading only the outputs you need for review, and clear old result nodes when moving between cases. Batch outputs remain on disk and can be loaded again.

## ReadTheDocs Build Fails

Run the same local check:

```bash
python3 -m pip install -r docs/requirements.txt
mkdocs build --strict
```

Fix broken links or missing files before pushing.
