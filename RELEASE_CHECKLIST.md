# Release Checklist (Slicer Extension)

## Metadata

- [ ] `CMakeLists.txt` homepage points to this repository.
- [ ] `EXTENSION_ICONURL` set to public raw URL.
- [ ] `EXTENSION_SCREENSHOTURLS` set to real screenshots.
- [ ] Extension description/category/contributors verified.

## Module

- [ ] All toolbox modules load in Slicer developer mode from the `HR-pQCT` category:
  - [ ] `Timelapsed HR-pQCT`
  - [ ] `Motion Scoring`
  - [ ] `Scanco I/O`
  - [ ] `Contours and Segmentation`
- [ ] `Install / Update timelapsed-hrpqct` works on clean Slicer install.
- [ ] `Install / Update AIM I/O` installs `aimio-py` without installing the full timelapsed pipeline.
- [ ] Full pipeline run works on representative dataset.
- [ ] Analyze rerun works with changed threshold/cluster.
- [ ] Raw/transformed/remodelling loading works.
- [ ] Scanco AIM import/export round trip works on one representative AIM.
- [ ] Contours and Segmentation creates full/trabecular/cortical/segmentation outputs.
- [ ] Motion Scoring can find a local or downloaded model bundle and export a review table.

## Testing

- [ ] `TimelapsedHRpQCTTest` passes.
- [ ] `MotionScoreHRpQCTTest` passes.
- [ ] `ScancoIOTest` passes.
- [ ] `HRpQCTSegmentationTest` passes.
- [ ] Manual smoke test done on at least one real dataset.

## Docs

- [ ] README reflects current UI labels and workflow.
- [ ] Changelog updated for release version.
- [ ] Troubleshooting notes include config/dependency tips.

## Publish

- [ ] Tag release (e.g. `v0.1.0`).
- [ ] Submit/update Slicer Extensions Index entry.
