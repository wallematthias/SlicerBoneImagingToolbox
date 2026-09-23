# Timelapsed Common-Region Scene Load-Back

## Goal

After a successful Timelapsed scene run, make the generated all-timepoint common regions immediately available to downstream scene workflows such as Mechanoregulation. Users should not need to locate or load temporary mask files manually.

## User Experience

Timelapsed scene mode will load one segmentation node named from the processed subject and site, for example `sub-001_site-scene_common-regions`. The node will appear beside the remodelling output under:

`TimelapsedHRpQCT Loaded/sub-001/site-scene`

The segmentation will contain the generated `full`, `trab`, and `cort` common-region segments when those roles were requested and produced. Mechanoregulation can select this node as its analysis mask and select the required segment, normally `full`.

## Data Flow

1. The existing Timelapsed pipeline generates common-region masks in the scene run output.
2. Scene load-back discovers common-region outputs for the processed subject and site.
3. Available requested roles are imported into one segmentation node without changing their voxel values or geometry.
4. Each segment receives a readable name and a role tag matching `full`, `trab`, or `cort`.
5. The segmentation node is placed in the existing subject/site results folder.

The temporary scene-run output remains an interchange artifact. Downstream interactive modules consume the loaded Slicer segmentation node and do not depend on a durable derivatives folder.

## Lifecycle

The node will be marked with Timelapsed-generated attributes identifying it as a common-region result and recording its scene run. Before loading a replacement for the same processed subject and site, Timelapsed will remove the previous generated common-region node. User-created segmentations and unrelated generated nodes will not be removed.

## Error Handling

- Missing optional common-region roles are skipped and reported in the module log.
- If no requested common-region masks exist, no empty segmentation node is retained.
- Failure to import one role is reported while allowing other roles and normal scene outputs to load.
- Existing remodelling output, transform, and results-table load-back behavior remains unchanged.

## Testing

Focused tests will verify:

- common-region output discovery is limited to the processed subject and site;
- requested `full`, `trab`, and `cort` masks are combined into one segmentation node;
- segment names and role tags are assigned consistently;
- the node is placed in the existing Timelapsed subject/site folder;
- an earlier generated common-region node is replaced without touching user nodes;
- missing roles and partial import failures are handled without breaking other load-back behavior;
- existing Timelapsed scene-mode tests continue to pass.
