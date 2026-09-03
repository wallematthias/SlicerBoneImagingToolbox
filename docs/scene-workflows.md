# Scene Workflows

Scene mode is for interactive work with nodes already loaded in Slicer.

Use scene mode when you want to:

- select a small number of loaded volumes or segmentations,
- inspect outputs immediately in 2D/3D views,
- adjust analysis settings interactively,
- load result tables into the Slicer scene.

Scene workflows should keep long processing off the Slicer UI thread. Modules use worker threads or subprocesses for heavier work and then load outputs back into the scene.

## Practical Pattern

1. Load the image, segmentation, masks, transforms, or fields needed by the tool.
2. Select exact segments when a segmentation node contains multiple labels.
3. Choose a profile or adjust exposed settings.
4. Run the workflow.
5. Inspect the loaded result nodes and table.
6. Export CSV outputs when needed.

## Scene Versus Batch

Scene mode favors direct review. Batch mode favors reproducible cohort processing from a normalized dataset root.

When both modes exist, they should call the same core package and write compatible derivative outputs.
