# Adding A Tool

New tools should keep Slicer UI code separate from reusable scientific logic.

## Ownership

- SlicerBoneImagingToolbox owns Slicer modules, Qt UI, MRML node selection, visualization, setup status, and documentation.
- Core packages own algorithms, profiles, CLI entry points, file-level batch execution, numerical tests, and PyPI releases.
- `bone-imaging-derivatives` owns shared dataset parsing, derivative discovery, manifests, portable paths, and prerequisite planning.

## Checklist

1. Define the core package that owns the logic.
2. Define derivative family names, artifact roles, required inputs, optional inputs, and profile names.
3. Add CLI support in the core package for batch execution.
4. Add tests for numerical behavior, discovery, profiles, and derivative writing.
5. Add a Slicer scene module only when loaded-node interaction is useful.
6. Add Batch Processor support for cohort processing.
7. Add runtime package status rows in the Setup module.
8. Add or update user documentation in this docs site.
9. Add citation and attribution guidance close to the tool description.
10. Run focused Slicer wrapper tests and relevant core package tests before release.

See `AGENTS.md` in the repository root for the full engineering contract.
