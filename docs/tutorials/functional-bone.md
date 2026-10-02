# Functional Bone: batch tutorial

This tutorial explains how to run **Native Functional Bone** or **Registered
Functional Bone** in the Batch Processor. Functional Bone measures
microarchitecture after excluding large voidspace from the reporting region.
Contouring and Voidspace are upstream prerequisites, not separate analyses covered
in detail here.

## 1. Update and restart

Use Toolbox **0.3.3** or newer. Updating Python packages does not update an old
toolbox checkout: update the extension through your installation method first
(Extension Manager or the linked Git checkout), then open
`Bone Imaging > Setup > Toolbox Setup` and update the runtime packages.

For this release, check these minimum versions in Setup:

| Package | Minimum version |
| --- | --- |
| bone-imaging-derivatives | 0.1.7 |
| bone-contouring | 0.3.3 |
| voidspace | 0.1.4 |
| bone-microarchitecture | 0.2.5 |

Restart Slicer after updating so it does not keep previously imported code in
memory. Start with a fresh scene for the first comparison. Keep a backup of your
test dataset before renaming or forcing analyses.

## 2. Prepare the required inputs

1. Open `Bone Imaging > I/O > Dataset Naming Helper` and select the dataset root.
2. Click **Analyze**. Review subject, session, VOI/laterality, stack and file role.
   Repeated scans of one person should share a subject and have different sessions.
3. Correct uncertain fields in the table before clicking **Rename files**.
   Native Scanco D-number filenames alone do not establish patient identity.
   Patient-header grouping is a heuristic: verify it against your study records,
   particularly on exports from different scanners or storage systems.
4. Confirm raw scans are in `sub-*/ses-*/xct/` and supplied Scanco/IPL/manual masks
   are in `derivatives/ImportedContours/`, with the matching case identity.

The dataset root is the folder containing both `sub-*` and `derivatives`, not an
individual session, mask folder, or AIM file. See the
[naming guide](../tools/dataset-naming-helper.md) for examples. Undo restores names
and sidecars and archives the completed rename manifest, allowing another rename.
Keep identity-bearing rename manifests private.

For each case, you need the grayscale scan, tissue SEG, full/TRAB/CORT compartment
masks and a matching Large voidspace map. A tissue SEG contains bone tissue only;
a full compartment mask also includes the enclosed marrow and voids. They are
not interchangeable.

1. Open `Bone Imaging > I/O > Batch Processor`.
2. Choose the normalized dataset root, **Local** backend, tool **Bone Contouring**,
   and the appropriate standard profile, such as **XtremeCT II** or **XtremeCT II - LH**.
3. Leave **Skip existing** checked and click **Analyze**.
4. Resolve **Missing** rows before running. Click the row's **Run** button for
   incomplete cases or **Run all**. Use **Load** to inspect completed cases.

Supplied ImportedContours take priority; a standard contouring profile fills only
missing outputs rather than replacing scanner masks. Inspect the loaded masks in
axial and coronal/sagittal views before continuing. See the
[contouring guide](../tools/segmentation-and-contours.md) for method choices.

Next, select tool **Voidspace**, profile **Voidspace**, and click **Analyze**,
**Run**, then **Load** for a test case. Check that its **Large voidspace** map
matches the scan and full mask, including at the acquisition ends. Once satisfied,
run the remaining cases. This native map is the recommended prerequisite for
both Functional Bone profiles. For Voidspace settings and other workflows, use
the [Voidspace guide](../tools/voidspace.md).

## 3. Choose the Functional Bone profile

| Profile | Use when | Additional prerequisite |
| --- | --- | --- |
| Native Functional Bone | measuring an individual scan over its native bone domain | matching native Large voidspace map |
| Registered Functional Bone | comparing longitudinal scans over common scan support | reviewed registration and native-space CommonRegion for each timepoint |

For the registered profile, first run and review the upstream
[Timelapsed Remodelling](../tools/timelapsed-hrpqct.md) workflow. A common region
is overlapping scan/FOV support, not a full bone mask. Dynamic Voidspace is not a
prerequisite for either Functional Bone profile.

## 4. Run Functional Bone

1. In the Batch Processor, keep the same normalized dataset root and choose tool
   **Microarchitecture**.
2. Choose **Native Functional Bone** or **Registered Functional Bone**.
3. Click **Analyze**. Resolve any **Missing** prerequisites before running.
   Confirm the case identity and Large voidspace input; for registered analysis,
   also confirm the matching native-space common region.
4. **Run** one test row, then **Load** it. Inspect the effective analysis mask and
   measurements before using **Run all** for the remaining cases.

Native Functional Bone reports within `full mask AND NOT large voidspace`;
Registered Functional Bone additionally intersects the native common region.
Each reporting compartment is intersected with that analysis region. Both reuse
or create native maps rather than recomputing thickness on a void-clipped domain.
The registered profile prefers native Large voidspace but can fall back to a
matching Registered voidspace map.

Measurements are saved below
`derivatives/Microarchitecture/sub-<id>/ses-<id>/xct/`:

- Native: `functional_bone_native_measurements/`
- Registered: `functional_bone_measurements/`

Do not compare native and registered measurements as if their reporting regions
were identical. Older saved `functional-bone` jobs retain registered semantics.

## 5. Rerunning and troubleshooting

Loading old outputs does not correct them. Back up the prior results, then
uncheck **Skip existing**, click **Analyze**, and rerun the applicable Voidspace
profile. Rerun dependent Functional Bone/Microarchitecture measurements afterwards.
Keep Skip existing enabled for contour completion if you want to preserve supplied
scanner masks. Corrected mask placement does not require rewriting those masks.

| Symptom | Check |
| --- | --- |
| Missing voidspace in Native Functional Bone | Run native **Voidspace** for the same subject/session/VOI/stack; a registered-only map is insufficient. |
| Missing common region | Run/review Timelapsed upstream; use **Native Functional Bone** if a native analysis is intended. |
| Masks appear shifted | Verify source and mask case identities and original physical geometry; cropped masks can have different array dimensions. Do not fix placement by changing the scan origin. |

If a test still fails, share the Functional Bone profile, package versions,
affected case identifiers, batch log and representative slices without private
patient names. The [Batch Processor video](index.md#batch-processor) covers the
general interface.
