# Batch Processor Contract Implementation Plan

Spec: `/Users/matthias.walle/Documents/14_GitHub/active/SlicerBoneImagingToolbox/docs/superpowers/specs/2026-09-02-hrpqct-batch-layout-contract.md`

## Global Constraints

- Prefer the clean normalized derivatives contract over historical layouts.
- Do not add new compatibility fallbacks for `site-*`, `native_space`, `RegisteredMicroarchitecture`, or `TimelapsedHRpQCT` paths.
- Keep discovery, derivative family naming, output paths, manifest records, and job planning centralized in `bone-imaging-derivatives`.
- Keep Slicer modules as thin scene/batch UI wrappers around package CLIs and shared discovery APIs.
- Write tests before behavior changes when changing package logic.

## Task 1: Shared Discovery Contract

Repo: `bone-imaging-derivatives`

Implement the shared batch discovery primitives that tool packages can call:

- normalized raw XCT image discovery under `sub-*/ses-*/xct`
- derivative manifest/file discovery under `derivatives/<Family>/sub-*/ses-*/xct`
- contour preference order: `ImportedContours`, then `BoneContours`
- VOI side preservation, including `radiusleft`, `radiusright`, `tibialeft`, `tibiaright`
- stack-aware keys and virtual stack metadata fields
- prerequisite status helpers returning `ready`, `loadable`, `missing`, or `review`

Tests:

- STRAMBO-style normalized sample with `ImportedContours` masks is discovered.
- `radiusleft` and `radiusright` do not collapse.
- missing required masks return `missing` rather than runnable.
- manifest paths are relative to dataset root.

## Task 2: Microarchitecture Batch Alignment

Repos: `bone-microarchitecture`, `SlicerBoneImagingToolbox`

Use the shared discovery contract for individual and registered microarchitecture batch rows:

- unregistered rows are subject-session-VOI-stack cases
- registered rows are subject-VOI-stack series
- `Register` implies common region
- `Skip existing` drives `Load` vs `Run`
- existing measurements and maps are discovered from the new `Microarchitecture` layout
- add missing `Tt.BMD` reporting for total/periosteal ROI

Tests:

- package batch finds masks from `ImportedContours` or `BoneContours`.
- package batch refuses incomplete rows before running.
- Slicer row status changes to `Load` when measurements exist.
- map discovery loads all generated maps.

## Task 3: Timelapse Batch Alignment

Repos: `TimelapsedHRpQCT`, `SlicerBoneImagingToolbox`

Make timelapse consume masks only from discovered/selected artifacts:

- no contour generation in timelapse pipeline
- registration/common-region/timelapse outputs use `Registration`, `CommonRegion`, and `Timelapse`
- row grouping uses subject-VOI-stack series
- selected generic ROI names stay generic in outputs and result tables
- role mapping uses selected segmentation plus arbitrary ROI labels without creating pseudo-sites

Tests:

- no generated mask stage is invoked from `run`.
- selected ROI labels do not become site names.
- baseline and adjacent pair modes produce the expected pair list from sessions.
- batch and scene load current-layout remodelling maps and result rows.

## Task 4: Plate/Rod And FEA Batch Consistency

Repos: `bone-plate-rod-thinning`, `SlicerBoneImagingToolbox`

Align batch UI and discovery with microarchitecture:

- action column first
- queue/cancel/load row states
- no top-level common-region checkbox when common region is implied by registered mode
- discovery prefers `ImportedContours` then `BoneContours`
- ParOsol-FEA discovers XCT material/model label maps for XCT profiles and exposes load-history profiles.

Tests:

- plate/rod discovers segmentation plus trab/selected ROI from derivatives.
- FEA batch discovers material label maps and reports non-XCT profiles as not implemented when needed.

## Task 5: Batch Processor Entry Point

Repo: `SlicerBoneImagingToolbox`

Add a stable Batch Processor entry point under I/O or Setup-adjacent workflow:

- requires normalized dataset root
- summarizes available subjects, sessions, VOIs, stacks, and derivatives
- selected tool/profile controls dynamic columns
- local backend queues jobs through existing package CLIs
- server backend slot is present but disabled/publicly non-operational

Tests:

- module imports without Slicer syntax errors.
- normalized dataset discovery populates expected rows.
- non-normalized root is rejected with a Dataset Naming Helper direction.

## Task 6: Verification And Cleanup

Run focused package tests and Slicer wrapper tests:

- `bone-imaging-derivatives`: discovery/layout/naming tests
- `bone-microarchitecture`: batch and measurement tests
- `TimelapsedHRpQCT`: timelapse discovery/registration/analysis tests
- `bone-plate-rod-thinning`: batch tests
- `SlicerBoneImagingToolbox`: focused module import/batch UI tests

Then inspect diffs for duplicated discovery logic and remove or delegate it to the shared package.
