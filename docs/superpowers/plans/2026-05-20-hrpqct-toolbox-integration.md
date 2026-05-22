# HR-pQCT Toolbox Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve the existing Timelapsed Slicer extension into an HR-pQCT toolbox that contains Timelapsed, MotionScore, Scanco I/O, and contouring/segmentation modules.

**Architecture:** Keep one Slicer extension package and expose separate scripted modules under the `HR-pQCT` category. The Slicer code remains a GUI/workflow wrapper; AIM reading/writing delegates to the core `timelapsedhrpqct.io.aim` APIs and MotionScore remains its own module.

**Tech Stack:** 3D Slicer scripted modules, Qt/CTK widgets, SimpleITK, `timelapsedhrpqct`, `motionscorehrpqct`.

---

### Task 1: Toolbox Metadata And Module Wiring

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `TimelapsedHRpQCT/TimelapsedHRpQCT.py`
- Create: `MotionScoreHRpQCT/`
- Create: `ScancoIO/`
- Create: `HRpQCTSegmentation/`

- [x] Change the extension project/display metadata to HR-pQCT Toolbox while keeping the existing repository URL.
- [x] Add the MotionScore, Scanco I/O, and segmentation modules as CMake subdirectories.
- [x] Put all modules in the Slicer module category `HR-pQCT`.

### Task 2: Scanco I/O Module

**Files:**
- Create: `ScancoIO/CMakeLists.txt`
- Create: `ScancoIO/ScancoIO.py`
- Create: `ScancoIO/Resources/Icons/ScancoIO.png`

- [x] Add AIM import with `bmd`, `native`, `mu`, and `hu` scaling choices.
- [x] Store AIM metadata on the imported Slicer volume so export does not require a reference AIM.
- [x] Add AIM export for grayscale or mask nodes, with optional external metadata JSON fallback.

### Task 3: Contours And Segmentation Module

**Files:**
- Create: `HRpQCTSegmentation/CMakeLists.txt`
- Create: `HRpQCTSegmentation/HRpQCTSegmentation.py`
- Create: `HRpQCTSegmentation/Resources/Icons/HRpQCTSegmentation.png`

- [x] Add a simple threshold-to-segmentation workflow for HR-pQCT volumes.
- [x] Open Slicer Segment Editor for manual contour cleanup after creating the segmentation.
- [x] Keep this module as Slicer workflow glue rather than core algorithm logic.

### Task 4: Documentation And Validation

**Files:**
- Modify: `README.md`
- Validate: Python syntax for all scripted modules.

- [x] Update README to describe the toolbox and each module.
- [x] Run syntax compilation for scripted module files.
