# External Modules

Place maintained forks, git submodules, or git subtrees for optional Slicer modules here when they should ship inside the Bone Imaging Toolbox extension.

Supported layouts:

```text
ExternalModules/
  ExampleModule/
    CMakeLists.txt
    ExampleModule.py

ExternalModules/
  ExampleExtensionFork/
    ExampleModule/
      CMakeLists.txt
      ExampleModule.py
```

The top-level toolbox CMake file and `scripts/link_local_toolbox_modules.py` discover scripted module folders with a same-named `.py` file and `CMakeLists.txt`.
