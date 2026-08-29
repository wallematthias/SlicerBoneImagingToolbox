cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(BoneImagingToolbox)

set(EXTENSION_HOMEPAGE "https://github.com/wallematthias/SlicerBoneImagingToolbox")
set(EXTENSION_CATEGORY "Quantification")
set(EXTENSION_CONTRIBUTORS "Matthias Walle (University of Calgary)")
set(EXTENSION_DESCRIPTION "Bone imaging toolbox for 3D Slicer with HR-pQCT workflows, Scanco AIM I/O, motion scoring, contours, segmentation, and vendored bone-analysis modules.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/wallematthias/SlicerBoneImagingToolbox/main/resources/icon.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/wallematthias/SlicerBoneImagingToolbox/main/resources/SlicerBoneImagingToolbox.png")
set(EXTENSION_DEPENDS "")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

set(TOOLBOX_SHARED_PYTHON_SCRIPTS
  SlicerBoneImagingToolboxLib/__init__.py
  SlicerBoneImagingToolboxLib/package_status.py
  SlicerBoneImagingToolboxLib/registry.py
  SlicerBoneImagingToolboxLib/segmentation_methods.py
  SlicerBoneImagingToolboxLib/slicer_pip.py
  SlicerBoneImagingToolboxLib/slicer_update_ui.py
  SlicerBoneImagingToolboxLib/updater.py
  SlicerBoneImagingToolboxLib/vertebra_labels.py
)

ctkMacroCompilePythonScript(
  TARGET_NAME SlicerBoneImagingToolboxLib
  SCRIPTS "${TOOLBOX_SHARED_PYTHON_SCRIPTS}"
  DESTINATION_DIR ${CMAKE_BINARY_DIR}/${Slicer_QTSCRIPTEDMODULES_LIB_DIR}
  INSTALL_DIR ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
  NO_INSTALL_SUBDIR
)

add_subdirectory(HRpQCTTools/TimelapsedHRpQCT)
add_subdirectory(HRpQCTTools/MotionScoreHRpQCT)
add_subdirectory(HRpQCTTools/SegmentationHRpQCT)
add_subdirectory(HRpQCTTools/BoneMicroarchitecture)
add_subdirectory(HRpQCTTools/PlateRodMorphometryHRpQCT)
add_subdirectory(IOTools/ScancoIO)
add_subdirectory(CTTools/SpineSegmentationCT)
add_subdirectory(Setup/BoneImagingToolboxSetup)

file(GLOB TOOLBOX_EXTERNAL_CMAKES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/ExternalModules/*/CMakeLists.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/ExternalModules/*/*/CMakeLists.txt"
)
list(SORT TOOLBOX_EXTERNAL_CMAKES)
foreach(external_cmake ${TOOLBOX_EXTERNAL_CMAKES})
  get_filename_component(external_module_dir "${external_cmake}" DIRECTORY)
  get_filename_component(external_module_name "${external_module_dir}" NAME)
  if(EXISTS "${external_module_dir}/${external_module_name}.py")
    file(RELATIVE_PATH external_module_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${external_module_dir}")
    message(STATUS "Adding Bone Imaging Toolbox external module: ${external_module_relative}")
    add_subdirectory("${external_module_relative}")
  endif()
endforeach()

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
