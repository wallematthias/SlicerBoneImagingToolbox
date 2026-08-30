cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

set(MODULE_NAME MechanoregulationHRpQCT)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
