cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ParOSolFEA)

set(MODULE_PYTHON_SCRIPTS
  ${PROJECT_NAME}.py
  ParOSolFEALib/__init__.py
  ParOSolFEALib/slicer_adapters.py
  ParOSolFEALib/workflow_controllers.py
  ParOSolFEALib/workflow_stage.py
  )

slicerMacroBuildScriptedModule(
  NAME ${PROJECT_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
