#!/usr/bin/env bash
set -euo pipefail

# Reproducible non-Slicer environment for running Bone Imaging Toolbox batch CLIs
# on ARC/SLURM. Run this on the remote machine after loading your preferred
# conda/mamba module.

ENV_NAME="${BONE_BATCH_CONDA_ENV:-bone-batch}"
PYTHON_VERSION="${BONE_BATCH_PYTHON_VERSION:-3.12}"
CONDA_BIN="${CONDA_BIN:-conda}"
BONE_IMAGING_DERIVATIVES_SPEC="${BONE_IMAGING_DERIVATIVES_SPEC:-bone-imaging-derivatives}"
BONE_CONTOURING_SPEC="${BONE_CONTOURING_SPEC:-bone-contouring}"
TIMELAPSED_HRPQCT_SPEC="${TIMELAPSED_HRPQCT_SPEC:-timelapsed-hrpqct}"
BONE_MICROARCHITECTURE_SPEC="${BONE_MICROARCHITECTURE_SPEC:-bone-microarchitecture}"
BONE_PLATE_ROD_THINNING_SPEC="${BONE_PLATE_ROD_THINNING_SPEC:-bone-plate-rod-thinning}"
PAROSOL_PY_SPEC="${PAROSOL_PY_SPEC:-parosol-py}"
BONEMECHREG_SPEC="${BONEMECHREG_SPEC:-bonemechreg}"

if command -v micromamba >/dev/null 2>&1; then
  CONDA_BIN="${CONDA_BIN:-micromamba}"
elif command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="${CONDA_BIN:-mamba}"
fi

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Could not find conda, mamba, or micromamba. Load your ARC conda module first." >&2
  exit 1
fi

"${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

if [ "${CONDA_BIN}" = "micromamba" ]; then
  eval "$("${CONDA_BIN}" shell hook --shell bash)"
else
  # shellcheck disable=SC1091
  source "$("${CONDA_BIN}" info --base)/etc/profile.d/conda.sh"
fi

"${CONDA_BIN}" activate "${ENV_NAME}"
python -m pip install --upgrade pip wheel setuptools

python -m pip install \
  "${BONE_IMAGING_DERIVATIVES_SPEC}" \
  "${BONE_CONTOURING_SPEC}" \
  "${TIMELAPSED_HRPQCT_SPEC}" \
  "${BONE_MICROARCHITECTURE_SPEC}" \
  "${BONE_PLATE_ROD_THINNING_SPEC}" \
  "${PAROSOL_PY_SPEC}" \
  "${BONEMECHREG_SPEC}"

python - <<'PY'
import importlib

for module in (
    "bone_imaging_derivatives",
    "bone_contouring",
    "timelapsedhrpqct",
    "bone_microarchitecture",
    "plate_rod_thinning",
    "parosol_py",
    "bonemechreg",
):
    importlib.import_module(module)
    print(f"ok {module}")
PY

echo "Environment ready: ${ENV_NAME}"
