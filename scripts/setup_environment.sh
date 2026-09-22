#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
MODEL_CACHE_PATH="${PROJECT_ROOT}/.cache/mujoco_menagerie"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"

# Avoid leaking ROS/system Python packages and native libraries into this project.
unset PYTHONPATH
unset LD_LIBRARY_PATH
mkdir -p "${MPLCONFIGDIR}"

python3 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install -e "${PROJECT_ROOT}[robotics,simulation,dev]"
"${VENV_PATH}/bin/python" -c \
  "import mujoco_menagerie as mm; mm.prefetch(['franka_emika_panda'], cache=mm.Cache(dir=r'${MODEL_CACHE_PATH}'))"

"${VENV_PATH}/bin/python" -c \
  "import crocoddyl, mujoco, pinocchio; print('Environment ready:', 'Pinocchio', pinocchio.__version__, '| Crocoddyl', crocoddyl.__version__, '| MuJoCo', mujoco.__version__)"
