#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The host may source ROS globally. Keep its Python and native libraries out of this venv.
unset PYTHONPATH
unset LD_LIBRARY_PATH
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
source "${PROJECT_ROOT}/.venv/bin/activate"
