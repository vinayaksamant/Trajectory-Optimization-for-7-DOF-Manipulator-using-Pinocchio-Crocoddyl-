#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The host may source ROS globally. Keep its Python packages out of this venv.
unset PYTHONPATH
source "${PROJECT_ROOT}/.venv/bin/activate"
