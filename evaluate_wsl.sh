#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SWARM_PYTHON:-}" ]]; then
  TASK_PYTHON="$SWARM_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  TASK_PYTHON="$CONDA_PREFIX/bin/python"
elif [[ -x "$HOME/miniconda3/envs/swarm-rescue/bin/python" ]]; then
  TASK_PYTHON="$HOME/miniconda3/envs/swarm-rescue/bin/python"
else
  TASK_PYTHON="$(command -v python3 || command -v python)"
fi
if [[ ! -x "$TASK_PYTHON" ]]; then
  echo "Python unavailable: $TASK_PYTHON; set SWARM_PYTHON to your project Python." >&2
  exit 1
fi
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8
export DISPLAY="${DISPLAY:-:0}"
if [[ $# -eq 0 ]]; then
  set -- --name "wsl_eval_$(date +%Y%m%d_%H%M%S)"     --cases Map01:historic Map05:historic --seed 12345 --contacts
fi
exec "$TASK_PYTHON" scripts/evaluate_generic_optimization.py "$@"
