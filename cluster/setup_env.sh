#!/bin/bash
# One-time environment setup on the LRZ Linux Cluster.
# Run this ONCE on a login node (cool.hpc.lrz.de) before submitting any job.
#
#   bash cluster/setup_env.sh
#
# Creates a venv at $HOME/venvs/gclm with numpy + scipy and installs this
# package.  The batch scripts activate it; nothing else is needed, because the
# default solver ("fista") is pure Python/NumPy -- no R, no glmnet, no ncvreg.
set -euo pipefail

# LRZ: modules are not auto-loaded in batch context, and `module` needs sourcing
source /etc/profile.d/modules.sh 2>/dev/null || true

# Adjust if `module avail python` shows a different name on your segment.
module load python 2>/dev/null || module load anaconda3 2>/dev/null || {
  echo "!! no python module found. Run 'module avail python' and edit this file." >&2
  exit 1
}

VENV="${GCLM_VENV:-$HOME/venvs/gclm}"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install "numpy>=1.24" "scipy>=1.10"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "repo: $REPO"
python - <<PY
import numpy, scipy, sys
print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__)
PY
echo
echo "OK. venv at $VENV"
echo "Next: sbatch cluster/s1_array.sbatch"
