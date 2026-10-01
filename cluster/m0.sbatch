#!/bin/bash
#SBATCH -J gclm-m0
#SBATCH -D ./
#SBATCH -o ./logs/m0_%j.out
#SBATCH -e ./logs/m0_%j.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#
# Figure 3 (Dettling Example 2) -- path vs 5-cycle across sample sizes.
# Small enough to be a single serial job (~5 min at 100 reps).
#
#   mkdir -p logs
#   sbatch cluster/m0.sbatch                      # -> runs/s1_dettling_reproduction
#   sbatch cluster/m0.sbatch runs/<other_run>     # argument 1: the run directory

set -euo pipefail
module load slurm_setup

# --export=NONE starts the job with a clean environment: nothing from the
# submitting shell -- no loaded modules, no active venv -- carries over.  Load
# the same Python module the venv was built on (cluster/setup_env.sh), then
# activate the venv.  Without the module, a venv built on a module's Python can
# fail at start-up with a missing libpython.
module load python 2>/dev/null || module load anaconda3 2>/dev/null || {
  echo "!! no python module found; see docs/REPRODUCTION.md Section 2.1" >&2
  exit 1
}
VENV="$HOME/venvs/gclm"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "python=$(command -v python) $(python --version 2>&1)"

RUN_DIR="${1:-runs/s1_dettling_reproduction}"
mkdir -p "$RUN_DIR"

echo "host=$(hostname) start=$(date -Is)"
srun python simulations/run_m0.py --reps 100 --out "$RUN_DIR/m0_reps100.csv"
echo "done=$(date -Is)"
