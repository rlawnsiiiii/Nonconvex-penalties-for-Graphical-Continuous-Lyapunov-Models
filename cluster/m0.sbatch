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
#   mkdir -p logs results
#   sbatch cluster/m0.sbatch

set -euo pipefail
module load slurm_setup

VENV="${GCLM_VENV:-$HOME/venvs/gclm}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "host=$(hostname) start=$(date -Is)"
srun python simulations/run_m0.py --reps 100 --out results/m0_reps100.csv
echo "done=$(date -Is)"
