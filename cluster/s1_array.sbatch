#!/bin/bash
#SBATCH -J gclm-s1
#SBATCH -D ./
#SBATCH -o ./logs/s1_%A_%a.out
#SBATCH -e ./logs/s1_%A_%a.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --array=0-63
#
# Figure 5 (Dettling Section 5) -- full grid, as a SLURM job array.
#
#   7 p-values x 4 k x 4 C-choices x 100 reps = 11,200 datasets
#   split into 64 shards of 175 datasets each.
#
# Each array task is a single core and writes ONE .npz of numbers.  No plots are
# produced here -- plotting happens locally from the aggregated output.
#
#   mkdir -p logs results/s1
#   sbatch cluster/s1_array.sbatch
#
# Memory: one worker peaks around 70 MB even at p=50 (the solver is matrix-free),
# so 2G is generous.  Time: the slowest shard should finish well inside 2 h;
# 6 h leaves room for the long tail and for a busy node.

set -euo pipefail
module load slurm_setup

VENV="${GCLM_VENV:-$HOME/venvs/gclm}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

N_SHARDS=64
OUT_DIR="${GCLM_OUT:-results/s1}"
mkdir -p "$OUT_DIR"

echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID}/${N_SHARDS} start=$(date -Is)"

srun python simulations/run_s1_shard.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${N_SHARDS}" \
    --out-dir "${OUT_DIR}"

echo "done=$(date -Is)"
