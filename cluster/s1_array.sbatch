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
#   mkdir -p logs
#   sbatch cluster/s1_array.sbatch                              # -> runs/s1_dettling_reproduction
#   sbatch cluster/s1_array.sbatch runs/s1b_mcp --penalty MCP   # a new run, its own folder
#   sbatch cluster/s1_array.sbatch runs/s2_loglik_scad --loss loglik --penalty SCAD
#
# Argument 1 is the run directory; shards go to <run>/s1_shards.  If that folder
# holds a file n_shards, it sets the shard count (default 64; see
# cluster/submit_nsweep.sh, which also sets --array and --time).  Any further
# arguments are passed through to simulations/run_s1_shard.py: --penalty,
# --gamma, --convention, --loss, --direction, --p, --reps, ...  (Arguments,
# not environment variables: with --export=NONE the job inherits no environment.)
# One submission per (loss, penalty) combination; see docs/REPRODUCTION.md
# Section 4 for the list and the cost of each.
#
# Memory: one worker peaks around 70 MB even at p=50 (the solver is matrix-free),
# so 2G is generous.  Time: the slowest shard should finish well inside 2 h;
# 6 h leaves room for the long tail and for a busy node.

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

# One core per task: keep NumPy/SciPy's BLAS single-threaded so a task never
# starts more threads than its one core (the local pilot scripts do the same).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN_DIR="${1:-runs/s1_dettling_reproduction}"
[ "$#" -gt 0 ] && shift
OUT_DIR="$RUN_DIR/s1_shards"
mkdir -p "$OUT_DIR"

# Shard count: 64, unless the run folder says otherwise (cluster/submit_nsweep.sh
# writes <run>/n_shards before submitting, together with a matching --array).  It
# is read from the folder rather than from the array size so that resubmitting a
# few failed shards (--array=3,17) still slices the grid the same way.
N_SHARDS=64
[ -f "$RUN_DIR/n_shards" ] && N_SHARDS="$(cat "$RUN_DIR/n_shards")"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_SHARDS}" ]; then
  echo "!! array index ${SLURM_ARRAY_TASK_ID} >= ${N_SHARDS} shards for $RUN_DIR" >&2
  exit 1
fi

echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID}/${N_SHARDS} start=$(date -Is)"

srun python simulations/run_s1_shard.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${N_SHARDS}" \
    --out-dir "${OUT_DIR}" ${1+"$@"}

echo "done=$(date -Is)"
