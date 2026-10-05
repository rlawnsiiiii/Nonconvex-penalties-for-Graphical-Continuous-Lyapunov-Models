#!/bin/bash
#SBATCH -J gclm-camp
#SBATCH -D ./
#SBATCH -o ./logs/camp_%A_%a.out
#SBATCH -e ./logs/camp_%A_%a.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#
# One cell of the campaign of October 2026 as a SLURM job array
# (next_steps/051026/cluster_campaign_051026.md).  Not called by hand: use
# cluster/submit_campaign.sh, which sets --array, --time and the job name, and
# writes <run>/n_shards before submitting.
#
#   sbatch [--array=...] cluster/campaign_array.sbatch <runner.py> <run dir> [runner args...]
#
#   <runner.py>   simulations/run_s1_shard.py      (an estimator: path, BIC, search)
#                 simulations/run_search_shard.py  (search without a penalty)
#   <run dir>     the cell's folder; shards go to <run dir>/shards
#
# Each array task is one core and writes one .npz of numbers; no plots.  Arguments,
# not environment variables: with --export=NONE the job inherits no environment.
#
# Memory: the direct-loss solver is matrix-free (70 MB at p = 50); the BIC refits
# hold the p^2 x p^2 Gram matrix once (50 MB at p = 50).  2G is generous.

set -euo pipefail
module load slurm_setup

# --export=NONE starts the job with a clean environment: load the same Python
# module the venv was built on (cluster/setup_env.sh), then activate the venv.
module load python 2>/dev/null || module load anaconda3 2>/dev/null || {
  echo "!! no python module found; see docs/REPRODUCTION.md Section 2.1" >&2
  exit 1
}
VENV="$HOME/venvs/gclm"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "python=$(command -v python) $(python --version 2>&1)"

# One core per task: keep NumPy/SciPy's BLAS single-threaded.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUNNER="${1:?usage: campaign_array.sbatch <runner.py> <run dir> [runner args...]}"
RUN_DIR="${2:?usage: campaign_array.sbatch <runner.py> <run dir> [runner args...]}"
shift 2
[ -f "$RUNNER" ] || { echo "!! no such runner: $RUNNER" >&2; exit 1; }
OUT_DIR="$RUN_DIR/shards"
mkdir -p "$OUT_DIR"

# The shard count is read from the run folder, not from the array size, so that
# resubmitting a few failed shards (--array=3,17) slices the grid the same way.
[ -f "$RUN_DIR/n_shards" ] || { echo "!! $RUN_DIR/n_shards is missing; submit with cluster/submit_campaign.sh" >&2; exit 1; }
N_SHARDS="$(cat "$RUN_DIR/n_shards")"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_SHARDS}" ]; then
  echo "!! array index ${SLURM_ARRAY_TASK_ID} >= ${N_SHARDS} shards for $RUN_DIR" >&2
  exit 1
fi

echo "host=$(hostname) runner=$RUNNER run=$RUN_DIR shard=${SLURM_ARRAY_TASK_ID}/${N_SHARDS} start=$(date -Is)"

srun python "$RUNNER" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${N_SHARDS}" \
    --out-dir "${OUT_DIR}" ${1+"$@"}

echo "done=$(date -Is)"
