#!/bin/bash
# n-sweep on the LRZ serial cluster: Figure 5's DGP at p = 10, 20 for growing n,
# every loss x penalty.  4 k x 4 C x 25 reps = 800 datasets per cell, one SLURM
# array per cell (loss, penalty, n), each in its own run folder:
#
#     runs/nsweep_p10-20/<loss>_<penalty>_n<n>/s1_shards/shard_XXXX_of_YYYY.npz
#
# Every cell sees the same 800 drift matrices: M* and C depend only on (seed, p,
# k, C choice, rep) and are drawn before the data, so the runs are paired across
# losses, penalties and n.  n = inf feeds the population covariance.
#
# From ~/repo on the login node (nothing needs to be activated first):
#     bash cluster/submit_nsweep.sh --dry-run          # print what would be submitted
#     bash cluster/submit_nsweep.sh --n 1000           # first wave: checks + timings
#     bash cluster/submit_nsweep.sh --n 1e4 1e5 inf    # the rest
#     bash cluster/submit_nsweep.sh --status           # state of every cell; submits nothing
#     bash cluster/submit_nsweep.sh --fill             # resubmit missing shards, only
#                                                      # once that cell's jobs have ended
# --n, --loss and --penalty select a subset (one or more values each); --time
# HH:MM:SS overrides the per-loss limit, e.g. with --fill after a TIMEOUT.  A cell
# that has been submitted (its folder holds n_shards) is never submitted again
# except by --fill.  docs/REPRODUCTION.md Section 2.6.
set -euo pipefail
cd "$(dirname "$0")/.."

NS=(1000 1e4 1e5 inf)
LOSSES=(direct loglik frobenius)
PENALTIES=(lasso MCP SCAD)
ROOT=${NSWEEP_ROOT:-runs/nsweep_p10-20}      # override only for tests
DRY=0
FILL=0
STATUS=0
TIME=""

# Shards and wall-time limit per loss.  Sized from the p = 10 costs in
# simulations/S2_penalties_losses.md Section 4 with p = 20 taken as ~10x (one
# measured p = 20 dataset: 5-8x), so the limits are generous; the n = 1000 wave
# measures the real cost (sacct) before the other waves go in.
shards_for() { case "$1" in direct) echo 16 ;; loglik) echo 32 ;; frobenius) echo 64 ;; esac; }
time_for()   { case "$1" in direct) echo 06:00:00 ;; loglik) echo 24:00:00 ;; frobenius) echo 48:00:00 ;; esac; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --fill) FILL=1; shift ;;
    --status) STATUS=1; shift ;;
    --time) TIME=${2:?--time needs HH:MM:SS}; shift 2 ;;
    --n|--loss|--penalty)
      opt=$1; shift; vals=()
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do vals+=("$1"); shift; done
      [ ${#vals[@]} -gt 0 ] || { echo "$opt needs at least one value" >&2; exit 2; }
      case "$opt" in
        --n) NS=("${vals[@]}") ;;
        --loss) LOSSES=("${vals[@]}") ;;
        --penalty) PENALTIES=("${vals[@]}") ;;
      esac ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

for n in "${NS[@]}"; do
  for loss in "${LOSSES[@]}"; do
    shards=$(shards_for "$loss")
    [ -n "$shards" ] || { echo "unknown loss: $loss" >&2; exit 2; }
    for pen in "${PENALTIES[@]}"; do
      run="$ROOT/${loss}_${pen}_n${n}"
      if [ -f "$run/n_shards" ]; then
        prev=$(cat "$run/n_shards")
        missing=""
        for ((i = 0; i < prev; i++)); do
          f=$(printf '%s/s1_shards/shard_%04d_of_%04d.npz' "$run" "$i" "$prev")
          [ -f "$f" ] || missing="${missing:+$missing,}$i"
        done
        if [ -z "$missing" ]; then
          echo "complete    $run"
          continue
        fi
        n_missing=$(echo "$missing" | tr ',' '\n' | wc -l | tr -d ' ')
        if [ "$FILL" -eq 0 ] || [ "$STATUS" -eq 1 ]; then
          echo "submitted   $run  ($((prev - n_missing))/$prev shards written)"
          continue
        fi
        array="$missing"
      else
        if [ "$FILL" -eq 1 ] || [ "$STATUS" -eq 1 ]; then
          echo "not started $run"
          continue
        fi
        prev=$shards
        array="0-$((shards - 1))"
      fi
      job="n${loss:0:1}${pen:0:1}${n}"          # LRZ asks for names under 10 characters
      cmd=(sbatch --array="$array" --time="${TIME:-$(time_for "$loss")}" -J "$job"
           cluster/s1_array.sbatch "$run"
           --p 10 20 --reps 25 --n-obs "$n" --loss "$loss" --penalty "$pen")
      if [ "$DRY" -eq 1 ]; then
        echo "${cmd[*]}"
      else
        mkdir -p "$run" logs
        new=0
        [ -f "$run/n_shards" ] || { echo "$prev" > "$run/n_shards"; new=1; }
        if ! "${cmd[@]}"; then
          [ "$new" -eq 1 ] && rm -f "$run/n_shards"
          echo "!! sbatch failed for $run -- nothing recorded; rerun the script to retry" >&2
          exit 1
        fi
      fi
    done
  done
done
