#!/bin/bash
# The campaign of October 2026 on the LRZ serial cluster
# (next_steps/051026/cluster_campaign_051026.md): Figure 5's data-generating process,
# every estimator with C = 2I and with the rescaled C, for growing n.  One SLURM
# array per cell, each cell in its own folder:
#
#     runs/campaign/<cell>_n<n>/shards/shard_XXXX_of_YYYY.npz
#
# Every cell sees the same graphs (the seeds of the n-sweep baseline), so all
# results are paired, also with runs/nsweep_p10-20.
#
#   wave 1  direct loss, p = 10 and 20, 800 graphs per cell.  16 cells per n:
#           lasso / MCP / SCAD (standard path), MCP-up / SCAD-up (dense -> sparse),
#           MCP-lla / SCAD-lla (local linear approximation from the lasso),
#           adaptive (adaptive lasso), each with C2I and Cresc.  Every cell records
#           the path, the BIC-selected graph and the graph after the BIC search.
#   wave 2  search without a penalty and search started from the truth:
#           p = 10 with C2I and Cresc, p = 20 with Cresc.  3 cells per n.
#   wave 3  log-likelihood loss, p = 10, 400 graphs per cell.  8 cells per n:
#           lasso / MCP in both path orders (-up = dense -> sparse), C2I and Cresc.
#           Path and BIC-selected graph.
#
# From ~/repo on the login node (nothing needs to be activated first):
#     bash cluster/submit_campaign.sh --wave 1 --list            # the cells of a wave
#     bash cluster/submit_campaign.sh --wave 1 --dry-run         # print the sbatch calls
#     bash cluster/submit_campaign.sh --wave 1 --n 1000          # submit one sample size
#     bash cluster/submit_campaign.sh --wave 1 --status          # state of every cell
#     bash cluster/submit_campaign.sh --wave 1 --fill            # resubmit missing shards,
#                                                                # once the jobs have ended
# Options: --n takes one or more of 1000 1e4 1e5 inf (default: 1000 1e4 inf).
# --only PATTERN... restricts to cells whose name contains one of the patterns
# (e.g. --only MCP-up, --only Cresc).  --time HH:MM:SS overrides the time limit
# (e.g. with --fill after a TIMEOUT; the partition's cap is 24 h).  --shards N
# overrides the shard count of cells not yet submitted.
#
# A cell that has been submitted (its folder holds n_shards) is never submitted
# again except by --fill, and keeps its shard count.  If sbatch refuses a cell
# (LRZ allows about 200 queued tasks per user), nothing is recorded for it and the
# script stops: run the same command again later and it continues there.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=${CAMPAIGN_ROOT:-runs/campaign}         # override only for tests
BATCH=cluster/campaign_array.sbatch
NS=(1000 1e4 inf)
WAVE=""
ONLY=()
DRY=0
FILL=0
STATUS=0
LIST=0
TIME=""
SHARDS=""

# The cells of a wave, one per line:
#     name | job tag | runner | shards | time limit | runner arguments (without --n-obs)
# Shards are sized so that a task takes about 1 to 3 hours (costs measured in the
# n-sweep, in the timing pilots of the campaign note, Section 3.4, and in the
# rehearsal of Section 4.5).  In wave 1 the BIC search costs more per graph at
# p = 20 than a lasso path, so even the cheap paths get 4 shards; the standard
# MCP / SCAD paths get 8; a dense -> sparse path costs 2 to 5 standard paths,
# hence 16.
cells() {
  local c cs ci s1 s3 run=simulations/run_s1_shard.py
  case "$1" in
    1)
      s1="--p 10 20 --reps 25 --select search"
      for c in C2I Cresc; do
        if [ "$c" = C2I ]; then cs=identity; ci=i; else cs=variance; ci=r; fi
        echo "direct_lasso_${c}|1la${ci}|$run|4|12:00:00|$s1 --c-scale $cs --penalty lasso"
        echo "direct_MCP_${c}|1Ms${ci}|$run|8|12:00:00|$s1 --c-scale $cs --penalty MCP"
        echo "direct_SCAD_${c}|1Ss${ci}|$run|8|12:00:00|$s1 --c-scale $cs --penalty SCAD"
        echo "direct_MCP-up_${c}|1Mu${ci}|$run|16|12:00:00|$s1 --c-scale $cs --penalty MCP --direction up"
        echo "direct_SCAD-up_${c}|1Su${ci}|$run|16|12:00:00|$s1 --c-scale $cs --penalty SCAD --direction up"
        echo "direct_MCP-lla_${c}|1Ml${ci}|$run|4|12:00:00|$s1 --c-scale $cs --penalty MCP --method lla"
        echo "direct_SCAD-lla_${c}|1Sl${ci}|$run|4|12:00:00|$s1 --c-scale $cs --penalty SCAD --method lla"
        echo "direct_adaptive_${c}|1ad${ci}|$run|4|12:00:00|$s1 --c-scale $cs --method adaptive"
      done ;;
    2)
      run=simulations/run_search_shard.py
      echo "search_p10_C2I|2p10i|$run|2|12:00:00|--p 10 --reps 25 --c-scale identity"
      echo "search_p10_Cresc|2p10r|$run|2|12:00:00|--p 10 --reps 25 --c-scale variance"
      echo "search_p20_Cresc|2p20r|$run|16|12:00:00|--p 20 --reps 25 --c-scale variance" ;;
    3)
      s3="--loss loglik --p 10 --reps 25 --select bic"
      for c in C2I Cresc; do
        if [ "$c" = C2I ]; then cs=identity; ci=i; else cs=variance; ci=r; fi
        echo "loglik_lasso_${c}|3la${ci}|$run|8|24:00:00|$s3 --c-scale $cs --penalty lasso"
        echo "loglik_lasso-up_${c}|3lu${ci}|$run|8|24:00:00|$s3 --c-scale $cs --penalty lasso --direction up"
        echo "loglik_MCP_${c}|3Ms${ci}|$run|4|24:00:00|$s3 --c-scale $cs --penalty MCP"
        echo "loglik_MCP-up_${c}|3Mu${ci}|$run|8|24:00:00|$s3 --c-scale $cs --penalty MCP --direction up"
      done ;;
    *) echo "unknown wave: $1 (1, 2 or 3)" >&2; return 2 ;;
  esac
}

# one character for the sample size, so that job names stay under LRZ's 10 characters
n_code() { case "$1" in 1000) echo 3 ;; 1e4) echo 4 ;; 1e5) echo 5 ;; inf) echo i ;; *) echo x ;; esac; }

while [ $# -gt 0 ]; do
  case "$1" in
    --wave) WAVE=${2:?--wave needs 1, 2 or 3}; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --fill) FILL=1; shift ;;
    --status) STATUS=1; shift ;;
    --list) LIST=1; shift ;;
    --time) TIME=${2:?--time needs HH:MM:SS}; shift 2 ;;
    --shards)
      SHARDS=${2:?--shards needs a number}; shift 2
      [[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "--shards needs a positive integer" >&2; exit 2; } ;;
    --n|--only)
      opt=$1; shift; vals=()
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do vals+=("$1"); shift; done
      [ ${#vals[@]} -gt 0 ] || { echo "$opt needs at least one value" >&2; exit 2; }
      case "$opt" in
        --n) NS=("${vals[@]}") ;;
        --only) ONLY=("${vals[@]}") ;;
      esac ;;
    -h|--help) sed -n '2,39p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ -n "$WAVE" ] || { echo "--wave is required (1, 2 or 3); see --help" >&2; exit 2; }
CELLS=$(cells "$WAVE") || exit 2
for n in "${NS[@]}"; do
  [ "$(n_code "$n")" != x ] || { echo "unknown sample size: $n (1000, 1e4, 1e5 or inf)" >&2; exit 2; }
done

wanted() {                                    # does the cell name pass --only?
  [ ${#ONLY[@]} -eq 0 ] && return 0
  local pat
  for pat in "${ONLY[@]}"; do [[ "$1" == *"$pat"* ]] && return 0; done
  return 1
}

if [ "$LIST" -eq 1 ]; then
  printf '%-24s %-7s %-6s %-9s %s\n' cell job shards time "runner and arguments"
  while IFS='|' read -r name tag runner shards time args; do
    wanted "$name" || continue
    printf '%-24s %-7s %-6s %-9s %s\n' "$name" "$tag" "$shards" "$time" "$(basename "$runner") $args"
  done <<< "$CELLS"
  exit 0
fi

n_cells=0
n_tasks=0
for n in "${NS[@]}"; do
  while IFS='|' read -r name tag runner shards time args; do
    wanted "$name" || continue
    run="$ROOT/${name}_n${n}"
    shards=${SHARDS:-$shards}
    if [ -f "$run/n_shards" ]; then
      prev=$(cat "$run/n_shards")
      missing=""
      for ((i = 0; i < prev; i++)); do
        f=$(printf '%s/shards/shard_%04d_of_%04d.npz' "$run" "$i" "$prev")
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
      count=$n_missing
    else
      if [ "$FILL" -eq 1 ] || [ "$STATUS" -eq 1 ]; then
        echo "not started $run"
        continue
      fi
      prev=$shards
      array="0-$((shards - 1))"
      count=$shards
    fi
    read -r -a argv <<< "$args"
    cmd=(sbatch --array="$array" --time="${TIME:-$time}" -J "${tag}$(n_code "$n")"
         "$BATCH" "$runner" "$run" "${argv[@]}" --n-obs "$n")
    if [ "$DRY" -eq 1 ]; then
      echo "${cmd[*]}"
    else
      mkdir -p "$run" logs
      new=0
      [ -f "$run/n_shards" ] || { echo "$prev" > "$run/n_shards"; new=1; }
      if ! "${cmd[@]}" < /dev/null; then
        [ "$new" -eq 1 ] && rm -f "$run/n_shards"
        echo "!! sbatch failed for $run -- nothing recorded for this cell." >&2
        echo "!! $n_tasks tasks in $n_cells cells were submitted before it; run the same command again later to continue." >&2
        exit 1
      fi
    fi
    n_cells=$((n_cells + 1))
    n_tasks=$((n_tasks + count))
  done <<< "$CELLS"
done
if [ "$STATUS" -eq 0 ]; then
  verb=submitted
  [ "$DRY" -eq 1 ] && verb="would submit"
  echo "$verb $n_tasks tasks in $n_cells cells"
fi
