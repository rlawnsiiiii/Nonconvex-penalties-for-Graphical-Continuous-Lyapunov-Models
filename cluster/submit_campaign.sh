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
#           the path, the graph selected by the score and the graph after the greedy search.
#   wave 2  search from random graphs and search started from the truth:
#           p = 10 with C2I and Cresc, p = 20 with Cresc.  3 cells per n.
#   wave 3  log-likelihood loss, p = 10, 400 graphs per cell.  8 cells per n:
#           lasso / MCP in both path orders (-up = dense -> sparse), C2I and Cresc.
#           Path and the graph selected by the score.
#   wave 4  larger p for the thesis figure: p = 15, 25, 30, 40, 50 at n = 1000
#           (Figure 5's setting), 400 graphs per cell.  12 cells per p: lasso / MCP /
#           SCAD (standard path), MCP-up / SCAD-up, adaptive, with C2I and Cresc.
#           Path and the graph selected by the score, no search (too slow at this size).  Cells
#           are named <...>_p<p>_n1000; --p selects the sizes.  --n is 1000 unless
#           given (e.g. --n inf --only Cresc for the population version of a subset).
#   wave 5  two checks of the selection step, p = 10 (decided on 7 October):
#           (a) the score with the maximised likelihood instead of the least-squares
#               refit (--refit loglik), with the search, for lasso / MCP-up / adaptive
#               and both C: cells direct_<estimator>-ml_<C>, 6 per n;
#           (b) the pure search with 100 random starting graphs instead of 10, drawn
#               sparse as before or uniformly (--starts), rescaled C: cells
#               search100s_p10_Cresc and search100u_p10_Cresc; and, optional and
#               expensive, 30 sparse starts at p = 20 on 5 replicates (search30s_p20_Cresc);
#           (c) the eBIC term (Dettling's 4 gamma |E| log p, gamma = 0.5 and 1)
#               inside the selection and the search, next to the BIC penalty on the same
#               path (--ebic-gamma), for lasso / MCP-up / adaptive, p = 10 and 20, both C:
#               cells direct_<estimator>-ebic_<C>; and the pure search scored with
#               gamma = 1: searche1_p10_C2I, searche1_p10_Cresc, searche1_p20_Cresc;
#           and (9 October) 300 sparse starting graphs at p = 10: search300s_p10_Cresc.
#   wave 6  (9 October) every wave 1 cell rescored with the eBIC penalty (gamma = 1)
#           inside the selection and the search, from its stored supports: no path is
#           recomputed (simulations/rescore_shard.py).  Cells rescore1_<source cell>, the
#           source's shard count; campaign.py overlays them on the source cell's rows.
#   wave 7  (10 October) (a) the log-likelihood loss over p = 10 and 20, both C, with the
#           selection and the search of waves 1 to 4 (least-squares refit): lasso, MCP sparse ->
#           dense, MCP dense -> sparse from the exact fit and from the lasso solution, adaptive
#           lasso.  At p = 10 only the two that wave 3 lacks (loglik_<estimator>_<C>); at p = 20
#           all five (loglik_<estimator>_<C>_p20).  p = 30 is left out for cost for now.
#   wave 8  (10 October) the likelihood refit.  (b) The stored paths rescored with the Gaussian
#           likelihood maximised on every graph (simulations/rescore_shard.py --refit loglik):
#           selection and search at p = 10 (the log-likelihood cells of waves 3 and 7, the
#           direct-loss cells of wave 1), the selection only at p = 20 (wave 7's cells);
#           cells <loss>_<estimator>-ml_<C>[_p<p>].  Run it after wave 7 is complete.
#           (c) The search from 100 random graphs, the empty graph and the truth with the
#           likelihood refit, p = 10 (search100sml_p10_<C>).
#
# From ~/repo on the login node (nothing needs to be activated first):
#     bash cluster/submit_campaign.sh --wave 1 --list            # the cells of a wave
#     bash cluster/submit_campaign.sh --wave 1 --dry-run         # print the sbatch calls
#     bash cluster/submit_campaign.sh --wave 1 --n 1000          # submit one sample size
#     bash cluster/submit_campaign.sh --wave 4 --p 15 25         # wave 4: one or more p
#     bash cluster/submit_campaign.sh --wave 5 --n 1e4 --only -ml # wave 5a at one n
#     bash cluster/submit_campaign.sh --wave 1 --status          # state of every cell
#     bash cluster/submit_campaign.sh --wave 1 --fill            # resubmit missing shards,
#                                                                # once the jobs have ended
# Options: --n takes one or more of 1000 1e4 1e5 inf (default: 1000 1e4 inf; for
# wave 4 the default is 1000 alone).  --p takes one or more of 15 25 30 40 50 (wave 4 only; default
# all five).  --only PATTERN... restricts to cells whose name contains one of the
# patterns (e.g. --only MCP-up, --only Cresc).  --time HH:MM:SS overrides the time
# limit (e.g. with --fill after a TIMEOUT; the partition's cap is 24 h).  --shards N
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
NS_GIVEN=0
PS=(15 25 30 40 50)
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
# rehearsal of Section 4.5).  In wave 1 the greedy search costs more per graph at
# p = 20 than a lasso path, so even the cheap paths get 4 shards; the standard
# MCP / SCAD paths get 8; a dense -> sparse path costs 2 to 5 standard paths,
# hence 16.  Wave 4 scales the shards with p (a path costs about p^3: 36 s per
# graph for the lasso at p = 50 against 6 s at p = 20) and allows 24 h from
# p = 30 on.  Wave 5a: the likelihood refit costs 15 to 90 s per graph at p = 10
# on the laptop (the search with every add move scored), hence 8 shards.  Wave 5b:
# 100 starts cost about ten times the 11 of wave 2 (9 to 15 s per graph there).
# Wave 5c computes the paths again and searches three times per graph, hence 8
# shards for the cheap paths.
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
    4)
      local p
      for p in "${PS[@]}"; do
        s4="--p $p --reps 25 --select bic"
        if [ "$p" -ge 40 ]; then sh=(8 16 16 32 32 8); t4=24:00:00
        elif [ "$p" -ge 30 ]; then sh=(4 8 8 16 16 4); t4=24:00:00
        else sh=(4 8 8 16 16 4); t4=12:00:00; fi
        for c in C2I Cresc; do
          if [ "$c" = C2I ]; then cs=identity; ci=i; else cs=variance; ci=r; fi
          echo "direct_lasso_${c}_p${p}|4la${ci}${p}|$run|${sh[0]}|$t4|$s4 --c-scale $cs --penalty lasso"
          echo "direct_MCP_${c}_p${p}|4Ms${ci}${p}|$run|${sh[1]}|$t4|$s4 --c-scale $cs --penalty MCP"
          echo "direct_SCAD_${c}_p${p}|4Ss${ci}${p}|$run|${sh[2]}|$t4|$s4 --c-scale $cs --penalty SCAD"
          echo "direct_MCP-up_${c}_p${p}|4Mu${ci}${p}|$run|${sh[3]}|$t4|$s4 --c-scale $cs --penalty MCP --direction up"
          echo "direct_SCAD-up_${c}_p${p}|4Su${ci}${p}|$run|${sh[4]}|$t4|$s4 --c-scale $cs --penalty SCAD --direction up"
          echo "direct_adaptive_${c}_p${p}|4ad${ci}${p}|$run|${sh[5]}|$t4|$s4 --c-scale $cs --method adaptive"
        done
      done ;;
    5)
      s5="--p 10 --reps 25 --select search --refit loglik"
      for c in C2I Cresc; do
        if [ "$c" = C2I ]; then cs=identity; ci=i; else cs=variance; ci=r; fi
        echo "direct_lasso-ml_${c}|5la${ci}|$run|8|12:00:00|$s5 --c-scale $cs --penalty lasso"
        echo "direct_MCP-up-ml_${c}|5Mu${ci}|$run|8|12:00:00|$s5 --c-scale $cs --penalty MCP --direction up"
        echo "direct_adaptive-ml_${c}|5ad${ci}|$run|8|12:00:00|$s5 --c-scale $cs --method adaptive"
      done
      run=simulations/run_search_shard.py
      echo "search100s_p10_Cresc|5s10r|$run|8|12:00:00|--p 10 --reps 25 --c-scale variance --methods pure --restarts 100 --starts sparse"
      echo "search100u_p10_Cresc|5u10r|$run|8|12:00:00|--p 10 --reps 25 --c-scale variance --methods pure --restarts 100 --starts uniform"
      echo "search30s_p20_Cresc|5s20r|$run|32|24:00:00|--p 20 --reps 5 --c-scale variance --methods pure --restarts 30 --starts sparse"
      echo "search300s_p10_Cresc|5t10r|$run|16|12:00:00|--p 10 --reps 25 --c-scale variance --methods pure --restarts 300 --starts sparse"
      run=simulations/run_s1_shard.py
      s5c="--p 10 20 --reps 25 --select search --ebic-gamma 0.5 1"
      for c in C2I Cresc; do
        if [ "$c" = C2I ]; then cs=identity; ci=i; else cs=variance; ci=r; fi
        echo "direct_lasso-ebic_${c}|5le${ci}|$run|8|12:00:00|$s5c --c-scale $cs --penalty lasso"
        echo "direct_MCP-up-ebic_${c}|5Me${ci}|$run|16|12:00:00|$s5c --c-scale $cs --penalty MCP --direction up"
        echo "direct_adaptive-ebic_${c}|5ae${ci}|$run|8|12:00:00|$s5c --c-scale $cs --method adaptive"
      done
      run=simulations/run_search_shard.py
      echo "searche1_p10_C2I|5e10i|$run|2|12:00:00|--p 10 --reps 25 --c-scale identity --ebic-gamma 1"
      echo "searche1_p10_Cresc|5e10r|$run|2|12:00:00|--p 10 --reps 25 --c-scale variance --ebic-gamma 1"
      echo "searche1_p20_Cresc|5e20r|$run|16|12:00:00|--p 20 --reps 25 --c-scale variance --ebic-gamma 1" ;;
    6)
      run=simulations/rescore_shard.py
      s6="--select search --ebic-gamma 1"
      for c in C2I Cresc; do
        if [ "$c" = C2I ]; then ci=i; else ci=r; fi
        echo "rescore1_direct_lasso_${c}|6la${ci}|$run|4|12:00:00|$s6 --source-cell direct_lasso_${c}"
        echo "rescore1_direct_MCP_${c}|6Ms${ci}|$run|8|12:00:00|$s6 --source-cell direct_MCP_${c}"
        echo "rescore1_direct_SCAD_${c}|6Ss${ci}|$run|8|12:00:00|$s6 --source-cell direct_SCAD_${c}"
        echo "rescore1_direct_MCP-up_${c}|6Mu${ci}|$run|16|12:00:00|$s6 --source-cell direct_MCP-up_${c}"
        echo "rescore1_direct_SCAD-up_${c}|6Su${ci}|$run|16|12:00:00|$s6 --source-cell direct_SCAD-up_${c}"
        echo "rescore1_direct_MCP-lla_${c}|6Ml${ci}|$run|4|12:00:00|$s6 --source-cell direct_MCP-lla_${c}"
        echo "rescore1_direct_SCAD-lla_${c}|6Sl${ci}|$run|4|12:00:00|$s6 --source-cell direct_SCAD-lla_${c}"
        echo "rescore1_direct_adaptive_${c}|6ad${ci}|$run|4|12:00:00|$s6 --source-cell direct_adaptive_${c}"
      done ;;
    7)
      # (a) the log-likelihood loss over p, selection and search with the least-squares refit as
      # in waves 1 to 4.  p = 10: the two estimators wave 3 lacks; p = 20: all five.
      # Shards per estimator (lasso, MCP, MCP-up, MCP-up-lasso, adaptive) keep a task under ~6 h.
      local s7 p pc sh
      for c in C2I Cresc; do
        if [ "$c" = C2I ]; then cs=identity; ci=i; else cs=variance; ci=r; fi
        s7="--loss loglik --reps 25 --select search --c-scale $cs"
        if [ "$c" = C2I ]; then sh=(8 8); else sh=(16 16); fi
        echo "loglik_MCP-up-lasso_${c}|7ML${ci}1|$run|${sh[0]}|24:00:00|$s7 --p 10 --penalty MCP --direction up --up-start lasso"
        echo "loglik_adaptive_${c}|7ad${ci}1|$run|${sh[1]}|24:00:00|$s7 --p 10 --method adaptive"
        for p in 20; do        # p = 30 left out for cost (10 October); its shards were
          pc=${p:0:1}          # C2I (32 32 32 48 32), Cresc (48 40 64 100 56)
          case "$c$p" in
            C2I20) sh=(8 8 12 16 16) ;;
            Cresc20) sh=(24 24 24 32 32) ;;
          esac
          echo "loglik_lasso_${c}_p${p}|7la${ci}${pc}|$run|${sh[0]}|24:00:00|$s7 --p $p --penalty lasso"
          echo "loglik_MCP_${c}_p${p}|7Ms${ci}${pc}|$run|${sh[1]}|24:00:00|$s7 --p $p --penalty MCP"
          echo "loglik_MCP-up_${c}_p${p}|7Mu${ci}${pc}|$run|${sh[2]}|24:00:00|$s7 --p $p --penalty MCP --direction up"
          echo "loglik_MCP-up-lasso_${c}_p${p}|7ML${ci}${pc}|$run|${sh[3]}|24:00:00|$s7 --p $p --penalty MCP --direction up --up-start lasso"
          echo "loglik_adaptive_${c}_p${p}|7ad${ci}${pc}|$run|${sh[4]}|24:00:00|$s7 --p $p --method adaptive"
        done
      done ;;
    8)
      # (b) the likelihood refit behind the score on stored paths (rescore_shard.py --refit
      # loglik; one task per source shard, so the shard counts are the sources'): selection and
      # search at p = 10, the selection only at p = 20.  Needs the source cells complete.
      run=simulations/rescore_shard.py
      local r8s="--refit loglik --select search" r8b="--refit loglik --select bic" e p pc sh
      for c in C2I Cresc; do
        if [ "$c" = C2I ]; then ci=i; sh=(8 8); else ci=r; sh=(16 16); fi
        echo "loglik_lasso-ml_${c}|8ll${ci}1|$run|8|24:00:00|$r8s --source-cell loglik_lasso_${c}"
        echo "loglik_lasso-up-ml_${c}|8lu${ci}1|$run|8|24:00:00|$r8s --source-cell loglik_lasso-up_${c}"
        echo "loglik_MCP-ml_${c}|8lM${ci}1|$run|4|24:00:00|$r8s --source-cell loglik_MCP_${c}"
        echo "loglik_MCP-up-ml_${c}|8lU${ci}1|$run|8|24:00:00|$r8s --source-cell loglik_MCP-up_${c}"
        echo "loglik_MCP-up-lasso-ml_${c}|8lL${ci}1|$run|${sh[0]}|24:00:00|$r8s --source-cell loglik_MCP-up-lasso_${c}"
        echo "loglik_adaptive-ml_${c}|8la${ci}1|$run|${sh[1]}|24:00:00|$r8s --source-cell loglik_adaptive_${c}"
        echo "direct_lasso-ml_${c}|8dl${ci}1|$run|4|24:00:00|$r8s --p 10 --source-cell direct_lasso_${c}"
        echo "direct_MCP-up-ml_${c}|8dU${ci}1|$run|16|24:00:00|$r8s --p 10 --source-cell direct_MCP-up_${c}"
        echo "direct_adaptive-ml_${c}|8da${ci}1|$run|4|24:00:00|$r8s --p 10 --source-cell direct_adaptive_${c}"
        for p in 20; do        # p = 30 left out for cost (10 October); its shards were
          pc=${p:0:1}          # C2I (32 32 32 48 32), Cresc (48 40 64 100 56)
          case "$c$p" in
            C2I20) sh=(8 8 12 16 16) ;;
            Cresc20) sh=(24 24 24 32 32) ;;
          esac
          local i=0
          for e in lasso MCP MCP-up MCP-up-lasso adaptive; do
            local code
            case "$e" in lasso) code=ll ;; MCP) code=lM ;; MCP-up) code=lU ;; MCP-up-lasso) code=lL ;; adaptive) code=la ;; esac
            echo "loglik_${e}-ml_${c}_p${p}|8${code}${ci}${pc}|$run|${sh[$i]}|24:00:00|$r8b --source-cell loglik_${e}_${c}_p${p}"
            i=$((i + 1))
          done
        done
      done
      # (c) the search from 100 random sparse graphs and the empty graph, and from the truth,
      # with the likelihood refit (Amendola et al.'s procedure), p = 10
      run=simulations/run_search_shard.py
      echo "search100sml_p10_C2I|8si|$run|32|24:00:00|--p 10 --reps 2 --c-scale identity --methods pure truth --restarts 100 --starts sparse --refit loglik --add-screen 20"
      echo "search100sml_p10_Cresc|8sr|$run|32|24:00:00|--p 10 --reps 2 --c-scale variance --methods pure truth --restarts 100 --starts sparse --refit loglik --add-screen 20" ;;
    *) echo "unknown wave: $1 (1 to 8)" >&2; return 2 ;;
  esac
}

# one character for the sample size, so that job names stay under LRZ's 10 characters
n_code() { case "$1" in 1000) echo 3 ;; 1e4) echo 4 ;; 1e5) echo 5 ;; inf) echo i ;; *) echo x ;; esac; }

while [ $# -gt 0 ]; do
  case "$1" in
    --wave) WAVE=${2:?--wave needs 1 to 8}; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --fill) FILL=1; shift ;;
    --status) STATUS=1; shift ;;
    --list) LIST=1; shift ;;
    --time) TIME=${2:?--time needs HH:MM:SS}; shift 2 ;;
    --shards)
      SHARDS=${2:?--shards needs a number}; shift 2
      [[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "--shards needs a positive integer" >&2; exit 2; } ;;
    --n|--p|--only)
      opt=$1; shift; vals=()
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do vals+=("$1"); shift; done
      [ ${#vals[@]} -gt 0 ] || { echo "$opt needs at least one value" >&2; exit 2; }
      case "$opt" in
        --n) NS=("${vals[@]}"); NS_GIVEN=1 ;;
        --p) PS=("${vals[@]}") ;;
        --only) ONLY=("${vals[@]}") ;;
      esac ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ -n "$WAVE" ] || { echo "--wave is required (1 to 8); see --help" >&2; exit 2; }
if [ "$WAVE" = 4 ]; then
  [ "$NS_GIVEN" -eq 1 ] || NS=(1000)          # Figure 5's sample size unless --n says otherwise
  for p in "${PS[@]}"; do
    [[ "$p" =~ ^(15|25|30|40|50)$ ]] || { echo "unknown p for wave 4: $p (15, 25, 30, 40 or 50)" >&2; exit 2; }
  done
fi
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
