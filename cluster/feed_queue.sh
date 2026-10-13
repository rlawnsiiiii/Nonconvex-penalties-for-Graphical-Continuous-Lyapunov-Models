#!/bin/bash
# Feed the campaign to the LRZ queue without watching it.  The submit commands of a
# plan file are run in order, each as soon as the queue has room for all of its tasks.
#
#     bash cluster/feed_queue.sh cluster/plan_071026.txt                       # foreground
#     nohup bash cluster/feed_queue.sh cluster/plan_071026.txt > logs/feed.out 2>&1 &   # background
#     tail -f logs/feed_queue.log                                              # what it does
#
# The plan file holds one argument list of cluster/submit_campaign.sh per line, in the
# order wanted; blank lines and lines starting with # are skipped.  Example:
#     --wave 4 --p 30
#     --wave 5 --n 1e4 --only -ml
#     --wave 4 --fill --time 24:00:00
# Every INTERVAL seconds (default 600) the script looks at the first line that is not
# done yet:
#   * --dry-run says how many tasks the line would submit, and how large its largest
#     cell is; 0 tasks means done, next line;
#   * the queued tasks of the user are counted (squeue -M serial -r); the line waits
#     while LIMIT (default 200, LRZ's cap on queued tasks per user) minus that count is
#     smaller than the tasks needed.  A line larger than the cap (wave 4 at p = 40 or
#     50: 224 tasks) can never fit at once, so it goes in parts: it is run whenever
#     there is room for its largest cell, submit_campaign.sh submits cells until sbatch
#     refuses one, and the rest follows in later rounds;
#   * a --fill line also waits until no job whose name starts with the wave's digit is
#     in the queue, because a fill resubmits the shards that ended jobs did not write;
#   * then the line is run.  sbatch refusing a cell is not fatal: submit_campaign.sh
#     records only what was accepted and the line is retried at the next round.  A line
#     is given up on, with a warning, after MAX_RUNS runs (default 5) in which sbatch
#     accepted nothing; a --fill line is given up on when its jobs have ended and the
#     number of missing shards has not fallen (the resubmitted tasks fail every time, for
#     instance a task that always exceeds its time limit -- code review C1).
# Safeguards (code review C2, 13 October 2026): one feeder at a time (a lock directory,
# logs/feed_queue.lock, holding the feeder's pid; a lock whose feeder is gone is taken over);
# a squeue failure makes a --fill line wait, never proceed; a plan line whose dry run fails
# (e.g. --only matching no cell) is skipped with the dry run's message.
# Every action is logged with a time stamp in logs/feed_queue.log, including a row for
# the submission log of the command sheet.  The script exits when every line is done.
# LIMIT, INTERVAL, MAX_RUNS, FEED_LOG (the log file) and FEED_LOCK (the lock directory) can be
# set in the environment.  Stop it with Ctrl-C or kill; restarting it is safe (done lines are
# recognised by their dry run).
set -uo pipefail
cd "$(dirname "$0")/.."

PLAN=${1:?usage: feed_queue.sh PLAN_FILE}
[ -f "$PLAN" ] || { echo "no such plan file: $PLAN" >&2; exit 2; }
LIMIT=${LIMIT:-200}
INTERVAL=${INTERVAL:-600}
MAX_RUNS=${MAX_RUNS:-5}
SUBMIT=cluster/submit_campaign.sh
LOG=${FEED_LOG:-logs/feed_queue.log}      # FEED_LOG: another log file (the tests use a temporary one)
LOCK=${FEED_LOCK:-logs/feed_queue.lock}   # one feeder at a time: two would submit every cell twice
mkdir -p logs "$(dirname "$LOG")" "$(dirname "$LOCK")"

if ! mkdir "$LOCK" 2>/dev/null; then
  pid=$(cat "$LOCK/pid" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "another feeder (pid $pid) holds $LOCK; two feeders would submit every cell twice" >&2
    exit 1
  fi
  rm -rf "$LOCK"                           # a stale lock: its feeder is gone
  mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
nap() { sleep "$1" & wait $!; }            # a sleep that a signal interrupts at once

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

queued() {                       # the user's queued or running tasks; LIMIT if squeue fails
  local n
  n=$(squeue -M serial -u "$USER" -h -r 2>/dev/null | wc -l | tr -d ' ') || n=$LIMIT
  echo "${n:-$LIMIT}"
}
wave_jobs() {                    # queued or running jobs of one wave (job names start with its
  local out                     # digit); "error" if squeue fails, so that a fill never proceeds blindly
  out=$(squeue -M serial -u "$USER" -h -r -o %j 2>/dev/null) || { echo error; return; }
  printf '%s\n' "$out" | grep -c "^$1" || true
}
inspect() {                      # sets NEED (tasks a line would submit now) and BIG (its largest
  local out spec a b n          # cell's tasks) from the dry run; NEED=error if the dry run fails
  NEED=error; BIG=0; INSPECT_MSG=""
  out=$(bash "$SUBMIT" $1 --dry-run 2>&1) || { INSPECT_MSG=$(echo "$out" | tail -1); return 0; }
  NEED=$(echo "$out" | sed -n 's/^would submit \([0-9]*\) tasks.*/\1/p' | tail -1)
  NEED=${NEED:-0}
  for spec in $(echo "$out" | grep -o -- '--array=[0-9,-]*' | cut -d= -f2); do
    if [[ "$spec" == *-* ]]; then a=${spec%-*}; b=${spec#*-}; n=$((b - a + 1))
    else n=$(echo "$spec" | tr ',' '\n' | wc -l | tr -d ' '); fi
    [ "$n" -gt "$BIG" ] && BIG=$n
  done
  return 0
}
wave_of() { echo "$1" | sed -n 's/.*--wave *\([0-9]*\).*/\1/p'; }

LINES=()                          # read line by line: works in bash 3.2 (macOS) as well as 5
N_LINES=0
while IFS= read -r raw || [ -n "$raw" ]; do
  line=$(printf '%s' "$raw" | sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  [ -n "$line" ] || continue
  LINES[$N_LINES]=$line
  N_LINES=$((N_LINES + 1))
done < "$PLAN"
[ "$N_LINES" -gt 0 ] || { echo "the plan file is empty" >&2; exit 2; }
IDLE=()                           # runs of a line in which sbatch accepted nothing
LAST_NEED=()                      # for a --fill line: the shards missing when it was last submitted
for ((i = 0; i < N_LINES; i++)); do IDLE[$i]=0; LAST_NEED[$i]=""; done
log "feeding $N_LINES lines of $PLAN (LIMIT=$LIMIT, INTERVAL=${INTERVAL}s)"

i=0
while [ "$i" -lt "$N_LINES" ]; do
  line=${LINES[$i]}
  inspect "$line"
  n=$NEED
  if [ "$n" = error ]; then
    log "line $((i + 1)) '$line': the dry run failed, skipping it (check the arguments): $INSPECT_MSG"
    i=$((i + 1)); continue
  fi
  if [ "$n" -eq 0 ]; then
    log "line $((i + 1)) '$line': nothing left to submit, done"
    i=$((i + 1)); continue
  fi
  if [ "${IDLE[$i]}" -ge "$MAX_RUNS" ]; then
    log "line $((i + 1)) '$line': $MAX_RUNS runs in which sbatch accepted nothing and still $n tasks to submit, giving up on it"
    i=$((i + 1)); continue
  fi
  q=$(queued)
  room=$((LIMIT - q))
  want=$n                        # room for the whole line ...
  if [ "$n" -gt "$LIMIT" ]; then want=$BIG; fi   # ... or, if it can never fit at once, for its largest cell
  if [[ "$line" == *--fill* ]]; then
    w=$(wave_of "$line")
    j=$(wave_jobs "$w")
    if [ "$j" = error ]; then
      log "line $((i + 1)) '$line': squeue failed, cannot tell whether jobs of wave $w are still running; waiting"
      nap "$INTERVAL"; continue
    fi
    if [ "$j" -gt 0 ]; then
      log "line $((i + 1)) '$line': $j jobs of wave $w still queued or running, waiting (fills resubmit ended shards only)"
      nap "$INTERVAL"; continue
    fi
    if [ -n "${LAST_NEED[$i]}" ] && [ "$n" -ge "${LAST_NEED[$i]}" ]; then
      log "line $((i + 1)) '$line': its jobs have ended and $n shards are still missing (${LAST_NEED[$i]} before the last fill): the resubmitted tasks fail every time, giving up on it (check logs/camp_*.err and sacct)"
      i=$((i + 1)); continue
    fi
  fi
  if [ "$room" -lt "$want" ]; then
    if [ "$want" -eq "$n" ]; then
      log "line $((i + 1)) '$line': needs $n tasks, room for $room ($q queued of $LIMIT), waiting"
    else
      log "line $((i + 1)) '$line': $n tasks, more than the cap, goes in parts; its largest cell needs $want, room for $room ($q queued of $LIMIT), waiting"
    fi
    nap "$INTERVAL"; continue
  fi
  log "line $((i + 1)) '$line': $n tasks, room for $room, submitting"
  if out=$(bash "$SUBMIT" $line 2>&1); then
    log "  $(echo "$out" | tail -1)"
    log "  | $(date '+%d%m%y') | bash cluster/submit_campaign.sh $line | $(echo "$out" | tail -1) |"
    IDLE[$i]=0
    [[ "$line" == *--fill* ]] && LAST_NEED[$i]=$n
  else
    before=$(echo "$out" | sed -n 's/^!! \([0-9]*\) tasks in \([0-9]*\) cells were submitted before it.*/\1/p' | tail -1)
    if [ -n "$before" ] && [ "$before" -gt 0 ]; then
      log "  sbatch refused a cell after $before tasks were accepted; the rest follows in a later round"
      log "  | $(date '+%d%m%y') | bash cluster/submit_campaign.sh $line | $before tasks accepted, then sbatch refused |"
      IDLE[$i]=0
    else
      IDLE[$i]=$((IDLE[$i] + 1))
      log "  sbatch refused it and accepted nothing (${IDLE[$i]} of $MAX_RUNS such runs); retrying later"
    fi
  fi
  nap "$INTERVAL"                    # let the queue show the new jobs before the next decision
done
log "plan complete: every line is done"
