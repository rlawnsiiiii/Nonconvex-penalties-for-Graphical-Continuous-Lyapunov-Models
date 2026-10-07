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
#   * --dry-run says how many tasks the line would submit; 0 means done, next line;
#   * the queued tasks of the user are counted (squeue -M serial -r); the line waits
#     while LIMIT (default 200, LRZ's cap on queued tasks per user) minus that count is
#     smaller than the tasks needed;
#   * a --fill line also waits until no job whose name starts with the wave's digit is
#     in the queue, because a fill resubmits the shards that ended jobs did not write;
#   * then the line is run.  sbatch refusing part of it is not fatal: submit_campaign.sh
#     records only what was accepted and the line is retried at the next round.  A line
#     that has been run MAX_RUNS times (default 5) is given up on, with a warning.
# Every action is logged with a time stamp in logs/feed_queue.log, including a row for
# the submission log of the command sheet.  The script exits when every line is done.
# LIMIT, INTERVAL, MAX_RUNS and FEED_LOG (the log file) can be set in the environment.  Stop it with Ctrl-C or
# kill; restarting it is safe (done lines are recognised by their dry run).
set -uo pipefail
cd "$(dirname "$0")/.."

PLAN=${1:?usage: feed_queue.sh PLAN_FILE}
[ -f "$PLAN" ] || { echo "no such plan file: $PLAN" >&2; exit 2; }
LIMIT=${LIMIT:-200}
INTERVAL=${INTERVAL:-600}
MAX_RUNS=${MAX_RUNS:-5}
SUBMIT=cluster/submit_campaign.sh
LOG=${FEED_LOG:-logs/feed_queue.log}      # FEED_LOG: another log file (the tests use a temporary one)
mkdir -p logs "$(dirname "$LOG")"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

queued() {                       # the user's queued or running tasks; LIMIT if squeue fails
  local n
  n=$(squeue -M serial -u "$USER" -h -r 2>/dev/null | wc -l | tr -d ' ') || n=$LIMIT
  echo "${n:-$LIMIT}"
}
wave_jobs() {                    # queued or running jobs of one wave (job names start with its digit)
  squeue -M serial -u "$USER" -h -r -o %j 2>/dev/null | grep -c "^$1" || true
}
needed() {                       # tasks a line would submit now, from its dry run; "error" if it fails
  local out
  if ! out=$(bash "$SUBMIT" $1 --dry-run 2>&1); then echo error; return; fi
  echo "$out" | sed -n 's/^would submit \([0-9]*\) tasks.*/\1/p' | tail -1
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
RUNS=()
for ((i = 0; i < N_LINES; i++)); do RUNS[$i]=0; done
log "feeding $N_LINES lines of $PLAN (LIMIT=$LIMIT, INTERVAL=${INTERVAL}s)"

i=0
while [ "$i" -lt "$N_LINES" ]; do
  line=${LINES[$i]}
  n=$(needed "$line")
  if [ "$n" = error ]; then
    log "line $((i + 1)) '$line': the dry run failed, skipping it (check the arguments)"
    i=$((i + 1)); continue
  fi
  if [ -z "$n" ] || [ "$n" -eq 0 ]; then
    log "line $((i + 1)) '$line': nothing left to submit, done"
    i=$((i + 1)); continue
  fi
  if [ "${RUNS[$i]}" -ge "$MAX_RUNS" ]; then
    log "line $((i + 1)) '$line': run $MAX_RUNS times and still $n tasks to submit, giving up on it"
    i=$((i + 1)); continue
  fi
  q=$(queued)
  room=$((LIMIT - q))
  if [[ "$line" == *--fill* ]]; then
    w=$(wave_of "$line")
    j=$(wave_jobs "$w")
    if [ "$j" -gt 0 ]; then
      log "line $((i + 1)) '$line': $j jobs of wave $w still queued or running, waiting (fills resubmit ended shards only)"
      sleep "$INTERVAL"; continue
    fi
  fi
  if [ "$room" -lt "$n" ]; then
    log "line $((i + 1)) '$line': needs $n tasks, room for $room ($q queued of $LIMIT), waiting"
    sleep "$INTERVAL"; continue
  fi
  RUNS[$i]=$((RUNS[$i] + 1))
  log "line $((i + 1)) '$line': $n tasks, room for $room, submitting (run ${RUNS[$i]})"
  if out=$(bash "$SUBMIT" $line 2>&1); then
    log "  $(echo "$out" | tail -1)"
    log "  | $(date '+%d%m%y') | bash cluster/submit_campaign.sh $line | $(echo "$out" | tail -1) |"
    sleep "$INTERVAL"                # let the queue show the new jobs before the next decision
  else
    log "  sbatch refused part of it; retrying later: $(echo "$out" | grep '^!!' | head -1)"
    sleep "$INTERVAL"
  fi
done
log "plan complete: every line is done"
