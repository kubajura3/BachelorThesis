#!/usr/bin/env bash
#
# Queue the thesis training runs so they execute one after another, unattended.
#
#   ./run_campaign.sh smoke     5 short runs; this is what picks NUM_ENVS
#   ./run_campaign.sh exp1     12 speed runs (6 CUDA + 6 PyTorch), flat terrain
#   ./run_campaign.sh exp2     13 locomotion runs on the Rudin curriculum
#   ./run_campaign.sh all      exp1 then exp2
#
# Runs are sequential on purpose: one GPU, and two training processes would
# contend for it and corrupt every timing measurement.
#
# Resumable: a run whose RUN_DIR already contains summary.json is skipped, so if
# the queue dies at run 9 of 13 you just start it again and it picks up there.
# Delete that run's folder to force a redo.
#
# Deliberately NOT `set -e`: one failing run must not take the rest of the queue
# with it. Failures are recorded and reported in the summary at the end.

set -uo pipefail

# ----------------------------------------------------------------------------
# Settings -- edit these two, everything else follows
# ----------------------------------------------------------------------------
BSTAR="${BSTAR:-1024}"          # parallel robots for exp2; set from the smoke runs
ITERS="${ITERS:-5000}"          # training iterations for exp2
BENCH_ITERS="${BENCH_ITERS:-100}"
SMOKE_ITERS="${SMOKE_ITERS:-30}"
SMOKE_ENVS="${SMOKE_ENVS:-2048}"   # start here; drop to 1024 if depth OOMs
PYTHON="${PYTHON:-python}"
RESULTS="${RESULTS:-results}"
# Kill a single run after this many seconds so a hang cannot block the queue
# overnight. 0 disables. Default 4 h, comfortably above a 5000-iteration run.
RUN_TIMEOUT="${RUN_TIMEOUT:-14400}"

LOG_DIR="$RESULTS/logs"
mkdir -p "$LOG_DIR"

QUEUE_NAME=(); QUEUE_ENV=()
add_run() {           # add_run <name> <"VAR=val VAR=val ...">
  QUEUE_NAME+=("$1")
  QUEUE_ENV+=("$2")
}

# ----------------------------------------------------------------------------
# The run lists
# ----------------------------------------------------------------------------
build_smoke() {
  for mode in blind_rudin hobs hloss height depth; do
    add_run "smoke/$mode" \
      "MODE=$mode SEED=0 ITERS=$SMOKE_ITERS NUM_ENVS=$SMOKE_ENVS DEBUG_TRAIN=1"
  done
}

build_exp1() {
  for b in 16 64 256 1024 2048 4096; do
    add_run "bench/cuda_B$b" \
      "CUDA_KERNEL_SRBD=1 MODE=blind SEED=0 ITERS=$BENCH_ITERS NUM_ENVS=$b"
  done
  for b in 16 64 256 1024 2048 4096; do
    add_run "bench/torch_B$b" \
      "CUDA_KERNEL_SRBD=0 MODE=blind SEED=0 ITERS=$BENCH_ITERS NUM_ENVS=$b"
  done
}

build_exp2() {
  for mode in blind_rudin hobs hloss height; do
    for seed in 0 1 2; do
      add_run "train/${mode}_s${seed}" \
        "MODE=$mode SEED=$seed ITERS=$ITERS NUM_ENVS=$BSTAR"
    done
  done
  add_run "train/depth_s0" "MODE=depth SEED=0 ITERS=$ITERS NUM_ENVS=$BSTAR"
}

case "${1:-}" in
  smoke) build_smoke ;;
  exp1)  build_exp1 ;;
  exp2)  build_exp2 ;;
  all)   build_exp1; build_exp2 ;;
  *) echo "usage: $0 {smoke|exp1|exp2|all}" >&2; exit 2 ;;
esac

# ----------------------------------------------------------------------------
# Run them
# ----------------------------------------------------------------------------
total=${#QUEUE_NAME[@]}
FAILED=(); SKIPPED=()
started_at=$(date +%s)

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
trap 'echo; echo "[$(stamp)] interrupted -- stopping the queue"; exit 130' INT TERM

echo "[$(stamp)] campaign '${1}': $total runs, results under $RESULTS/"
echo "[$(stamp)] BSTAR=$BSTAR ITERS=$ITERS BENCH_ITERS=$BENCH_ITERS"
echo

for i in "${!QUEUE_NAME[@]}"; do
  name="${QUEUE_NAME[$i]}"
  vars="${QUEUE_ENV[$i]}"
  run_dir="$RESULTS/$name"
  log="$LOG_DIR/$(echo "$name" | tr '/' '_').log"
  n=$((i + 1))

  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[$(stamp)] ($n/$total) SKIP  $name -- summary.json already there"
    SKIPPED+=("$name")
    continue
  fi

  echo "[$(stamp)] ($n/$total) START $name"
  echo "              $vars RUN_DIR=$run_dir"
  echo "              log: $log"
  t0=$(date +%s)

  if [[ "$RUN_TIMEOUT" -gt 0 ]]; then
    env $vars RUN_DIR="$run_dir" timeout "$RUN_TIMEOUT" "$PYTHON" train.py >"$log" 2>&1
  else
    env $vars RUN_DIR="$run_dir" "$PYTHON" train.py >"$log" 2>&1
  fi
  rc=$?
  dt=$(( $(date +%s) - t0 ))

  if [[ $rc -eq 0 && -f "$run_dir/summary.json" ]]; then
    ips=$(grep -o '"it_per_s_median": [0-9.]*' "$run_dir/summary.json" | head -1 | awk '{print $2}')
    echo "[$(stamp)] ($n/$total) DONE  $name in ${dt}s  (${ips:-?} it/s median)"
  else
    [[ $rc -eq 124 ]] && why="timed out after ${RUN_TIMEOUT}s" || why="exit code $rc"
    echo "[$(stamp)] ($n/$total) FAIL  $name after ${dt}s -- $why"
    echo "              last lines of $log:"
    tail -n 12 "$log" | sed 's/^/                | /'
    FAILED+=("$name")
  fi
  echo
done

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
elapsed=$(( $(date +%s) - started_at ))
echo "======================================================================"
printf '[%s] campaign finished in %02dh%02dm\n' "$(stamp)" $((elapsed/3600)) $(((elapsed%3600)/60))
echo "  ran     : $(( total - ${#SKIPPED[@]} - ${#FAILED[@]} ))"
echo "  skipped : ${#SKIPPED[@]}  (already had results)"
echo "  failed  : ${#FAILED[@]}"
for f in ${FAILED[@]+"${FAILED[@]}"}; do echo "            - $f"; done
echo

if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "Aggregating..."
  "$PYTHON" collect_bench.py --results-dir "$RESULTS" --out "$RESULTS/figures"
  echo
  echo "Results: $RESULTS/figures/"
else
  echo "Fix the failures, re-run this script (finished runs are skipped), then:"
  echo "  $PYTHON collect_bench.py --results-dir $RESULTS --out $RESULTS/figures"
fi
