#!/usr/bin/env bash
#
# Step 3 perceptive foothold -- everything that has to happen on the GPU box, in one queue.
# See STEP3_FOOTHOLD.md for why each stage exists and what its gate is.
#
#   ./run_step3.sh seeds     ~28 min  the 22.8 item-1 baseline seeds -- RUN THIS FIRST
#   ./run_step3.sh phase1    ~45 min  inertness / z-only / z+apex (no action-space change)
#   ./run_step3.sh wiring    ~40 s    is loss_fq actually in the loss and in the gradient?
#   ./run_step3.sh probe     ~2 min   weights-0 run -- calibrates FOOT_Q_W / FOOT_RES_W
#   ./run_step3.sh arms      ~28 min  Phase 2 at the calibrated 5% and 15% weights
#   ./run_step3.sh detach    ~14 min  FOOT_RES_DETACH=0 ablation
#   ./run_step3.sh all                seeds -> phase1 -> wiring -> probe -> arms -> detach
#
# Then read every gate at once with:   python collect_step3.py
#
# Same conventions as run_step1c.sh: sequential (one GPU), resumable (a run whose RUN_DIR
# already has summary.json is skipped -- delete the folder to force a redo), and deliberately
# NOT `set -e` so one failure does not take the queue with it. Every run is tee'd to
# results/logs/ -- and for Step 3 that matters more than usual, because the per-terrain-type
# curriculum table is the primary gate and it exists ONLY in the log, not in iters.csv.

set -uo pipefail

PYTHON="${PYTHON:-python}"
RESULTS="${RESULTS:-results}"
OUT="${OUT:-$RESULTS/diag22}"
OUT2="${OUT2:-$RESULTS/diag23}"
SEED="${SEED:-0}"
ITERS="${ITERS:-850}"
ENVS="${ENVS:-1024}"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"

# Phase 1 flags, shared by the two treatment arms.
FZ="FOOT_Z_TERRAIN=1"
FZA="FOOT_Z_TERRAIN=1 FOOT_APEX_TERRAIN=1"

# Wiring test: an absurd weight so the shift clears the float floor at iteration 0, which is the
# only iteration where the two runs share an identical rollout. It cannot destabilise anything --
# train.py clips the gradient to norm 0.3 before the optimiser sees it.
WIRE_W="${WIRE_W:-1e6}"
WIRE_ITERS="${WIRE_ITERS:-25}"

LOG_DIR="$RESULTS/logs"
mkdir -p "$LOG_DIR" "$OUT" "$OUT2"

QUEUE_NAME=(); QUEUE_ENV=()
add_run() { QUEUE_NAME+=("$1"); QUEUE_ENV+=("$2"); }
stamp() { date "+%H:%M:%S"; }
FAILED=(); SKIPPED=(); RAN=0

drain_queue() {
  local total=${#QUEUE_NAME[@]}
  local i n rc t0 dt run_dir vars log why
  for i in "${!QUEUE_NAME[@]}"; do
    run_dir="${QUEUE_NAME[$i]}"
    vars="${QUEUE_ENV[$i]}"
    log="$LOG_DIR/$(basename "$run_dir").log"
    n=$((i + 1))

    if [[ -f "$run_dir/summary.json" ]]; then
      echo "[$(stamp)] ($n/$total) SKIP  $run_dir -- summary.json already there"
      SKIPPED+=("$run_dir"); continue
    fi

    echo "[$(stamp)] ($n/$total) START $run_dir"
    echo "              $vars"
    t0=$(date +%s)
    env $vars RUN_DIR="$run_dir" timeout "$RUN_TIMEOUT" "$PYTHON" train.py 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    dt=$(( $(date +%s) - t0 ))

    if [[ $rc -eq 0 && -f "$run_dir/summary.json" ]]; then
      echo "[$(stamp)] ($n/$total) DONE  $run_dir in ${dt}s"
      RAN=$((RAN + 1))
    else
      if [[ $rc -eq 124 ]]; then why="timed out after ${RUN_TIMEOUT}s"; else why="exit code $rc"; fi
      echo "[$(stamp)] ($n/$total) FAIL  $run_dir after ${dt}s -- $why"
      FAILED+=("$run_dir")
    fi
    echo
  done
  QUEUE_NAME=(); QUEUE_ENV=()
}

# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------

stage_seeds() {
  # CAMPAIGN_FINDINGS.md 22.8 item 1. There is no TILT_W=0 Rudin replicate at 850 iterations on
  # disk, so "the arm moved" currently has no noise band to be measured against -- and Phase 1's
  # inertness arm is read against exactly that band. Nothing downstream means much without this.
  local s
  for s in 1 2; do
    add_run "$RESULTS/diag21/rudin_fwd_w0_s$s" "MODE=blind_rudin_fwd SEED=$s ITERS=$ITERS NUM_ENVS=$ENVS"
  done
  drain_queue
}

stage_phase1() {
  # A is the inertness arm: perception built, every Step 3 flag off. It isolates "turning
  # use_perception on" from the change itself, which is the one confound the treatment carries.
  add_run "$OUT/fz_off" "MODE=fz_fwd SEED=$SEED ITERS=$ITERS NUM_ENVS=$ENVS"
  add_run "$OUT/fz_z"   "MODE=fz_fwd $FZ SEED=$SEED ITERS=$ITERS NUM_ENVS=$ENVS"
  add_run "$OUT/fz_za"  "MODE=fz_fwd $FZA SEED=$SEED ITERS=$ITERS NUM_ENVS=$ENVS"
  drain_queue
}

stage_wiring() {
  # Two short runs differing ONLY in FOOT_Q_W. Iteration 0 is logged before the first optimiser
  # step, so loss_on == loss_off + w*loss_fq must hold to float precision there: that is the
  # whole of "the term is added, added linearly, and added at the weight asked for". grad_norm,
  # logged pre-clip, must also move -- the separate claim that the path is live, not detached.
  add_run "$OUT2/wire_q0"  "MODE=fhold_fwd FOOT_RES=1 $FZA SEED=$SEED ITERS=$WIRE_ITERS NUM_ENVS=16 FOOT_Q_W=0"
  add_run "$OUT2/wire_qon" "MODE=fhold_fwd FOOT_RES=1 $FZA SEED=$SEED ITERS=$WIRE_ITERS NUM_ENVS=16 FOOT_Q_W=$WIRE_W"
  drain_queue
}

stage_probe() {
  # loss_fq and loss_fres are logged at weight 0, so a run that uses no weight measures the size
  # of the terms for one that will (the 22.3 procedure). 100 iterations covers the window the
  # curriculum actually collapses over. Also the 4.4 blur check: compare loss_fq on the stair
  # rows against the smooth-slope rows -- if they read the same, the blur is too wide.
  add_run "$OUT2/fhold_probe" "MODE=fhold_fwd FOOT_RES=1 $FZA SEED=$SEED ITERS=100 NUM_ENVS=$ENVS"
  drain_queue
}

stage_arms() {
  local qw rw
  qw="${FOOT_Q_W_CAL:-}"; rw="${FOOT_RES_W_CAL:-}"
  if [[ -z "$qw" ]]; then
    qw="$("$PYTHON" collect_step3.py --weight 0.05 --results-dir "$RESULTS" 2>/dev/null)"
  fi
  if [[ -z "$rw" ]]; then
    rw="$("$PYTHON" collect_step3.py --res-weight 0.02 --results-dir "$RESULTS" 2>/dev/null)"
  fi
  if [[ -z "${qw:-}" || "$qw" == "0" ]]; then
    echo "[$(stamp)] SKIP  arms -- no weight available; run './run_step3.sh probe' first"
    return
  fi
  echo "[$(stamp)] arms: FOOT_Q_W=$qw (5% of loss)  FOOT_RES_W=$rw (2%)"
  add_run "$OUT2/fhold_q$qw" \
    "MODE=fhold_fwd FOOT_RES=1 $FZA SEED=$SEED ITERS=$ITERS NUM_ENVS=$ENVS FOOT_Q_W=$qw FOOT_RES_W=$rw"
  local qw3=$(( qw * 3 ))
  add_run "$OUT2/fhold_q$qw3" \
    "MODE=fhold_fwd FOOT_RES=1 $FZA SEED=$SEED ITERS=$ITERS NUM_ENVS=$ENVS FOOT_Q_W=$qw3 FOOT_RES_W=$rw"
  drain_queue
}

stage_detach() {
  # Whether the degenerate minimum 4.3 guards against actually appears, rather than assuming it.
  local qw rw
  qw="${FOOT_Q_W_CAL:-$("$PYTHON" collect_step3.py --weight 0.05 --results-dir "$RESULTS" 2>/dev/null)}"
  rw="${FOOT_RES_W_CAL:-$("$PYTHON" collect_step3.py --res-weight 0.02 --results-dir "$RESULTS" 2>/dev/null)}"
  if [[ -z "${qw:-}" || "$qw" == "0" ]]; then
    echo "[$(stamp)] SKIP  detach -- no weight available; run './run_step3.sh probe' first"
    return
  fi
  add_run "$OUT2/fhold_nodetach" \
    "MODE=fhold_fwd FOOT_RES=1 $FZA FOOT_RES_DETACH=0 SEED=$SEED ITERS=$ITERS NUM_ENVS=$ENVS FOOT_Q_W=$qw FOOT_RES_W=$rw"
  drain_queue
}

case "${1:-}" in
  seeds)   stage_seeds ;;
  phase1)  stage_phase1 ;;
  wiring)  stage_wiring ;;
  probe)   stage_probe ;;
  arms)    stage_arms ;;
  detach)  stage_detach ;;
  all)
    # Seeds first (they are the band everything is read against), then Phase 1. Wiring before
    # any Phase 2 arm: if the term is not in the loss, nothing after it means anything.
    stage_seeds
    stage_phase1
    stage_wiring
    stage_probe
    stage_arms
    stage_detach
    ;;
  *) sed -n '2,21p' "$0"; exit 1 ;;
esac

echo "======================================================================"
echo "  ran     : $RAN"
echo "  skipped : ${#SKIPPED[@]}  (already had results)"
echo "  failed  : ${#FAILED[@]}"
for f in ${FAILED[@]+"${FAILED[@]}"}; do echo "            - $f"; done
echo
echo "Now read it:  $PYTHON collect_step3.py"
if (( ${#FAILED[@]} )); then exit 1; fi
