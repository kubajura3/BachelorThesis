#!/usr/bin/env bash
#
# Soft tilt barrier -- everything that has to happen on the GPU box, in one queue.
#
#   ./run_tilt.sh wiring       ~40 s   is the term actually wired into loss and gradient?
#   ./run_tilt.sh control      ~7 min  flat same-code control pair, for the noise floor
#   ./run_tilt.sh probe        ~2 min  Rudin TILT_W=0 run -- calibrates the weight where it matters
#   ./run_tilt.sh arms         ~4 min  short flat arms at w=100 / w=300
#   ./run_tilt.sh rudin-smoke  ~30 s   5 Rudin iterations at the probe-derived weight
#   ./run_tilt.sh all                  all of the above, in dependency order (~14 min)
#
# Then read every gate at once with:   python collect_tilt.py
#
# Same conventions as run_campaign.sh / run_saliency.sh: sequential (one GPU), resumable (a
# run whose RUN_DIR already has summary.json is skipped -- delete the folder to force a redo),
# and deliberately NOT `set -e` so one failure does not take the queue with it. Every run is
# tee'd to results/logs/, so the startup banner and any traceback survive the run.

set -uo pipefail

PYTHON="${PYTHON:-python}"
RESULTS="${RESULTS:-results}"
OUT="${OUT:-$RESULTS/diag21}"
SEED="${SEED:-0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"

# Flat-arm weights, calibrated off results/diag21/flat_omni_w0's last 200 iterations: mean
# loss 2.109, mean loss_tilt 1.145e-3, so these put the term at ~5% and ~15% of the objective.
# Not the 1e3-1e4 a first guess suggests -- that assumes ~1% activation and the measurement
# says 4.8-7.1%.
W_LO="${W_LO:-100}"
W_HI="${W_HI:-300}"

# Wiring test (stage 1). TILT_ON=0 makes cos_on = 1, so the hinge fires on any tilt at all and
# the term is non-zero at iteration 0 -- the only iteration where the two runs share an
# identical rollout, and so the only place an exact arithmetic identity can be checked. At the
# real TILT_ON=0.6 the barrier stays silent until iteration 54 on flat ground, by which point
# the runs have drifted and the identity has no power left.
#
# The weight is deliberately absurd: at iteration 0 the robots are near-upright (tilt ~0.01
# rad), so the squared hinge is ~1e-9 and only a large multiplier lifts the shift clear of the
# ~1e-6 float floor. It cannot destabilise anything -- train.py:794 clips the gradient to norm
# 0.3 before the optimiser sees it, and grad_norm is logged pre-clip, which is precisely what
# makes it usable as a liveness signal.
WIRE_W="${WIRE_W:-3e8}"
WIRE_ITERS="${WIRE_ITERS:-25}"

LOG_DIR="$RESULTS/logs"
mkdir -p "$LOG_DIR" "$OUT"

QUEUE_NAME=(); QUEUE_ENV=()
add_run() {            # add_run <run-dir> <"VAR=val VAR=val ...">
  QUEUE_NAME+=("$1"); QUEUE_ENV+=("$2")
}

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
      SKIPPED+=("$run_dir")
      continue
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

stage_wiring() {
  # Two 25-iteration flat runs differing ONLY in TILT_W. Iteration 0 is logged before the
  # first optimiser step, so both runs roll out the same policy through the same physics and
  #     loss_on(0) == loss_off(0) + TILT_W * loss_tilt_off(0)
  # must hold to float precision. That is the whole of "is it wired in": the value is added,
  # it is added linearly, and it is added with the right weight. grad_norm, logged pre-clip,
  # must also jump -- that is the separate claim that the gradient path is live rather than
  # detached, which train.py:727-728 makes conditional on the weight.
  add_run "$OUT/wire_w0"  "MODE=blind_omni SEED=$SEED ITERS=$WIRE_ITERS NUM_ENVS=16 TILT_W=0 TILT_ON=0.0"
  add_run "$OUT/wire_won" "MODE=blind_omni SEED=$SEED ITERS=$WIRE_ITERS NUM_ENVS=16 TILT_W=$WIRE_W TILT_ON=0.0"
  drain_queue
}

stage_control() {
  # The flat control pair that does not exist on disk: diag19/flat_omni_s0 is commit 924523c
  # with cmd_style unset, a different arm entirely. Identical to flat_omni_w0 in every respect,
  # so whatever it differs by IS the flat noise floor. Doubles as the second sample that gives
  # the short arms a spread to be read against -- without it n_falls on 16 envs is
  # uninterpretable.
  add_run "$OUT/flat_omni_w0_ctrl" "MODE=blind_omni SEED=$SEED ITERS=1000 NUM_ENVS=16 TILT_W=0"
  drain_queue
}

stage_probe() {
  # loss_tilt is logged at weight 0, so this measures the term's size in the regime the full
  # terrain run uses, without spending a full run on it. 100 iterations is the right window
  # rather than a truncation: terrain_level goes 2.485 -> 0.889 over exactly these iterations,
  # so this is the collapse the barrier is meant to interrupt. Calibrating on flat instead
  # would extrapolate from ground where the term is silent in 47% of iterations.
  add_run "$OUT/rudin_fwd_w0_probe" "MODE=blind_rudin_fwd SEED=$SEED ITERS=100 NUM_ENVS=1024 TILT_W=0"
  drain_queue
}

stage_arms() {
  # Short flat arms. 300 iterations at 16 envs is ~2 min each and enough to read direction.
  # collect_tilt.py compares them against results/diag21/flat_omni_w0 over the same
  # iteration window, never against their own first iterations.
  add_run "$OUT/flat_omni_w${W_LO}_short" "MODE=blind_omni SEED=$SEED ITERS=300 NUM_ENVS=16 TILT_W=$W_LO"
  add_run "$OUT/flat_omni_w${W_HI}_short" "MODE=blind_omni SEED=$SEED ITERS=300 NUM_ENVS=16 TILT_W=$W_HI"
  drain_queue
}

stage_rudin_smoke() {
  # 5 iterations on the terrain path at the weight the probe implies, before the 14-minute
  # terrain run is committed to it. Catches the two failures that can only show up on Rudin: a
  # NaN from the barrier meeting a real fall, and a realised share wildly off the prediction.
  local w
  w="${W_RUDIN:-}"
  if [[ -z "$w" ]]; then
    w="$("$PYTHON" collect_tilt.py --weight 0.05 --results-dir "$RESULTS" 2>/dev/null)"
  fi
  if [[ -z "${w:-}" || "$w" == "0" ]]; then
    echo "[$(stamp)] SKIP  rudin-smoke -- no weight available; run './run_tilt.sh probe' first"
    return
  fi
  echo "[$(stamp)] rudin-smoke weight from probe (5% of loss): TILT_W=$w"
  add_run "$OUT/rudin_fwd_wcal_smoke" "MODE=blind_rudin_fwd SEED=$SEED ITERS=5 NUM_ENVS=1024 TILT_W=$w"
  drain_queue
}

case "${1:-}" in
  wiring)      stage_wiring ;;
  control)     stage_control ;;
  probe)       stage_probe ;;
  arms)        stage_arms ;;
  rudin-smoke) stage_rudin_smoke ;;
  all)
    # Wiring first: if the term is not in the loss, nothing after it means anything.
    stage_wiring
    stage_probe
    stage_control
    stage_arms
    stage_rudin_smoke
    ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac

echo "======================================================================"
echo "  ran     : $RAN"
echo "  skipped : ${#SKIPPED[@]}  (already had results)"
echo "  failed  : ${#FAILED[@]}"
for f in ${FAILED[@]+"${FAILED[@]}"}; do echo "            - $f"; done
echo
echo "Now read it:  $PYTHON collect_tilt.py"
if (( ${#FAILED[@]} )); then exit 1; fi
