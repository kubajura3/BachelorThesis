#!/usr/bin/env bash
#
#
#   ./run_saliency.sh selftest   wiring checks only, seconds, no rollout
#   ./run_saliency.sh probe      one short run to MEASURE the per-run cost
#   ./run_saliency.sh matrix     the 13-run diagnostic matrix
#
# Re-evaluates the already-trained policies in results/train/*_s0 with their terrain input
# corrupted. Invariance to that corruption is direct evidence the policy learned to disregard
# the signal, which is what turns the degenerate sec 6.4/6.5 result into a mechanism claim
# (THESIS_PLAN sec E.3). No retraining.
#
# Same conventions as run_campaign.sh: sequential (one GPU), resumable (a run whose output
# json already exists is skipped), and deliberately NOT `set -e` so one failure does not take
# the queue with it.

set -uo pipefail

PYTHON="${PYTHON:-python}"
RESULTS="${RESULTS:-results}"
OUT="${OUT:-$RESULTS/saliency}"
TRAIN="${TRAIN:-$RESULTS/train}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEED="${SEED:-0}"
# 15000 steps = 30 s of sim at dt=0.002; one episode is 20 s. A short warmup keeps the metric
# window inside the phase where robots are still spread over levels 0-5 (spawn is
# randint(0, max_init_terrain_level+1), ceiling 5), which is where the height scan carries
# signal at all. See the plan's "warmup guidance" note.
STEPS="${STEPS:-15000}"
WARMUP="${WARMUP:-2000}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"

LOG_DIR="$RESULTS/logs"
mkdir -p "$LOG_DIR" "$OUT"

# mode -> obs-mode used by evaluate_rudin_comparison.py
obs_mode_for() {
  case "$1" in
    blind_rudin) echo blind  ;;
    hobs|height) echo height ;;
    depth)       echo depth  ;;
    *) echo "unknown mode $1" >&2; return 1 ;;
  esac
}

weights_for() { echo "$TRAIN/$1_s0/quad_diffsim_srbd_align_multi_robot.pth"; }

FAILED=()
run_eval() {          # run_eval <mode> <corrupt> <tag-suffix> [extra args...]
  local mode="$1" corrupt="$2" suffix="$3"; shift 3
  local tag="${mode}_${suffix}"
  local w; w="$(weights_for "$mode")"
  local om; om="$(obs_mode_for "$mode")" || return 1

  if [[ ! -f "$w" ]]; then
    echo "[skip] $tag -- no weights at $w"; FAILED+=("$tag(no-weights)"); return
  fi
  if [[ -f "$OUT/eval_results_${tag}.json" ]]; then
    echo "[skip] $tag -- already done"; return
  fi

  echo "[run ] $tag  (obs-mode=$om corrupt=$corrupt)"
  local t0=$SECONDS
  timeout "${RUN_TIMEOUT}" "$PYTHON" evaluate_rudin_comparison.py \
      --weights "$w" --obs-mode "$om" --corrupt "$corrupt" \
      --num_envs "$NUM_ENVS" --seed "$SEED" --steps "$STEPS" --warmup "$WARMUP" \
      --tag "$tag" --out "$OUT" "$@" \
      2>&1 | tee "$LOG_DIR/saliency_${tag}.log"
  local rc=${PIPESTATUS[0]}
  echo "[done] $tag  rc=$rc  $((SECONDS - t0))s"
  [[ $rc -ne 0 ]] && FAILED+=("$tag(rc=$rc)")
  return 0
}

case "${1:-}" in

  selftest)
    # Tier 1: does the corruption actually reach the policy? A zero action delta under
    # zero/shuffle means the channel slice is wrong and the whole diagnostic would return a
    # fake null. Run this before spending any GPU on the matrix.
    # hobs and height share the same architecture and channel, so hobs covers both. One
    # invocation per obs-mode: --selftest checks none/zero/shuffle internally and the Rudin
    # trimesh build is what actually costs time here.
    for mode in hobs depth; do
      w="$(weights_for "$mode")"; om="$(obs_mode_for "$mode")"
      [[ -f "$w" ]] || { echo "[skip] $mode -- no weights"; continue; }
      "$PYTHON" evaluate_rudin_comparison.py --weights "$w" --obs-mode "$om" \
          --num_envs 64 --seed "$SEED" --selftest || FAILED+=("selftest_$mode")
    done
    echo ""
    echo "Also expect a clean refusal from this one (blind has no terrain channel):"
    "$PYTHON" evaluate_rudin_comparison.py --obs-mode blind --corrupt zero --selftest \
      && echo "  FAIL -- that should have exited non-zero" \
      || echo "  OK -- refused as intended"
    ;;

  probe)
    # Tier 3 prerequisite: MEASURE the per-run cost instead of extrapolating it. Multiply the
    # reported seconds by STEPS/2000 to size the matrix, and remember depth is slower because
    # of the camera render.
    FULL_STEPS="$STEPS"
    STEPS=2000; WARMUP=500          # plain assignment: `VAR=x func` would leak into the shell
    run_eval height none probe
    echo "Scale that wall time by ~$((FULL_STEPS / STEPS))x for a full --steps $FULL_STEPS run."
    ;;

  matrix)
    # Tier 2 (noise floor) and tier 3 (diagnostic) in one queue. The two `none` runs per
    # obs-mode are the floor: sec 19.14 established this sim is not bit-reproducible
    # run-to-run, so a corruption effect only counts if it exceeds the spread between them.
    run_eval blind_rudin none none          # terrain-blind floor -- the other half of the test
    for mode in hobs height depth; do
      run_eval "$mode" none    none
      run_eval "$mode" none    none2        # same seed, second draw -> the noise floor
      run_eval "$mode" zero    zero
      run_eval "$mode" shuffle shuffle
    done
    echo ""
    echo "Now read it:  $PYTHON collect_saliency.py"
    ;;

  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac

if (( ${#FAILED[@]} )); then
  echo ""; echo "FAILED: ${FAILED[*]}"; exit 1
fi
echo ""; echo "All runs OK."
