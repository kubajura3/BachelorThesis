#!/usr/bin/env python
"""Read every tilt-barrier check produced by ``run_tilt.sh`` in one pass.

    python collect_tilt.py                    # all five sections, with verdicts
    python collect_tilt.py --weight 0.05      # just the Rudin weight, for scripting

Stdlib only, so it runs on a laptop with no torch as happily as on the GPU box. Sections whose
runs are missing say which stage produces them instead of failing.

The five checks, and why each one is here:

1. WIRING     -- at iteration 0 the two wiring runs share an identical rollout (the row is
                 logged before the first optimiser step), so ``loss_on == loss_off + w*loss_tilt``
                 is an exact arithmetic identity. It is the only cheap check that the term is
                 added, added linearly, and added with the weight that was asked for.
2. NOISE      -- this sim is not bit-reproducible. The same-code control pair is what turns
                 "the runs differ" into a number, and the Rudin floor must not be reused for
                 flat ground.
3. CALIBRATION-- loss_tilt is logged even at weight 0, so a run that used no barrier measures
                 the weight for one that will. Flat ground is the wrong regime to read it from;
                 the Rudin probe is the right one.
4. ARMS       -- the short flat arms, read against the flat baseline over the SAME iteration
                 window and against the noise band from section 2.
5. RUDIN SMOKE-- five terrain iterations at the calibrated weight: no NaN, and a realised share
                 close to what section 3 predicted, before the full terrain run is
                 committed to it.
"""

import argparse
import csv
import json
import math
import os
import sys

# Run directories, all relative to --results-dir. Kept in one place so a rename is one edit.
BASE = "diag21/flat_omni_w0"                 # the TILT_W=0 baseline already on disk
CTRL = "diag21/flat_omni_w0_ctrl"            # stage `control`
OLD = "diag19/flat_omni_gatefix_s0"          # the pre-barrier run the baseline was compared against
WIRE_OFF = "diag21/wire_w0"                  # stage `wiring`
WIRE_ON = "diag21/wire_won"
PROBE = "diag21/rudin_fwd_w0_probe"          # stage `probe`
SMOKE = "diag21/rudin_fwd_wcal_smoke"        # stage `rudin-smoke`
ARM_GLOB = "diag21/flat_omni_w{}_short"      # stage `arms`
ARM_WEIGHTS = (100, 300)

# The window the flat arms are read over. Starts at 100 so the comparison is not dominated by
# the shared initial transient, ends at 300 because that is how long the short arms run.
ARM_LO, ARM_HI = 100, 300


def rows(results_dir, run):
    """iters.csv as a list of dicts, or None if the run has not been done."""
    path = os.path.join(results_dir, run, "iters.csv")
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def meta(results_dir, run):
    path = os.path.join(results_dir, run, "meta.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def col(rs, key, lo=0, hi=None):
    """Column as floats over rows[lo:hi], skipping blanks and NaNs."""
    out = []
    for r in rs[lo:hi]:
        v = r.get(key, "")
        if v in ("", None):
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if not math.isnan(x):
            out.append(x)
    return out


def mean(rs, key, lo=0, hi=None):
    v = col(rs, key, lo, hi)
    return sum(v) / len(v) if v else float("nan")


def total(rs, key, lo=0, hi=None):
    return sum(col(rs, key, lo, hi))


def reldev(a, b):
    """|a-b| / |b|, guarding a zero reference."""
    return abs(a - b) / abs(b) if b else abs(a - b)


def missing(name, stage):
    print("  not run yet -- ./run_tilt.sh %s   (%s)" % (stage, name))


# ---------------------------------------------------------------------------
# 1. Wiring
# ---------------------------------------------------------------------------
def section_wiring(rd):
    print("\n1. WIRING -- is the term in the loss, and is the gradient path live?")
    off, on = rows(rd, WIRE_OFF), rows(rd, WIRE_ON)
    if not off or not on:
        return missing("%s / %s" % (WIRE_OFF, WIRE_ON), "wiring")

    w = float(meta(rd, WIRE_ON).get("tilt_w", 0.0))
    o, n = off[0], on[0]
    lt_off, lt_on = float(o["loss_tilt"]), float(n["loss_tilt"])
    l_off, l_on = float(o["loss"]), float(n["loss"])
    fa_off, fa_on = float(o["tilt_frac_active"]), float(n["tilt_frac_active"])
    g_off, g_on = float(o["grad_norm"]), float(n["grad_norm"])

    shift = w * lt_off
    resid = abs(l_on - (l_off + shift))
    # The float floor on `loss` at iteration 0 is ~1e-6 relative, measured from the flat
    # baseline-vs-diag19 pair. 1e-4 is two orders above that and still ~1e-3 of the shift.
    tol = 1e-4 * abs(l_off)

    print("     TILT_W = %-10g TILT_ON = %s   (iteration 0, before the first update)"
          % (w, meta(rd, WIRE_ON).get("tilt_on")))
    print("     loss_tilt      off %.6e   on %.6e" % (lt_off, lt_on))
    print("     frac_active    off %.4f          on %.4f" % (fa_off, fa_on))
    print("     loss           off %.8f     on %.8f" % (l_off, l_on))
    print("     grad_norm      off %.6f       on %.6f   (pre-clip; train.py:794 clips to 0.3)"
          % (g_off, g_on))
    print("     predicted on   %.8f  = off + w*loss_tilt   (shift %.6f)" % (l_off + shift, shift))
    print("     residual       %.3e   tol %.3e" % (resid, tol))

    ok_value = lt_off > 0 and reldev(lt_on, lt_off) < 1e-4
    ok_power = shift > 100 * tol
    ok_ident = resid <= tol
    ok_grad = g_off > 0 and reldev(g_on, g_off) > 1e-3
    # Not "== 1": the robots leave reset exactly upright, so the first timestep of the rollout
    # has cos_tilt == 1 and contributes no violation. What must hold is that the hinge fires
    # broadly, and fires identically in both runs -- the weight must not touch the mask.
    ok_active = fa_off > 0.5 and abs(fa_on - fa_off) < 1e-9

    print("     [%s] hinge fires broadly at TILT_ON=0, and identically in both runs"
          % ("PASS" if ok_active else "FAIL"))
    print("     [%s] same value computed with and without the graph"
          % ("PASS" if ok_value else "FAIL"))
    print("     [%s] the test has power (shift is %.0fx the tolerance)"
          % ("PASS" if ok_power else "WEAK", shift / tol if tol else 0))
    print("     [%s] loss identity: the term is added linearly at exactly TILT_W"
          % ("PASS" if ok_ident else "FAIL"))
    print("     [%s] grad_norm moved: the gradient path is live, not detached"
          % ("PASS" if ok_grad else "FAIL"))
    if not (ok_ident and ok_grad):
        print("     -> STOP. Nothing downstream means anything until this passes.")


# ---------------------------------------------------------------------------
# 2. Noise floor
# ---------------------------------------------------------------------------
def section_noise(rd):
    print("\n2. NOISE FLOOR -- the flat control pair")
    base, ctrl, old = rows(rd, BASE), rows(rd, CTRL), rows(rd, OLD)
    if not base:
        return missing(BASE, "(baseline already on disk?)")
    if not ctrl:
        return missing(CTRL, "control")

    print("     %-6s %-24s %-24s" % ("", "|base - ctrl|  (same code)", "|base - diag19| (across 1c)"))
    print("     %-6s %-11s %-11s %-11s %-11s" % ("iter", "loss", "grad_norm", "loss", "grad_norm"))
    verdict = None
    for i in (0, 1, 2, 5, 10, 50, 100, 500, 999):
        if i >= min(len(base), len(ctrl)):
            continue
        dc_l = reldev(float(ctrl[i]["loss"]), float(base[i]["loss"]))
        dc_g = reldev(float(ctrl[i]["grad_norm"]), float(base[i]["grad_norm"]))
        line = "     %-6d %-11.2e %-11.2e" % (i, dc_l, dc_g)
        if old and i < len(old):
            do_l = reldev(float(old[i]["loss"]), float(base[i]["loss"]))
            do_g = reldev(float(old[i]["grad_norm"]), float(base[i]["grad_norm"]))
            line += " %-11.2e %-11.2e" % (do_l, do_g)
            if i == 0:
                verdict = (do_l <= 1.5 * dc_l, do_l, dc_l)
        print(line)

    # Only iteration 0 can decide this. Both runs roll out the same initial policy there, so
    # the two columns measure the same thing. From iteration 1 the trajectories have already
    # forked and every later row is chaos amplifying that fork -- about 500x growth was seen
    # in 30 iterations on Rudin, and this is a 16-env flat run, which averages even less.
    if verdict is None:
        print("     no diag19 run to compare against -- floor only")
    else:
        ok, do_l, dc_l = verdict
        print("     [%s] iteration 0: across-barrier deviation %.2e vs same-code floor %.2e"
              % ("PASS" if ok else "FAIL", do_l, dc_l))
        print("         Later rows are informational: the trajectories have forked by then and")
        print("         chaos, not the tilt term, sets their size.")

    print("\n     Falls, which is what the flat arms must NOT be read on:")
    for name, rs in (("base", base), ("ctrl", ctrl), ("diag19", old)):
        if not rs:
            continue
        lo = max(0, len(rs) - 200)
        print("       %-8s total %4.0f over %d iters   last200 %.3f falls/iter"
              % (name, total(rs, "n_falls"), len(rs), mean(rs, "n_falls", lo)))
    if ctrl:
        lo_b, lo_c = max(0, len(base) - 200), max(0, len(ctrl) - 200)
        a, b = mean(base, "n_falls", lo_b), mean(ctrl, "n_falls", lo_c)
        spread = abs(a - b) / max(a, b, 1e-9)
        print("       same-policy spread on last200 falls/iter: %.0f%%  <- the flat noise band"
              % (100 * spread))


# ---------------------------------------------------------------------------
# 3. Rudin calibration
# ---------------------------------------------------------------------------
def rudin_weight(rd, frac):
    """Weight putting the barrier at `frac` of the objective on the Rudin probe. 0 if unknown."""
    p = rows(rd, PROBE)
    if not p:
        return 0
    lt = mean(p, "loss_tilt")
    ls = mean(p, "loss")
    if not lt or math.isnan(lt) or math.isnan(ls):
        return 0
    return int(round(frac * ls / lt))


def section_calibration(rd):
    print("\n3. CALIBRATION -- the weight, read in the regime the terrain run uses")
    p = rows(rd, PROBE)
    if not p:
        return missing(PROBE, "probe")

    for tag, lo, hi in (("iters 0-49", 0, 50), ("iters 50-99", 50, None), ("all", 0, None)):
        ls, lt, fa = mean(p, "loss", lo, hi), mean(p, "loss_tilt", lo, hi), mean(p, "tilt_frac_active", lo, hi)
        w5 = 0.05 * ls / lt if lt else float("nan")
        w15 = 0.15 * ls / lt if lt else float("nan")
        print("     %-11s loss=%7.4f  loss_tilt=%.4e  frac_active=%.4f   w@5%%=%7.0f  w@15%%=%7.0f"
              % (tag, ls, lt, fa, w5, w15))
    print("     terrain_level  %s"
          % "  ".join("i%d=%.3f" % (i, float(p[i]["terrain_level"]))
                      for i in (0, 25, 50, 75, len(p) - 1) if i < len(p)))
    print("     falls/iter     %.2f over the probe   (terrain baseline is 13.06, last200 of"
          " diag20/blind_rudin_fwd_s0)" % mean(p, "n_falls"))
    print("     -> terrain weight at 5%% of loss: %d      at 15%%: %d"
          % (rudin_weight(rd, 0.05), rudin_weight(rd, 0.15)))
    print("     Compare against the flat numbers (w=100 / w=300). If these differ by more than"
          " ~3x,\n     the flat calibration was the wrong anchor and the terrain run should use these.")


# ---------------------------------------------------------------------------
# 4. Flat arms
# ---------------------------------------------------------------------------
def section_arms(rd):
    print("\n4. FLAT ARMS -- short arms, all read over iterations %d-%d" % (ARM_LO, ARM_HI))
    base = rows(rd, BASE)
    if not base:
        return missing(BASE, "(baseline already on disk?)")

    arms = [("w=0 base", 0.0, base)]
    ctrl = rows(rd, CTRL)
    if ctrl:
        arms.append(("w=0 ctrl", 0.0, ctrl))
    any_arm = False
    for w in ARM_WEIGHTS:
        rs = rows(rd, ARM_GLOB.format(w))
        if rs:
            arms.append(("w=%d" % w, float(w), rs))
            any_arm = True
    if not any_arm:
        return missing(ARM_GLOB.format("*"), "arms")

    hdr = ("     %-10s %8s %8s %9s %11s %8s %8s %8s"
           % ("arm", "loss", "loss_v", "loss_gproj", "loss_tilt", "frac", "falls/it", "share"))
    print(hdr)
    for name, w, rs in arms:
        hi = min(ARM_HI, len(rs))
        lt = mean(rs, "loss_tilt", ARM_LO, hi)
        ls = mean(rs, "loss", ARM_LO, hi)
        share = (w * lt / ls) if (w and ls) else 0.0
        print("     %-10s %8.4f %8.4f %9.5f %11.4e %8.4f %8.3f %7.1f%%"
              % (name, ls, mean(rs, "loss_v", ARM_LO, hi), mean(rs, "loss_gproj", ARM_LO, hi),
                 lt, mean(rs, "tilt_frac_active", ARM_LO, hi), mean(rs, "n_falls", ARM_LO, hi),
                 100 * share))

    print("\n     How to read it, in order:")
    print("       share      -- what the term actually cost. Far off 5%/15% means the weight")
    print("                     was calibrated on the wrong window; recalibrate, do not retune.")
    print("       frac/gproj -- the term working. Both must FALL against the w=0 rows, and by")
    print("                     more than the w=0-vs-ctrl gap on the same line.")
    print("       loss_v     -- the cost. If it rises much, the barrier is buying uprightness")
    print("                     by refusing to walk, which fails the arm whatever falls does.")
    print("       falls/it   -- ignore on flat at 16 envs; section 2 prints why.")


# ---------------------------------------------------------------------------
# 5. Rudin smoke
# ---------------------------------------------------------------------------
def section_smoke(rd):
    print("\n5. RUDIN SMOKE -- the terrain path at the calibrated weight")
    s = rows(rd, SMOKE)
    if not s:
        return missing(SMOKE, "rudin-smoke")
    w = float(meta(rd, SMOKE).get("tilt_w", 0.0))
    print("     TILT_W = %g" % w)
    bad = []
    for r in s:
        vals = {k: float(r[k]) for k in ("loss", "loss_tilt", "tilt_frac_active", "grad_norm")}
        if any(math.isnan(v) or math.isinf(v) for v in vals.values()):
            bad.append(r["iter"])
        share = (w * vals["loss_tilt"] / vals["loss"]) if vals["loss"] else float("nan")
        print("     i%-3s loss=%8.4f  loss_tilt=%.4e  frac=%.4f  grad=%.4f  share=%5.1f%%"
              % (r["iter"], vals["loss"], vals["loss_tilt"], vals["tilt_frac_active"],
                 vals["grad_norm"], 100 * share))
    print("     [%s] no NaN/inf on the terrain path" % ("PASS" if not bad else "FAIL %s" % bad))
    print("     -> if share is in the 3-8% band, the terrain run can use this weight:")
    print("        MODE=blind_rudin_fwd TILT_W=%g SEED=0 ITERS=850 NUM_ENVS=1024 \\" % w)
    print("          RUN_DIR=results/diag21/rudin_fwd_w%g python train.py \\" % w)
    print("          2>&1 | tee results/logs/diag21_rudin_fwd_w%g.log" % w)
    print("        Baseline to beat: terrain_level 0.028 and 13.06 falls/iter over the last 200")
    print("        of results/diag20/blind_rudin_fwd_s0.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--weight", type=float, metavar="FRACTION",
                    help="print only the Rudin weight for this share of the loss, then exit")
    args = ap.parse_args()
    rd = args.results_dir

    if args.weight is not None:
        print(rudin_weight(rd, args.weight))
        return 0

    print("Tilt-barrier checks, from %s/" % rd)
    section_wiring(rd)
    section_noise(rd)
    section_calibration(rd)
    section_arms(rd)
    section_smoke(rd)
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
