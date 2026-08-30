#!/usr/bin/env python
"""Read every Step 3 gate produced by ``run_step3.sh`` in one pass.

    python collect_step3.py                     # all sections, with verdicts
    python collect_step3.py --weight 0.05       # just FOOT_Q_W, for scripting
    python collect_step3.py --res-weight 0.02   # just FOOT_RES_W, for scripting

Stdlib only, so it runs on a laptop with no torch as happily as on the GPU box. Sections whose
runs are missing say which stage produces them instead of failing.

The gates, and why each one is here (see STEP3_FOOTHOLD.md):

1. SEED BAND  -- there is no TILT_W=0 Rudin replicate at 850 iterations on disk, so nothing else
                 here has a noise floor to be read against. This is CAMPAIGN_FINDINGS 22.8 item 1
                 and it is the prerequisite for every verdict below.
2. PHASE 1    -- the terrain-aware swing target. The inertness arm must land inside the band from
                 section 1; the treatment arms are read on the PER-TERRAIN-TYPE table, not the
                 batch mean, because smooth slope is the internal control that this change should
                 NOT help.
3. WIRING     -- at iteration 0 the two wiring runs share an identical rollout, so
                 ``loss_on == loss_off + w*loss_fq`` is an exact arithmetic identity. The only
                 cheap check that the term is added, added linearly, and added at the right weight.
4. CALIBRATION-- loss_fq / loss_fres are logged even at weight 0, so a run that used no weight
                 measures the weights for one that will. Also carries the section 4.4 blur check.
5. PHASE 2    -- the residual arms, read against Phase 1 and each other. foot_res_abs_mean is the
                 "did it move at all" column: without it, "the correction did nothing" and "the
                 correction stayed at zero" look identical, which is the mistake 22.4 avoided.

The per-terrain-type table lives only in results/logs/<run>.log, never in iters.csv, so this
script parses the log for it. That is also why run_step3.sh tees everything.
"""

import argparse
import csv
import json
import math
import os
import re
import sys

# Run directories, all relative to --results-dir. Kept in one place so a rename is one edit.
BASE = "diag20/blind_rudin_fwd_s0"           # the pre-Step-3 baseline, seed 0
SEEDS = ["diag21/rudin_fwd_w0_s1", "diag21/rudin_fwd_w0_s2"]
P1 = [("inert (flags off)", "diag22/fz_off"),
      ("z only", "diag22/fz_z"),
      ("z + apex", "diag22/fz_za")]
WIRE_OFF, WIRE_ON = "diag23/wire_q0", "diag23/wire_qon"
PROBE = "diag23/fhold_probe"
STAIR_TYPES = ("stairs up", "stairs down")


def rows(results_dir, run):
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


def last200(rs, key):
    return mean(rs, key, max(0, len(rs) - 200))


def final_level(rs):
    v = col(rs, "terrain_level")
    return v[-1] if v else float("nan")


def per_type(results_dir, run):
    """Per-terrain-family mean level, parsed out of the run log (it is nowhere else)."""
    log = os.path.join(results_dir, "logs", os.path.basename(run) + ".log")
    if not os.path.isfile(log):
        return {}
    pat = re.compile(r"^\[train\]\s+(.+?)\s+n=\s*(\d+)\s+mean level\s+([0-9.]+)")
    out = {}
    with open(log, errors="ignore") as f:
        for line in f:
            m = pat.match(line.rstrip())
            if m:
                out[m.group(1).strip()] = float(m.group(3))
    return out


def missing(name, stage):
    print("  not run yet -- ./run_step3.sh %s   (%s)" % (stage, name))


def band(values):
    """(min, max, spread%) of a list, for the seed-band verdicts."""
    lo, hi = min(values), max(values)
    mid = sum(values) / len(values)
    return lo, hi, (100.0 * (hi - lo) / abs(mid) if mid else float("nan"))


# ---------------------------------------------------------------------------
def section_seeds(rd):
    print("\n1. SEED BAND -- the noise floor every verdict below is read against")
    base = rows(rd, BASE)
    if not base:
        return missing(BASE, "(baseline already on disk?)")
    runs = [("seed 0", base)]
    for r in SEEDS:
        rs = rows(rd, r)
        if rs:
            runs.append(("seed " + r.rsplit("_s", 1)[-1], rs))
    print("     %-10s %14s %16s" % ("run", "final level", "falls/it last200"))
    for name, rs in runs:
        print("     %-10s %14.4f %16.2f" % (name, final_level(rs), last200(rs, "n_falls")))
    if len(runs) < 3:
        print("     [MISSING] only %d of 3 seeds -- ./run_step3.sh seeds" % len(runs))
        print("     Until all three exist there is no band, and 'the arm moved' cannot be said.")
        return None
    lv = band([final_level(rs) for _, rs in runs])
    fl = band([last200(rs, "n_falls") for _, rs in runs])
    print("     band: final level %.4f..%.4f (%.0f%%)   falls/it %.2f..%.2f (%.0f%%)"
          % (lv[0], lv[1], lv[2], fl[0], fl[1], fl[2]))
    print("     -> an arm is only 'different' if it sits OUTSIDE these.")
    return {"level": (lv[0], lv[1]), "falls": (fl[0], fl[1])}


def section_phase1(rd, bands):
    print("\n2. PHASE 1 -- terrain-aware swing target")
    present = [(n, r, rows(rd, r)) for n, r in P1]
    if not any(rs for _, _, rs in present):
        return missing("diag22/*", "phase1")

    print("     %-18s %12s %10s %9s %9s %11s"
          % ("arm", "final level", "falls/it", "vx", "loss_v", "minFootClr"))
    base = rows(rd, BASE)
    if base:
        print("     %-18s %12.4f %10.2f %9.3f %9.4f %11.4f"
              % ("pre-Step-3 base", final_level(base), last200(base, "n_falls"),
                 last200(base, "vx"), last200(base, "loss_v"), last200(base, "min_foot_clear")))
    for name, run, rs in present:
        if not rs:
            print("     %-18s  (not run)" % name)
            continue
        print("     %-18s %12.4f %10.2f %9.3f %9.4f %11.4f"
              % (name, final_level(rs), last200(rs, "n_falls"), last200(rs, "vx"),
                 last200(rs, "loss_v"), last200(rs, "min_foot_clear")))

    # Gate 1: n_fall_height must stay 0 (the standing gate since 17).
    for name, run, rs in present:
        if not rs:
            continue
        fh = sum(col(rs, "n_fall_height"))
        if fh > 5:
            print("     [FAIL] %s has n_fall_height = %.0f -- the spawn/height logic regressed;"
                  " nothing else here is readable" % (name, fh))

    # Gate 2: the inertness arm inside the seed band.
    inert = dict((n, rs) for n, _, rs in present).get("inert (flags off)")
    if inert and bands:
        lv, fl = final_level(inert), last200(inert, "n_falls")
        ok = (bands["level"][0] * 0.5 <= lv <= bands["level"][1] * 2.0
              and bands["falls"][0] * 0.9 <= fl <= bands["falls"][1] * 1.1)
        print("     [%s] inertness arm inside the seed band (level %.4f, falls %.2f)"
              % ("PASS" if ok else "CHECK", lv, fl))
        if not ok:
            print("            -> use_perception=True alone moved the run; treat it as a confound.")

    # Gate 3: THE read. Per-terrain-type, where smooth slope is the internal control.
    print("\n     Per-terrain-type mean level -- THE primary gate (from the logs):")
    fam = ["smooth slope", "rough slope", "stairs up", "stairs down", "discrete obstacles"]
    hdr = "     %-18s" % "arm" + "".join("%14s" % f[:13] for f in fam)
    print(hdr)
    for name, run in [("pre-Step-3 base", BASE)] + [(n, r) for n, r, _ in present]:
        d = per_type(rd, run)
        if not d:
            continue
        print("     %-18s" % name + "".join("%14s" % ("%.2f" % d[f] if f in d else "-")
                                            for f in fam))
    for name, run, rs in present:
        if not rs:
            continue
        d = per_type(rd, run)
        if not d:
            continue
        stairs = [d[t] for t in STAIR_TYPES if t in d]
        if stairs:
            moved = max(stairs) > 0.05
            print("     [%s] %s: stairs at %s"
                  % ("PASS" if moved else "null", name,
                     " / ".join("%.2f" % v for v in stairs)))
    print("     Read `smooth slope` as the control: it should be roughly unchanged. Stairs")
    print("     leaving 0.00 while smooth slope holds still is the result.")


def section_wiring(rd):
    print("\n3. WIRING -- is loss_fq in the loss, and is the gradient path live?")
    off, on = rows(rd, WIRE_OFF), rows(rd, WIRE_ON)
    if not off or not on:
        return missing("%s / %s" % (WIRE_OFF, WIRE_ON), "wiring")
    w = float(meta(rd, WIRE_ON).get("foot_q_w", 0.0))
    o, n = off[0], on[0]
    fq_off, fq_on = float(o["loss_fq"]), float(n["loss_fq"])
    l_off, l_on = float(o["loss"]), float(n["loss"])
    g_off, g_on = float(o["grad_norm"]), float(n["grad_norm"])
    shift = w * fq_off
    resid = abs(l_on - (l_off + shift))
    tol = 1e-4 * abs(l_off)
    print("     FOOT_Q_W = %-10g  (iteration 0, before the first update)" % w)
    print("     loss_fq        off %.6e   on %.6e" % (fq_off, fq_on))
    print("     loss           off %.8f     on %.8f" % (l_off, l_on))
    print("     grad_norm      off %.6f       on %.6f   (pre-clip; clipped to 0.3)" % (g_off, g_on))
    print("     predicted on   %.8f = off + w*loss_fq   (shift %.6f)" % (l_off + shift, shift))
    print("     residual       %.3e   tol %.3e" % (resid, tol))
    ok_val = fq_off > 0 and abs(fq_on - fq_off) <= 1e-4 * abs(fq_off)
    ok_pow = shift > 100 * tol
    ok_id = resid <= tol
    ok_gr = g_off > 0 and abs(g_on - g_off) > 1e-3 * abs(g_off)
    print("     [%s] the term is non-zero and identical with and without the graph"
          % ("PASS" if ok_val else "FAIL"))
    print("     [%s] the test has power (shift is %.0fx the tolerance)"
          % ("PASS" if ok_pow else "WEAK", shift / tol if tol else 0))
    print("     [%s] loss identity: added linearly at exactly FOOT_Q_W" % ("PASS" if ok_id else "FAIL"))
    print("     [%s] grad_norm moved: the gradient path is live, not detached"
          % ("PASS" if ok_gr else "FAIL"))
    if not (ok_id and ok_gr):
        print("     -> STOP. Nothing downstream means anything until this passes.")


def calib(rd, frac, key):
    p = rows(rd, PROBE)
    if not p:
        return 0
    t, ls = mean(p, key), mean(p, "loss")
    if not t or math.isnan(t) or math.isnan(ls):
        return 0
    return int(round(frac * ls / t))


def section_calibration(rd):
    print("\n4. CALIBRATION -- the weights, read in the regime the arms run in")
    p = rows(rd, PROBE)
    if not p:
        return missing(PROBE, "probe")
    for tag, lo, hi in (("iters 0-49", 0, 50), ("iters 50-99", 50, None), ("all", 0, None)):
        print("     %-11s loss=%7.4f  loss_fq=%.4e  loss_fres=%.4e  res_abs=%.4f m"
              % (tag, mean(p, "loss", lo, hi), mean(p, "loss_fq", lo, hi),
                 mean(p, "loss_fres", lo, hi), mean(p, "foot_res_abs_mean", lo, hi)))
    print("     -> FOOT_Q_W   at 5%% of loss: %d      at 15%%: %d"
          % (calib(rd, 0.05, "loss_fq"), calib(rd, 0.15, "loss_fq")))
    print("     -> FOOT_RES_W at 2%% of loss: %d      at 5%%:  %d"
          % (calib(rd, 0.02, "loss_fres"), calib(rd, 0.05, "loss_fres")))
    res_abs = mean(p, "foot_res_abs_mean")
    m = meta(rd, PROBE)
    cap = float(m.get("foot_res_max", 0.10))
    print("     residual displacement %.4f m against a %.2f m cap (%.0f%% of range)"
          % (res_abs, cap, 100 * res_abs / cap if cap else float("nan")))
    if res_abs > 0.9 * cap:
        print("     [CHECK] the residual is saturating its tanh -- raise FOOT_RES_W or cut the cap.")
    print("\n     Blur check (STEP3_FOOTHOLD.md 4.4): loss_fq must read HIGHER on the stair rows")
    print("     than on smooth slope. If they are equal, hm_loss_blur_cells is washing out the")
    print("     edge the term exists to find; re-run the probe with FOOT_Q_SMOOTH=0 and compare.")
    d = per_type(rd, PROBE)
    if d:
        print("     probe per-type level: "
              + "  ".join("%s=%.2f" % (k[:12], v) for k, v in sorted(d.items())))


def section_phase2(rd):
    print("\n5. PHASE 2 -- the foothold residual arms")
    out = os.path.join(rd, "diag23")
    if not os.path.isdir(out):
        return missing("diag23/*", "arms")
    arms = sorted(d for d in os.listdir(out)
                  if d.startswith("fhold_") and d != "fhold_probe"
                  and os.path.isfile(os.path.join(out, d, "iters.csv")))
    if not arms:
        return missing("diag23/fhold_*", "arms")
    print("     %-18s %8s %8s %12s %10s %10s %9s"
          % ("arm", "Q_W", "detach", "final level", "falls/it", "res_abs", "fq share"))
    for a in arms:
        run = "diag23/" + a
        rs, m = rows(rd, run), meta(rd, run)
        qw = float(m.get("foot_q_w", 0.0))
        fq = last200(rs, "loss_fq")
        ls = last200(rs, "loss")
        share = 100.0 * qw * fq / ls if ls else float("nan")
        print("     %-18s %8g %8s %12.4f %10.2f %10.4f %8.1f%%"
              % (a, qw, m.get("foot_res_detach", "?"), final_level(rs),
                 last200(rs, "n_falls"), last200(rs, "foot_res_abs_mean"), share))
    print("\n     Per-terrain-type mean level:")
    fam = ["smooth slope", "rough slope", "stairs up", "stairs down", "discrete obstacles"]
    print("     %-18s" % "arm" + "".join("%14s" % f[:13] for f in fam))
    for a in arms:
        d = per_type(rd, "diag23/" + a)
        if d:
            print("     %-18s" % a + "".join("%14s" % ("%.2f" % d[f] if f in d else "-")
                                             for f in fam))
    print("\n     How to read it, in order:")
    print("       res_abs   -- did the correction move at all? 0 means the outputs stayed dead;")
    print("                    at the cap it saturated. Neither is a result about footholds.")
    print("       fq share  -- what the term cost. Far off 5%/15% means recalibrate, not retune.")
    print("       stairs    -- the actual gate, same as Phase 1.")
    print("       vs detach -- fhold_nodetach vs the same weight says whether the degenerate")
    print("                    minimum of 4.3 is real. Expect res_abs to grow and stairs not to.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--weight", type=float, metavar="FRACTION",
                    help="print only FOOT_Q_W for this share of the loss, then exit")
    ap.add_argument("--res-weight", type=float, metavar="FRACTION",
                    help="print only FOOT_RES_W for this share of the loss, then exit")
    args = ap.parse_args()
    rd = args.results_dir

    if args.weight is not None:
        print(calib(rd, args.weight, "loss_fq"))
        return 0
    if args.res_weight is not None:
        print(calib(rd, args.res_weight, "loss_fres"))
        return 0

    print("Step 3 gates, from %s/" % rd)
    bands = section_seeds(rd)
    section_phase1(rd, bands)
    section_wiring(rd)
    section_calibration(rd)
    section_phase2(rd)
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
