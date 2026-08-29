# -*- coding: utf-8 -*-
"""
collect_saliency.py -- read the sec 6.8 corruption diagnostic and apply its decision rule.

Pure stdlib (no torch, no numpy) so it runs on the dev machine, same as the CAMPAIGN_FINDINGS
sec 19.4 reader. Point it at the folder run_saliency.sh wrote:

    python collect_saliency.py [results/saliency]

For each policy it prints the clean baseline, the noise floor measured from the two `none` runs,
and each corruption arm's deviation from clean -- then says whether that deviation clears the
floor. The floor matters: sec 19.14 established this sim is not bit-reproducible run-to-run, and
the reprocheck gate was originally written as an exact match, which could only ever fail.
"""

import json
import os
import sys

# Metrics the decision rule reads. Behavioural outcomes only -- corrupt_l1 / channel_l1 are
# reported separately because they describe the *input*, not the policy's response to it.
METRICS = ("mean_terrain_level", "lin_vel_track_err", "ang_vel_track_err", "fall_rate")
POLICIES = ("hobs", "height", "depth")
ARMS = ("zero", "shuffle")


def load(d, tag):
    """Return the metrics dict for <d>/eval_results_<tag>.json, or None if absent."""
    p = os.path.join(d, "eval_results_%s.json" % tag)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)["metrics"]


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "saliency")
    if not os.path.isdir(d):
        raise SystemExit("no such directory: %s (run ./run_saliency.sh matrix first)" % d)

    blind = load(d, "blind_rudin_none")
    if blind:
        print("Terrain-blind floor (blind_rudin_s0, clean):")
        for m in METRICS:
            print("  %-22s %.6g" % (m, blind[m]))
        print("  A perception policy whose CLEAN score does not beat this is not using the")
        print("  terrain signal either -- that conclusion needs no corruption run at all.")
    print("")

    for pol in POLICIES:
        clean, clean2 = load(d, pol + "_none"), load(d, pol + "_none2")
        if clean is None:
            print("== %-7s -- not run" % pol)
            continue
        print("== %s" % pol)
        print("   channel_l1 (mean |clean terrain input|) = %.6g" % clean["channel_l1"])
        if clean["channel_l1"] < 1e-3:
            print("   !! channel is near-constant: the robots saw almost no terrain geometry.")
            print("      A null here is NOT evidence the policy ignores terrain -- there was")
            print("      nothing to ignore. Take the plan's fixed-level fallback once, then stop.")
        print("   terrain level entering the window: %s"
              % clean.get("terrain_level_hist_start", "n/a"))

        if clean2 is None:
            print("   !! no `none2` run -- cannot measure the noise floor, so no arm can be")
            print("      called significant. Run it before drawing any conclusion.")
            floor = None
        else:
            floor = {m: abs(clean2[m] - clean[m]) for m in METRICS}

        print("   %-22s %-12s %-12s %-12s %s"
              % ("metric", "clean", "zero", "shuffle", "1.5x floor"))
        arms = {a: load(d, pol + "_" + a) for a in ARMS}
        for m in METRICS:
            row = "   %-22s %-12.6g" % (m, clean[m])
            for a in ARMS:
                row += " %-12s" % ("%.6g" % arms[a][m] if arms[a] else "--")
            row += " %-12s" % ("%.4g" % (1.5 * floor[m]) if floor else "?")
            print(row)

        for a in ARMS:
            if arms[a] is None:
                continue
            print("   -- %s: corrupt_l1 = %.6g" % (a, arms[a]["corrupt_l1"]))
            if arms[a]["corrupt_l1"] < 1e-6:
                print("      !! the corruption changed nothing. Check --selftest wiring.")
                continue
            if floor is None:
                continue
            # 1.5x the measured floor, matching the CAMPAIGN_FINDINGS sec 19.14 gate: the floor
            # comes from a single pair of clean runs, so it is a noisy estimate and a bare
            # comparison over-reports. The absolute term keeps a metric whose two clean runs
            # happened to agree exactly from making every rounding difference "significant".
            beats = [m for m in METRICS
                     if abs(arms[a][m] - clean[m]) > max(1.5 * floor[m], 1e-9)]
            if beats:
                print("      effect EXCEEDS the noise floor on: %s" % ", ".join(beats))
                print("      -> the policy does use the terrain signal; report the effect size.")
            else:
                print("      effect is WITHIN the noise floor on every metric")
                print("      -> the policy is invariant to its terrain input. This is the sec 6.8")
                print("         result and the mechanism claim for sec E.3.")
        print("")


if __name__ == "__main__":
    main()
