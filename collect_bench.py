"""Merge the per-run folders written by bench_log.py into thesis tables and figures.

Reads every ``*/summary.json`` (+ ``meta.json``, + ``iters.csv``) under a results
root and produces:

    runs.csv                one row per run -- the table for the thesis
    fig_speed_itps.png      Experiment 1: iterations/s vs. num_envs, one line per variant
    fig_speed_throughput.png    Experiment 1: robot-steps/s vs. num_envs -- the metric
                            that shows batching paying off, where it/s alone does not
    fig_speed_mem.png       Experiment 1: peak PyTorch-allocated GPU memory vs. num_envs
    fig_terrain_level.png   Experiment 2: mean terrain level vs. simulated time,
                            one line per condition, mean over seeds with a min/max band
    fig_curriculum_final.png    where each condition ENDED UP: the distribution of
                            robots over the 10 difficulty rows at the end of training
    fig_terrain_by_type.png     mean terrain level per terrain family per condition
    terrain_by_type.csv     the numbers behind that figure -- mean AND max level per
                            family, because at the collapse floor the batch mean is a
                            handful of robots on smooth slope and max is the honest
                            statistic
    fig_campaign_arms.png   the intervention campaign as small multiples: one panel per
                            intervention family, each arm read against the blind seed band
    fig_campaign_curriculum_final.png   difficulty-row distribution, one row per arm
    campaign_terrain_by_type.csv        per-family mean/max for every arm

Everything is derived from the run folders, so runs made days apart, or in a
different checkout, land on the same axes as long as they were logged by
``bench_log.py``.

Runs are sorted into experiments by their folder (``bench/`` -> Experiment 1,
``train/`` -> Experiment 2, ``smoke/`` -> sizing probe, ``control/`` -> flat-ground
learning-parity check) and each figure draws only the experiment it belongs to. Any
other folder holding curriculum runs (the ad-hoc ``diag*/`` RUN_DIRs) is the
intervention campaign and gets its own figures -- it is a different task from
Experiment 2 (forward commands, 1024 robots, 850 iterations against omnidirectional
commands, 2048 robots, 5000 iterations) and the two must never share an axis.

``train_invalid/`` is the pre-ground-contact-fix run set. It carries the same mode
names as ``train/``, so it is excluded by name; without that it silently doubles
every Experiment 2 series and drags retired numbers into the figures.

``runs.csv`` is the thesis table, so it carries **Experiment 1 and 2 only**. Smoke
probes and control runs are diagnostics -- they answer "does this work?", not "what
did we measure?" -- and are left out; pass ``--include-diagnostics`` to list them
too. Either way the run counts per experiment are printed, so nothing is hidden
silently.

Note on memory: ``peak_mem_mb`` is ``torch.cuda.max_memory_allocated()``, i.e. the
PyTorch allocator only. It excludes Isaac Gym / PhysX buffers, the terrain trimesh
and the CUDA context, so it is a fair variant-to-variant comparison but NOT total
GPU usage. The figure is labelled accordingly.

Usage:
    python collect_bench.py                                   # scans results/
    python collect_bench.py --results-dir results --out results/figures
    python collect_bench.py --exp1-only                       # skip the training figures

    # optional: draw the inherited (pre-rewrite) build as a reference point
    python collect_bench.py --inherited-s-per-iter 4.64 --inherited-num-envs 16

CPU-only: needs numpy + matplotlib, not torch or Isaac Gym.
"""

import argparse
import collections
import csv
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical palette, light mode, in the documented fixed slot order (blue,
# orange, aqua, yellow, magenta, green). Colour follows the entity, never its
# rank, so a run missing from a figure never repaints the others. Yellow and
# magenta fall below 3:1 on a white surface, which is why every line also carries
# an end label and every figure has runs.csv behind it.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# Stable slot per entity. Experiment 1 variants first, then Experiment 2 conditions.
COLOR_BY_NAME = {
    "inherited": 0, "torch": 1, "cuda": 2, "torch-fp64": 3, "cuda-fp64": 4,
    "blind_rudin": 0, "hobs": 1, "hloss": 2, "height": 3, "depth": 4, "blind": 5,
}
LABEL = {
    "blind_rudin": "blind", "hobs": "height map (obs)", "hloss": "gradient (loss)",
    "height": "gradient + map", "depth": "camera",
}

# Campaign small multiples. Each panel is one controlled comparison, so it carries at
# most three arms plus the blind seed band -- which is what keeps every panel inside
# the three categorical hues that clear the all-pairs CVD and normal-vision gates
# (blue/orange/aqua; adding yellow drops the worst normal-vision pair to dE 13.7).
# Identity is never colour alone: every line is labelled at its right end.
CAMPAIGN_PANELS = [
    ("Observation and commands", "terrain in the observation; random vs. forward commands",
     ["height map (obs)", "random commands"]),
    ("Attitude - Step 1(c)", "soft tilt barrier, two weights",
     ["tilt w=16", "tilt w=48"]),
    ("Swing target - Step 3 Phase 1", "terrain-aware landing height and apex",
     ["swing target off", "swing target z", "swing target z+apex"]),
    ("Foothold residual - Step 3 Phase 2",
     "explicit foot-placement outputs (450 iterations, half the baseline)",
     ["foothold residual unsupervised", "foothold residual x",
      "foothold residual x+y"]),
    ("Gradient window - Step 4", "48 differentiated physics steps per update, not 24",
     ["window 48 steps (41 s)", "window 48 steps (82 s)"]),
]
CAMPAIGN_BASELINE = "blind"          # the 3-seed band every panel is read against
BASELINE_INK = "#8a8985"             # neutral: the reference is not a category
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# Difficulty rows are ordered magnitude, not identity, so they get a single-hue
# sequential ramp rather than categorical hues. Ordinal steps 250->700: the
# lightest still clears 2:1 on a white surface, which a true sequential ramp's
# 100 step would not.
DIFFICULTY_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                   "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# Reading order for the terrain families, easiest to hardest.
FAMILY_ORDER = ["smooth slope", "rough slope", "stairs down", "stairs up",
                "discrete obstacles"]


def color_for(name, fallback_index=0):
    """Fixed palette slot for a run label; unknown names take the next free slot."""
    return PALETTE[COLOR_BY_NAME.get(name, fallback_index) % len(PALETTE)]


# Which experiment a run belongs to. The campaign writes each run into
# results/<experiment>/<name>/, so the top-level folder is the primary signal; the
# fallback keeps ad-hoc RUN_DIRs classifiable.
EXPERIMENT_BY_PREFIX = {"bench": "exp1", "train": "exp2", "smoke": "smoke",
                        "control": "control"}
SMOKE_MAX_ITERS = 100      # a sizing probe is shorter than a bench run

# Run sets that are no longer results. `train_invalid/` is the Experiment 2 campaign
# from before the ground-contact fix, retired in CAMPAIGN_FINDINGS §17.6. It uses the
# same mode names as `train/`, so it groups straight into the live series unless it is
# named here -- and because seed_label() counts DISTINCT seeds, the doubled group still
# reported "3 seeds" while averaging six runs.
RETIRED_PREFIXES = {"train_invalid"}


def experiment_of(run):
    """'exp1' (speed), 'exp2' (locomotion), 'smoke', 'control' or 'other'.

    Mixing these is what broke the old figures: the 30-iteration smoke runs landed
    in the Experiment 2 groups, which inflated every n= and -- far worse -- pulled
    ``min(len(curve))`` down to 30, truncating the 5000-iteration curves.

    'campaign' is every curriculum run outside train/: the intervention arms, run with
    forward commands at 1024 robots for 850 iterations. They answer a different question
    on a different task from Experiment 2 and get their own figures.

    'control' is every flat-ground blind run that is not a bench run: the
    learning-parity check against the inherited build, and the flat baseline-
    reproduction runs under the ad-hoc diag RUN_DIRs. They are blind + flat like a
    bench run, so the `mode == "blind"` fallback used to call them Experiment 1 and
    drop a second marker on top of bench/cuda_B16 -- `diag19/flat_fwd_s0` is 16
    robots on flat ground at 2.69 it/s, which sat right beside the 2.58 it/s bench
    point and read as a duplicate measurement. Experiment 1 is the `bench/` sweep and
    nothing else: a run that did not vary num_envs measures no scaling curve. Control
    runs stay out of the figures, and out of runs.csv unless --include-diagnostics.
    """
    prefix = run["dir"].split("/")[0]
    if prefix in RETIRED_PREFIXES:
        return "retired"
    if prefix in EXPERIMENT_BY_PREFIX:
        return EXPERIMENT_BY_PREFIX[prefix]
    meta = run["meta"]
    if meta.get("mode") == "blind":               # flat ground, no curriculum
        return "control"
    iters = meta.get("iters")
    if isinstance(iters, int) and iters <= SMOKE_MAX_ITERS:
        return "smoke"
    return "campaign" if meta.get("terrain_type") == "rudin" else "other"


def select(runs, *experiments):
    """The subset of `runs` belonging to the given experiments."""
    return [r for r in runs if experiment_of(r) in experiments]


def seeds_of(runs):
    """Distinct seeds across `runs`, sorted; runs with no recorded seed are skipped."""
    return sorted({r["meta"].get("seed") for r in runs
                   if r["meta"].get("seed") is not None})


def seed_label(runs):
    """'3 seeds' / '1 seed' -- what the mean and the band are actually taken over.

    Counting runs (the old behaviour) reports 4 when three of them are seeds 0/1/2
    and the fourth is a smoke probe. Counting distinct seeds cannot.
    """
    n = len(seeds_of(runs)) or len(runs)
    return f"{n} seed" + ("s" if n != 1 else "")


def steps_per_iter_of(run):
    """Physics steps differentiated per training iteration (the BPTT window)."""
    return int(run["meta"].get("steps_per_iter") or 24)


def sim_seconds_of(run):
    """Total simulated seconds per robot the run differentiated through."""
    meta = run["meta"]
    dt_sim = float(meta.get("sim_seconds_per_iter") or 0.048)
    return dt_sim * int(meta.get("iters") or 0)


def condition_of(run):
    """The experimental ARM a run belongs to -- what every figure must group by.

    `mode` stopped identifying a condition once the campaign began varying
    interventions through the environment rather than through the mode name: the three
    blind seeds, both tilt-barrier weights and both gradient-window arms all log
    ``mode = "blind_rudin_fwd"``. Grouping by mode averaged seven different arms into a
    single line labelled "blind" -- the treatment quietly became part of its own
    control. This reads the distinguishing flags back out of meta.json instead, so each
    arm is its own series with its own seed count.

    Experiment 2's five conditions ARE its five modes, so there the mode is returned
    unchanged.
    """
    meta = run["meta"]
    mode = meta.get("mode")
    if experiment_of(run) != "campaign":
        return mode

    bits = []
    spi = steps_per_iter_of(run)
    if spi != 24:
        # Two window arms exist at the same width and differ only in how long they ran
        # (matched simulated time vs. twice it), so the label has to carry the duration.
        bits.append(f"window {spi} steps ({sim_seconds_of(run):.0f} s)")
    if float(meta.get("tilt_w") or 0.0) > 0.0:
        bits.append(f"tilt w={float(meta['tilt_w']):g}")
    if meta.get("foot_res"):
        if float(meta.get("foot_q_w") or 0.0) > 0.0:
            bits.append("foothold residual " + ("x+y" if meta.get("foot_res_y") else "x"))
        else:
            bits.append("foothold residual unsupervised")
    elif meta.get("foot_apex_terrain"):
        bits.append("swing target z+apex")
    elif meta.get("foot_z_terrain"):
        bits.append("swing target z")
    elif mode in ("fz_fwd", "fhold_fwd"):
        bits.append("swing target off")     # flags present but every one of them off
    if mode == "hobs_fwd":
        bits.append("height map (obs)")
    elif mode == "blind_rudin_rand":
        bits.append("random commands")
    return ", ".join(bits) if bits else "blind"


def robot_steps_per_s(run):
    """Throughput: robot experience generated per wall-second.

    it/s falls as the batch grows, which makes fig_speed_itps read as if scaling
    hurts. This is the number that says otherwise -- and multiplied by dt it is
    also what puts this project and legged_gym on one axis.
    """
    ips = run["summary"].get("it_per_s_median")
    B = run["meta"].get("num_envs")
    if ips is None or B is None:
        return None
    return float(ips) * int(B) * steps_per_iter_of(run)


def load_runs(root):
    """Every run folder under `root` that bench_log.py finished writing."""
    runs = []
    for dirpath, _, filenames in os.walk(root):
        if "summary.json" not in filenames:
            continue
        with open(os.path.join(dirpath, "summary.json")) as f:
            summary = json.load(f)
        meta = {}
        meta_path = os.path.join(dirpath, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        runs.append({
            "dir": os.path.relpath(dirpath, root).replace("\\", "/"),
            "summary": summary,
            "meta": meta,
        })
    return sorted(runs, key=lambda r: r["dir"])


def variant_of(run):
    """Experiment 1 label: the SRBD backend, with the solver precision if not the default."""
    meta = run["meta"]
    backend = meta.get("srbd_backend", "torch")
    dtype = meta.get("force_dtype", "fp32")
    return backend if dtype == "fp32" else f"{backend}-{dtype}"


def write_table(runs, out_dir):
    """One row per run: what it was, and what it measured."""
    fields = ["dir", "experiment", "condition", "mode", "variant", "num_envs", "seed",
              "steps_per_iter", "sim_seconds", "iters_timed",
              "it_per_s_median", "it_per_s_p10", "it_per_s_p90", "robot_steps_per_s",
              "peak_mem_mb", "total_wall_s", "final_terrain_level", "final_loss",
              "final_vx", "gpu"]
    path = os.path.join(out_dir, "runs.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            row = {"dir": r["dir"], "experiment": experiment_of(r),
                   "condition": condition_of(r), "variant": variant_of(r),
                   "gpu": r["meta"].get("gpu"),
                   "steps_per_iter": steps_per_iter_of(r),
                   "sim_seconds": round(sim_seconds_of(r), 2),
                   "robot_steps_per_s": robot_steps_per_s(r)}
            row.update({k: v for k, v in r["summary"].items() if k in fields})
            row.setdefault("num_envs", r["meta"].get("num_envs"))
            row.setdefault("mode", r["meta"].get("mode"))
            row.setdefault("seed", r["meta"].get("seed"))
            w.writerow(row)
    print(f"[collect] wrote {path} ({len(runs)} runs)")
    return path


def _style(ax, xlabel, ylabel, title):
    """Recessive grid and axes; the data is the only prominent thing."""
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)


def _end_labels(ax, entries, headroom=0.22):
    """Label each line at its right end, nudged apart so labels never overlap.

    `entries` is a list of (x, y, text, color). Reserves horizontal room first,
    then walks the labels bottom-up enforcing a minimum vertical gap -- without
    this, converging lines stack their labels on top of each other.
    """
    if not entries:
        return
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x0 + (x1 - x0) * (1.0 + headroom))
    # Spread in DISPLAY space, not data space. A gap measured as a fraction of the
    # y-range is meaningless on the symlog axis these figures need: it is enormous at
    # the top and invisible at the floor, which threw every label in the collapsed
    # region far above the line it belongs to. Pixels are what "do not overlap" means.
    entries = sorted(entries, key=lambda e: e[1])
    trans = ax.transData                       # after set_xlim, or it is the old one
    pts = [trans.transform((e[0], e[1])) for e in entries]
    ys = [p[1] for p in pts]
    min_gap_px = 20.0        # 8pt text is ~18px tall at dpi 160
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap_px:
            ys[i] = ys[i - 1] + min_gap_px
    inv = trans.inverted()
    for (x, _, text, color), px, y_px in zip(entries, pts, ys):
        y = inv.transform((px[0], y_px))[1]
        ax.annotate(f"  {text}", (x, y), color=color, fontsize=8, va="center",
                    ha="left", fontweight="bold", annotation_clip=False)


def plot_scaling(runs, out_dir, value_fn, ylabel, title, filename, logy,
                 caption=None, inherited=None):
    """Experiment 1: one line per variant, x = num_envs (log).

    `runs` is expected to be already narrowed to Experiment 1 (see `select`).
    `value_fn(run) -> float | None` supplies the y value, so a figure can plot a
    derived quantity (robot-steps/s) as easily as a logged one.
    `inherited` is an optional (num_envs, y) reference point for the pre-rewrite
    build, which has no run folder of its own.
    """
    series = {}
    for r in runs:
        B, y = r["meta"].get("num_envs"), value_fn(r)
        if B is None or y is None:
            continue
        series.setdefault(variant_of(r), []).append((int(B), float(y)))
    if not series:
        print(f"[collect] no Experiment 1 runs found, skipping {filename}")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    for i, (name, pts) in enumerate(sorted(series.items())):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c = color_for(name, i)
        # The variants converge at large batch, so direct end labels would sit on
        # top of each other; identity comes from the legend, backed by runs.csv.
        ax.plot(xs, ys, "-o", color=c, linewidth=2, markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, label=name, zorder=3)
    xticks = {x for pts in series.values() for x, _ in pts}
    if inherited is not None:
        # A single measured point, not a curve -- drawn as a lone marker so it can
        # never be read as a scaling line it has no data for.
        bx, by = inherited
        ax.plot([bx], [by], "D", color=color_for("inherited", 0), markersize=8,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4,
                linestyle="none", label="inherited (single measurement)")
        xticks.add(int(bx))
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    else:
        # These are rates and byte counts: zero is a real value, and every one of
        # these figures is read as a ratio ("cuda is 26% faster", "4096 robots cost
        # 68x the memory of 16"). Anchoring the axis at 0 is what makes those ratios
        # readable straight off the gridlines instead of off a floating baseline.
        ax.set_ylim(bottom=0)
    ax.set_xticks(sorted(xticks))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _style(ax, "parallel robots (num_envs)", ylabel, title)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="best")
    if caption:
        # Wrapped by hand: fig.text does no wrapping, and an unwrapped caption
        # runs off the right edge of the canvas instead of being clipped visibly.
        lines = textwrap.wrap(caption, width=96)
        fig.text(0.012, 0.012, "\n".join(lines), color=INK_MUTED, fontsize=7,
                 ha="left", va="bottom", linespacing=1.5)
        fig.tight_layout(rect=(0, 0.035 + 0.030 * len(lines), 1, 1))
    else:
        fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"[collect] wrote {path}")


def _rolling(y, window):
    """Centred, nan-aware rolling mean.

    Once the curriculum has collapsed, `terrain_level` is a handful of robots hopping
    between rows and the raw curve is a band of noise around 0.02. Smoothing is what
    makes the post-collapse region -- the region every campaign arm is judged in --
    readable at all; the figure says the window it used.
    """
    if window <= 1:
        return y
    k = np.ones(window)
    present = np.isfinite(y).astype(float)
    total = np.convolve(np.nan_to_num(y), k, mode="same")
    count = np.convolve(present, k, mode="same")
    return np.where(count > 0, total / np.maximum(count, 1e-9), np.nan)


def _curve_stack(entries):
    """(x, stack) for a group of runs, on ONE simulated-time axis.

    Runs in a group can differ in simulated seconds per iteration -- the gradient-window
    arms differentiate 48 physics steps per update, so each of their iterations is 0.096
    simulated seconds against the baseline's 0.048. The old code took the group's first
    entry's dt for everybody, which drew those arms at half their true x. Every curve is
    resampled onto a shared grid at the group's finest step instead, and stays nan past
    its own end so a shorter run never invents data.
    """
    span = max(a.size * dt for a, dt, _ in entries)
    step = min(dt for _, dt, _ in entries)
    x = np.arange(0.0, span + step * 0.5, step)
    stack = np.full((len(entries), x.size), np.nan)
    for row_i, (a, dt, _) in enumerate(entries):
        xi = (np.arange(a.size) + 1) * dt
        inside = x <= xi[-1]
        stack[row_i, inside] = np.interp(x[inside], xi, a)
    return x, stack


def _terrain_curves(runs, results_dir):
    """{condition: [(terrain_level array, sim seconds per iteration, run), ...]}."""
    by_cond = {}
    for r in runs:
        if r["meta"].get("mode") in (None, "blind"):    # flat runs have no curriculum
            continue
        csv_path = os.path.join(results_dir, r["dir"], "iters.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        vals = []
        for row in rows:
            try:
                vals.append(float(row.get("terrain_level", "")))
            except ValueError:
                vals.append(np.nan)
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            continue
        dt_sim = float(r["meta"].get("sim_seconds_per_iter", 0.048))
        by_cond.setdefault(condition_of(r), []).append((arr, dt_sim, r))
    return by_cond


def _floor_yaxis(ax, linthresh, ticks):
    """Symmetric-log y below `linthresh`, so both halves of the story are legible.

    A curriculum curve falls from ~2.5 to a floor two to three orders of magnitude
    lower and then stays there. On a linear axis every condition is one flat line on
    zero and the figure says nothing; a plain log axis drops the exact zeros. Symlog
    keeps 0 and expands the floor, which is the region every condition is judged in.
    """
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
    ax.set_ylim(0, ticks[-1])
    ax.set_yticks(ticks)
    # ScalarFormatter pads every tick to the precision of the smallest one ("2.500"),
    # which reads as false precision on a curriculum level.
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def plot_terrain_level(runs, results_dir, out_dir, filename="fig_terrain_level.png",
                       smooth=100):
    """Experiment 2: mean terrain level vs simulated robot-seconds, mean + min/max band."""
    by_mode = _terrain_curves(runs, results_dir)
    if not by_mode:
        print("[collect] no Experiment 2 runs with a terrain_level column, skipping figure")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=160)
    labels = []
    for i, (mode, entries) in enumerate(sorted(by_mode.items())):
        # Resample to the LONGEST run and average with nan-aware ops, rather than
        # truncating to the shortest. Truncating let one short run chop the curve
        # for every seed -- that is what pinned this figure to 30 iterations while
        # smoke runs were still being grouped in here.
        x, stack = _curve_stack(entries)
        mean = _rolling(np.nanmean(stack, axis=0), smooth)
        c = color_for(mode, i)
        label = LABEL.get(mode, mode)
        if stack.shape[0] > 1:              # band only means something across seeds
            ax.fill_between(x, _rolling(np.nanmin(stack, axis=0), smooth),
                            _rolling(np.nanmax(stack, axis=0), smooth),
                            color=c, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, mean, color=c, linewidth=2, zorder=3,
                label=f"{label}  ({seed_label([e[2] for e in entries])})")
        end = np.flatnonzero(np.isfinite(mean))
        labels.append((x[end[-1]], float(mean[end[-1]]), label, c))
    # 1/2048 robots is 0.00049, so the floor lives between 0.001 and 0.005: a linear
    # axis draws all five conditions as the same line on zero.
    _floor_yaxis(ax, 0.001, [0, 0.001, 0.005, 0.02, 0.1, 0.5, 2.5])
    _style(ax, "simulated time per robot (s)", "mean terrain level",
           "Curriculum progress by perception condition")
    # No caption burned into the image: the thesis caption carries the smoothing window,
    # the symlog threshold and what the band is, so the two can never disagree.
    _end_labels(ax, labels)
    # "best" put the legend on top of the end labels; the floor is the empty corner.
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="lower left")
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"[collect] wrote {path}")


def _load_final_states(runs, results_dir):
    """{condition: [(final_state.json, run), ...]} for every run that reached the end.

    The run travels with its state so the figures can label by distinct seed
    rather than by run count.
    """
    by_mode = {}
    for r in runs:
        if r["meta"].get("mode") in (None, "blind"):
            continue
        path = os.path.join(results_dir, r["dir"], "final_state.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            by_mode.setdefault(condition_of(r), []).append((json.load(f), r))
    return by_mode


def plot_curriculum_final(runs, results_dir, out_dir, filename="fig_curriculum_final.png",
                          title=None):
    """Where each condition ended up: share of robots per difficulty row.

    The mean terrain level hides its own distribution -- 4.5 could be every robot
    on row 4-5, or half on row 0 and half on row 9. This is that distribution,
    pooled across seeds.
    """
    by_mode = _load_final_states(runs, results_dir)
    if not by_mode:
        print("[collect] no final_state.json found, skipping", filename)
        return

    modes = [m for m in sorted(by_mode, key=lambda m: -np.mean(
        [st["mean_terrain_level"] for st, _ in by_mode[m]]))]
    num_rows = by_mode[modes[0]][0][0]["num_rows"]
    ramp = [DIFFICULTY_RAMP[int(round(i * (len(DIFFICULTY_RAMP) - 1) / max(1, num_rows - 1)))]
            for i in range(num_rows)]

    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(modes) + 2.0), dpi=160)
    for row_i, mode in enumerate(modes):
        states = [st for st, _ in by_mode[mode]]
        counts = np.sum([st["terrain_level_hist"] for st in states], axis=0).astype(float)
        share = counts / max(counts.sum(), 1.0)
        left = 0.0
        for lvl in range(num_rows):
            w = share[lvl]
            if w <= 0:
                continue
            # 2px surface gap between segments (linewidth is in points at dpi 160).
            ax.barh(row_i, w, left=left, height=0.62, color=ramp[lvl],
                    edgecolor="white", linewidth=0.9, zorder=3)
            if w > 0.055:      # label only segments wide enough to hold a numeral
                ax.text(left + w / 2, row_i, str(lvl), ha="center", va="center",
                        fontsize=7.5, zorder=4,
                        color="white" if lvl >= num_rows // 2 else INK)
            left += w
        mean = float(np.mean([st["mean_terrain_level"] for st in states]))
        top = int(max(st["max_terrain_level"] for st in states))
        ax.text(1.015, row_i, f"mean {mean:.2f}   max {top}", va="center", ha="left",
                fontsize=8, color=INK_MUTED, transform=ax.get_yaxis_transform())

    # Arms of different length are not comparable on terrain level -- the Rudin
    # curriculum is still collapsing for the first ~300 iterations -- and this figure
    # sorts by mean, which puts the shortest runs on top. Disclose the length on the
    # row whenever the rows do not all share one.
    span = {m: round(np.mean([sim_seconds_of(r) for _, r in by_mode[m]])) for m in modes}
    show_span = len(set(span.values())) > 1
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(
        [f"{LABEL.get(m, m)}  ({seed_label([r for _, r in by_mode[m]])}"
         + (f", {span[m]:.0f} s)" if show_span else ")")
         for m in modes], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.invert_yaxis()
    # loc="left" titles are not wrapped by matplotlib and the y labels push the axes
    # more than a third of the way across, so a one-line title runs off the canvas.
    heading = title or ("Where training ended up: robots per difficulty row "
                        f"(0 easiest, {num_rows - 1} hardest)")
    _style(ax, "share of robots", "", "\n".join(textwrap.wrap(heading, width=58)))
    ax.grid(True, axis="y", color="white", linewidth=0)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"[collect] wrote {path}")


def plot_terrain_by_type(runs, results_dir, out_dir, filename="fig_terrain_by_type.png",
                        csv_name="terrain_by_type.csv", chart=True):
    """Per-family curriculum reached -- stairs is where perception should show.

    The CSV carries **mean and max** level per family. At the collapse floor the batch
    mean is a handful of robots sitting on smooth slope and its sampling spread is wider
    than three seeds can estimate, so max level per family -- did ANY robot ever clear a
    riser? -- is the statistic that actually discriminates. The chart stays on the mean
    because bars of mostly-zero integers say nothing; read it with the CSV open.

    `chart=False` writes the table only, for arm sets too wide to draw as grouped bars.
    """
    by_mode = _load_final_states(runs, results_dir)
    if not by_mode:
        print("[collect] no final_state.json found, skipping", filename)
        return

    families = [f for f in FAMILY_ORDER
                if any(f in st["by_terrain_family"]
                       for sts in by_mode.values() for st, _ in sts)]
    modes = sorted(by_mode)
    table = {}
    for mode in modes:
        for fam in families:
            vals = [st["by_terrain_family"][fam]["mean_terrain_level"]
                    for st, _ in by_mode[mode] if fam in st["by_terrain_family"]]
            table[(mode, fam)] = float(np.mean(vals)) if vals else np.nan

    top = {}
    for mode in modes:
        for fam in families:
            vals = [st["by_terrain_family"][fam]["max_terrain_level"]
                    for st, _ in by_mode[mode] if fam in st["by_terrain_family"]]
            top[(mode, fam)] = max(vals) if vals else ""

    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_seeds", "n_runs"]
                   + [f"{fam} (mean)" for fam in families]
                   + [f"{fam} (max)" for fam in families])
        for mode in modes:
            n_seeds = len(seeds_of([r for _, r in by_mode[mode]])) or len(by_mode[mode])
            w.writerow([mode, n_seeds, len(by_mode[mode])]
                       + [table[(mode, fam)] for fam in families]
                       + [top[(mode, fam)] for fam in families])
    print(f"[collect] wrote {csv_path}")
    if not chart:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    width = 0.8 / max(len(modes), 1)
    x = np.arange(len(families))
    for i, mode in enumerate(modes):
        vals = [table[(mode, fam)] for fam in families]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.88,
               color=color_for(mode, i),
               label=f"{LABEL.get(mode, mode)}  ({seed_label([r for _, r in by_mode[mode]])})",
               edgecolor="white", linewidth=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=8.5)
    _style(ax, "", "mean terrain level at end of training",
           "Curriculum reached, broken out by terrain type")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=2, loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"[collect] wrote {path}")


def _panel_labels(arms):
    """Arm names with the panel's shared leading words removed.

    "swing target off / z / z+apex" inside a panel already titled "Swing target"
    spends most of its width restating the title, and the three labels then collide
    with each other and with the neighbouring panel.
    """
    words = [a.split() for a in arms]
    shared = 0
    while all(len(w) > shared + 1 for w in words) and             len({tuple(w[:shared + 1]) for w in words}) == 1:
        shared += 1
    return [" ".join(w[shared:]) for w in words]


def plot_campaign_arms(runs, results_dir, out_dir, filename="fig_campaign_arms.png",
                       smooth=25):
    """The intervention campaign as small multiples -- one panel per intervention family.

    Eleven arms on one axes is not a chart, and it is not the comparison either: each
    family is its own controlled experiment and is only ever read against the same
    three-seed blind band. So the band is drawn once per panel in neutral grey (it is
    the reference, not a category) and each panel carries at most three arms in the
    fixed hue order. Every line is labelled at its right end, which is what licenses
    the two hues that sit below 3:1 on white.

    The y axis is symmetric-log below 0.05 so both halves of the story are legible: the
    fall from level ~2.5 in the first ten simulated seconds, and the floor the arms are
    actually judged on. Curves are smoothed -- the raw signal down there is a handful of
    robots changing rows.
    """
    by_cond = _terrain_curves(runs, results_dir)
    if not by_cond:
        print("[collect] no campaign runs with a terrain_level column, skipping", filename)
        return

    base = by_cond.get(CAMPAIGN_BASELINE)
    panels = [(t, sub, [a for a in arms if a in by_cond]) for t, sub, arms in CAMPAIGN_PANELS]
    panels = [pnl for pnl in panels if pnl[2]]
    if not panels:
        print("[collect] no campaign arms matched CAMPAIGN_PANELS, skipping", filename)
        return
    drawn = {a for _, _, arms in panels for a in arms} | ({CAMPAIGN_BASELINE} if base else set())
    missing = sorted(set(by_cond) - drawn)
    if missing:
        # Never silently. A new arm gets a new label from condition_of() and would
        # otherwise vanish from the figure with no trace.
        print(f"[collect] {filename}: {len(missing)} arm(s) not in CAMPAIGN_PANELS "
              f"and not drawn: {', '.join(missing)}")

    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.3 * nrow), dpi=160,
                             squeeze=False)
    flat = [ax for row in axes for ax in row]

    if base is not None:
        bx, bstack = _curve_stack(base)
        blo = _rolling(np.nanmin(bstack, axis=0), smooth)
        bhi = _rolling(np.nanmax(bstack, axis=0), smooth)
        bmean = _rolling(np.nanmean(bstack, axis=0), smooth)

    for ax, (title, subtitle, arms) in zip(flat, panels):
        if base is not None:
            ax.fill_between(bx, blo, bhi, color=BASELINE_INK, alpha=0.20, linewidth=0,
                            zorder=2)
            ax.plot(bx, bmean, color=BASELINE_INK, linewidth=1.6, zorder=3)
        labels = []
        short = _panel_labels(arms)
        for i, arm in enumerate(arms):
            entries = by_cond[arm]
            x, stack = _curve_stack(entries)
            y = _rolling(np.nanmean(stack, axis=0), smooth)
            c = PALETTE[i % 3]      # slots 0-2 clear the all-pairs gates; see CAMPAIGN_PANELS
            ax.plot(x, y, color=c, linewidth=2, zorder=4)
            last = np.flatnonzero(np.isfinite(y))
            if last.size:
                # The Phase 2 arms stop at half the baseline's simulated time, so the
                # label sits mid-panel; the dot is what ties it to its own line.
                ax.plot([x[last[-1]]], [y[last[-1]]], "o", color=c, markersize=5,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=5)
                # Seed count only when there is more than one -- the caption
                # carries the default, and the labels have no width to spare.
                n_seeds = len(seeds_of([e[2] for e in entries]))
                text = short[i] + (f"  ({n_seeds} seeds)" if n_seeds > 1 else "")
                labels.append((x[last[-1]], float(y[last[-1]]), text, c))
        _floor_yaxis(ax, 0.05, [0, 0.05, 0.2, 1.0, 3.0])
        _style(ax, "simulated time per robot (s)", "mean terrain level", "")
        # _style() sets its own title, so the panel heading goes on afterwards -- and
        # the subtitle is a separate text, or set_title() would overwrite the heading.
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=20)
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=INK_MUTED,
                fontsize=8, ha="left", va="bottom")
        ax.text(0.99, 0.96, "grey band: blind, 3 seeds", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=INK_MUTED)
        _end_labels(ax, labels, headroom=0.38)

    for ax in flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("Every intervention, read against the same blind seed band",
                 color=INK, fontsize=12, x=0.008, ha="left")
    caption = (f"Mean terrain level, {smooth}-iteration rolling mean, symlog below 0.05. "
               "Arms are one training seed unless the label says otherwise. All five "
               "terrain families are pooled here; 'stairs up' is 0.000 mean / 0 max in "
               "every run on this figure, so read the per-family table alongside "
               "(campaign_terrain_by_type.csv).")
    lines = textwrap.wrap(caption, width=150)
    fig.text(0.008, 0.008, "\n".join(lines), color=INK_MUTED, fontsize=7.5,
             ha="left", va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.02 + 0.022 * len(lines), 1, 0.965))
    path = os.path.join(out_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"[collect] wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results",
                        help="Root to scan for run folders (default: results)")
    parser.add_argument("--out", default=None,
                        help="Where to write the table and figures (default: --results-dir)")
    parser.add_argument("--exp1-only", action="store_true",
                        help="Skip the Experiment 2 figures")
    parser.add_argument("--include-diagnostics", action="store_true",
                        help="Also list smoke / control runs in runs.csv. They are "
                             "diagnostics, not results, so they are excluded by default.")
    parser.add_argument("--inherited-s-per-iter", type=float, default=None,
                        help="Seconds per iteration measured for the inherited "
                             "(pre-rewrite) build; drawn as a reference point on the "
                             "Experiment 1 figures. It has no run folder of its own.")
    parser.add_argument("--inherited-num-envs", type=int, default=16,
                        help="Batch size that measurement was taken at (default: 16)")
    parser.add_argument("--inherited-steps-per-iter", type=int, default=24,
                        help="Physics steps per iteration for that build (default: 24)")
    args = parser.parse_args()

    out_dir = args.out or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no run folders with summary.json found under {args.results_dir!r}")

    counts = collections.Counter(experiment_of(r) for r in runs)
    print("[collect] runs by experiment: "
          + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)))

    exp1, exp2 = select(runs, "exp1"), select(runs, "exp2")
    campaign = select(runs, "campaign")

    # runs.csv is a results table: measurements only. Smoke probes and control runs
    # are diagnostics and stay out unless asked for.
    table_runs = runs if args.include_diagnostics else exp1 + exp2 + campaign
    skipped = len(runs) - len(table_runs)
    if skipped:
        print(f"[collect] runs.csv: {skipped} diagnostic run(s) excluded "
              f"(--include-diagnostics to list them)")
    write_table(sorted(table_runs, key=lambda r: r["dir"]), out_dir)

    ref_itps = ref_steps = None
    if args.inherited_s_per_iter:
        ips = 1.0 / args.inherited_s_per_iter
        ref_itps = (args.inherited_num_envs, ips)
        ref_steps = (args.inherited_num_envs,
                     ips * args.inherited_num_envs * args.inherited_steps_per_iter)

    plot_scaling(exp1, out_dir, lambda r: r["summary"].get("it_per_s_median"),
                 "iterations / s", "Training throughput vs. batch size",
                 # 0.73 -> 2.58 it/s is well under one decade, so a log axis labels
                 # itself "10^0 / 2x10^0" -- powers of ten for numbers that are just
                 # 1 and 2 -- and visually flattens a 3.5x spread. Linear, from zero.
                 "fig_speed_itps.png", logy=False, inherited=ref_itps,
                 caption="Iterations/s at fixed work per iteration (24 differentiated "
                         "physics steps). Falling it/s is the per-iteration cost of a "
                         "bigger batch, not a scaling loss -- see robot-steps/s.")
    plot_scaling(exp1, out_dir, robot_steps_per_s,
                 "robot-steps / s", "Simulation throughput vs. batch size",
                 "fig_speed_throughput.png", logy=True, inherited=ref_steps,
                 caption="robot-steps/s = median it/s x num_envs x steps_per_iter. "
                         "Per-iteration cost grows sublinearly in the batch, so "
                         "throughput rises even as iterations/s falls.")
    plot_scaling(exp1, out_dir, lambda r: r["summary"].get("peak_mem_mb"),
                 "peak PyTorch-allocated GPU memory (MB)",
                 "Peak PyTorch-allocated GPU memory vs. batch size",
                 "fig_speed_mem.png", logy=False,
                 caption="torch.cuda.max_memory_allocated(): the PyTorch allocator only. "
                         "Excludes Isaac Gym / PhysX buffers, the terrain trimesh and the "
                         "CUDA context, so this is a variant-to-variant comparison, not "
                         "total GPU usage.")
    if not args.exp1_only:
        plot_terrain_level(exp2, args.results_dir, out_dir)
        plot_curriculum_final(exp2, args.results_dir, out_dir)
        plot_terrain_by_type(exp2, args.results_dir, out_dir)
        plot_campaign_arms(campaign, args.results_dir, out_dir)
        plot_curriculum_final(
            campaign, args.results_dir, out_dir,
            filename="fig_campaign_curriculum_final.png",
            title="Intervention campaign: robots per difficulty row at the end of training")
        plot_terrain_by_type(campaign, args.results_dir, out_dir,
                             csv_name="campaign_terrain_by_type.csv", chart=False)


if __name__ == "__main__":
    main()
