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
    terrain_by_type.csv     the numbers behind that figure

Everything is derived from the run folders, so runs made days apart, or in a
different checkout, land on the same axes as long as they were logged by
``bench_log.py``.

Runs are sorted into experiments by their folder (``bench/`` -> Experiment 1,
``train/`` -> Experiment 2, ``smoke/`` -> sizing probe, ``control/`` -> flat-ground
learning-parity check) and each figure draws only the experiment it belongs to.

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


def experiment_of(run):
    """'exp1' (speed), 'exp2' (locomotion), 'smoke', 'control' or 'other'.

    Mixing these is what broke the old figures: the 30-iteration smoke runs landed
    in the Experiment 2 groups, which inflated every n= and -- far worse -- pulled
    ``min(len(curve))`` down to 30, truncating the 5000-iteration curves.

    'control' is the flat-ground learning-parity run against the inherited build.
    It is blind + flat like a bench run, so without its own bucket it would land in
    Experiment 1 and put a second point on top of bench/cuda_B16. It belongs in
    runs.csv, but in no figure.
    """
    prefix = run["dir"].split("/")[0]
    if prefix in EXPERIMENT_BY_PREFIX:
        return EXPERIMENT_BY_PREFIX[prefix]
    meta = run["meta"]
    if meta.get("mode") == "blind":               # flat ground, no curriculum
        return "exp1"
    iters = meta.get("iters")
    if isinstance(iters, int) and iters <= SMOKE_MAX_ITERS:
        return "smoke"
    return "exp2" if meta.get("terrain_type") == "rudin" else "other"


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
    fields = ["dir", "experiment", "mode", "variant", "num_envs", "seed", "iters_timed",
              "it_per_s_median", "it_per_s_p10", "it_per_s_p90", "robot_steps_per_s",
              "peak_mem_mb", "total_wall_s", "final_terrain_level", "final_loss",
              "final_vx", "gpu"]
    path = os.path.join(out_dir, "runs.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            row = {"dir": r["dir"], "experiment": experiment_of(r),
                   "variant": variant_of(r), "gpu": r["meta"].get("gpu"),
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
    y0, y1 = ax.get_ylim()
    min_gap = (y1 - y0) * 0.05
    entries = sorted(entries, key=lambda e: e[1])
    ys = [e[1] for e in entries]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for (x, _, text, color), y in zip(entries, ys):
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


def plot_terrain_level(runs, results_dir, out_dir, filename="fig_terrain_level.png"):
    """Experiment 2: mean terrain level vs simulated robot-seconds, mean + min/max band."""
    by_mode = {}
    for r in runs:
        mode = r["meta"].get("mode")
        if mode in (None, "blind"):        # flat runs have no curriculum
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
        by_mode.setdefault(mode, []).append((arr, dt_sim, r))
    if not by_mode:
        print("[collect] no Experiment 2 runs with a terrain_level column, skipping figure")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=160)
    labels = []
    for i, (mode, entries) in enumerate(sorted(by_mode.items())):
        # Pad to the LONGEST run and average with nan-aware ops, rather than
        # truncating to the shortest. Truncating let one short run chop the curve
        # for every seed -- that is what pinned this figure to 30 iterations while
        # smoke runs were still being grouped in here.
        n = max(a.size for a, _, _ in entries)
        stack = np.full((len(entries), n), np.nan)
        for row_i, (a, _, _) in enumerate(entries):
            stack[row_i, :a.size] = a
        x = np.arange(n) * entries[0][1]
        mean = np.nanmean(stack, axis=0)
        c = color_for(mode, i)
        label = LABEL.get(mode, mode)
        if stack.shape[0] > 1:              # band only means something across seeds
            ax.fill_between(x, np.nanmin(stack, axis=0), np.nanmax(stack, axis=0),
                            color=c, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, mean, color=c, linewidth=2, zorder=3,
                label=f"{label}  ({seed_label([e[2] for e in entries])})")
        labels.append((x[-1], float(mean[-1]), label, c))
    _style(ax, "simulated time per robot (s)", "mean terrain level",
           "Curriculum progress by perception condition")
    _end_labels(ax, labels)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"[collect] wrote {path}")


def _load_final_states(runs, results_dir):
    """{mode: [(final_state.json, run), ...]} for every run that reached the end.

    The run travels with its state so the figures can label by distinct seed
    rather than by run count.
    """
    by_mode = {}
    for r in runs:
        mode = r["meta"].get("mode")
        if mode in (None, "blind"):
            continue
        path = os.path.join(results_dir, r["dir"], "final_state.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            by_mode.setdefault(mode, []).append((json.load(f), r))
    return by_mode


def plot_curriculum_final(runs, results_dir, out_dir, filename="fig_curriculum_final.png"):
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

    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([f"{LABEL.get(m, m)}  ({seed_label([r for _, r in by_mode[m]])})"
                        for m in modes], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.invert_yaxis()
    _style(ax, "share of robots", "",
           "Where training ended up: robots per difficulty row (0 easiest, "
           f"{num_rows - 1} hardest)")
    ax.grid(True, axis="y", color="white", linewidth=0)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"[collect] wrote {path}")


def plot_terrain_by_type(runs, results_dir, out_dir, filename="fig_terrain_by_type.png"):
    """Mean terrain level per terrain family -- stairs is where perception should show."""
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

    csv_path = os.path.join(out_dir, "terrain_by_type.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "n_seeds"] + families)
        for mode in modes:
            n_seeds = len(seeds_of([r for _, r in by_mode[mode]])) or len(by_mode[mode])
            w.writerow([mode, n_seeds] + [table[(mode, fam)] for fam in families])
    print(f"[collect] wrote {csv_path}")

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

    # runs.csv is a results table: measurements only. Smoke probes and control runs
    # are diagnostics and stay out unless asked for.
    table_runs = runs if args.include_diagnostics else exp1 + exp2
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
                 "fig_speed_itps.png", logy=True, inherited=ref_itps)
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


if __name__ == "__main__":
    main()
