"""Merge the per-run folders written by bench_log.py into thesis tables and figures.

Reads every ``*/summary.json`` (+ ``meta.json``, + ``iters.csv``) under a results
root and produces:

    runs.csv                one row per run -- the table for the thesis
    fig_speed_itps.png      Experiment 1: iterations/s vs. num_envs, one line per variant
    fig_speed_mem.png       Experiment 1: peak GPU memory vs. num_envs
    fig_terrain_level.png   Experiment 2: mean terrain level vs. simulated time,
                            one line per condition, mean over seeds with a min/max band
    fig_curriculum_final.png    where each condition ENDED UP: the distribution of
                            robots over the 10 difficulty rows at the end of training
    fig_terrain_by_type.png     mean terrain level per terrain family per condition
    terrain_by_type.csv     the numbers behind that figure

Everything is derived from the run folders, so runs made days apart, or in a
different checkout, land on the same axes as long as they were logged by
``bench_log.py``.

Usage:
    python collect_bench.py                                   # scans results/
    python collect_bench.py --results-dir results --out results/figures
    python collect_bench.py --exp1-only                       # skip the training figure

CPU-only: needs numpy + matplotlib, not torch or Isaac Gym.
"""

import argparse
import csv
import json
import os

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
    fields = ["dir", "mode", "variant", "num_envs", "seed", "iters_timed",
              "it_per_s_median", "it_per_s_p10", "it_per_s_p90", "peak_mem_mb",
              "total_wall_s", "final_terrain_level", "final_loss", "final_vx", "gpu"]
    path = os.path.join(out_dir, "runs.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            row = {"dir": r["dir"], "variant": variant_of(r), "gpu": r["meta"].get("gpu")}
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


def plot_scaling(runs, out_dir, key, ylabel, title, filename, logy):
    """Experiment 1: one line per variant, x = num_envs (log)."""
    series = {}
    for r in runs:
        B, y = r["meta"].get("num_envs"), r["summary"].get(key)
        if B is None or y is None or r["meta"].get("mode") != "blind":
            continue
        series.setdefault(variant_of(r), []).append((int(B), float(y)))
    if not series:
        print(f"[collect] no Experiment 1 runs found for {key}, skipping {filename}")
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
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(sorted({x for pts in series.values() for x, _ in pts}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _style(ax, "parallel robots (num_envs)", ylabel, title)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="best")
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
        by_mode.setdefault(mode, []).append((arr, dt_sim))
    if not by_mode:
        print("[collect] no Experiment 2 runs with a terrain_level column, skipping figure")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=160)
    labels = []
    for i, (mode, entries) in enumerate(sorted(by_mode.items())):
        n = min(a.size for a, _ in entries)
        stack = np.vstack([a[:n] for a, _ in entries])
        x = np.arange(n) * entries[0][1]
        mean = np.nanmean(stack, axis=0)
        c = color_for(mode, i)
        label = LABEL.get(mode, mode)
        if stack.shape[0] > 1:              # band only means something across seeds
            ax.fill_between(x, np.nanmin(stack, axis=0), np.nanmax(stack, axis=0),
                            color=c, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, mean, color=c, linewidth=2, zorder=3,
                label=f"{label}  (n={stack.shape[0]})")
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
    """{mode: [final_state.json, ...]} for every run that reached the end."""
    by_mode = {}
    for r in runs:
        mode = r["meta"].get("mode")
        if mode in (None, "blind"):
            continue
        path = os.path.join(results_dir, r["dir"], "final_state.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            by_mode.setdefault(mode, []).append(json.load(f))
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
        [st["mean_terrain_level"] for st in by_mode[m]]))]
    num_rows = by_mode[modes[0]][0]["num_rows"]
    ramp = [DIFFICULTY_RAMP[int(round(i * (len(DIFFICULTY_RAMP) - 1) / max(1, num_rows - 1)))]
            for i in range(num_rows)]

    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(modes) + 2.0), dpi=160)
    for row_i, mode in enumerate(modes):
        states = by_mode[mode]
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
    ax.set_yticklabels([f"{LABEL.get(m, m)}  (n={len(by_mode[m])})" for m in modes],
                       fontsize=9, color=INK)
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
                if any(f in st["by_terrain_family"] for sts in by_mode.values() for st in sts)]
    modes = sorted(by_mode)
    table = {}
    for mode in modes:
        for fam in families:
            vals = [st["by_terrain_family"][fam]["mean_terrain_level"]
                    for st in by_mode[mode] if fam in st["by_terrain_family"]]
            table[(mode, fam)] = float(np.mean(vals)) if vals else np.nan

    csv_path = os.path.join(out_dir, "terrain_by_type.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "n_seeds"] + families)
        for mode in modes:
            w.writerow([mode, len(by_mode[mode])] + [table[(mode, fam)] for fam in families])
    print(f"[collect] wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    width = 0.8 / max(len(modes), 1)
    x = np.arange(len(families))
    for i, mode in enumerate(modes):
        vals = [table[(mode, fam)] for fam in families]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.88,
               color=color_for(mode, i), label=f"{LABEL.get(mode, mode)}  (n={len(by_mode[mode])})",
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
                        help="Skip the Experiment 2 terrain-level figure")
    args = parser.parse_args()

    out_dir = args.out or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no run folders with summary.json found under {args.results_dir!r}")

    write_table(runs, out_dir)
    plot_scaling(runs, out_dir, "it_per_s_median", "iterations / s",
                 "Training throughput vs. batch size", "fig_speed_itps.png", logy=True)
    plot_scaling(runs, out_dir, "peak_mem_mb", "peak GPU memory (MB)",
                 "Peak GPU memory vs. batch size", "fig_speed_mem.png", logy=False)
    if not args.exp1_only:
        plot_terrain_level(runs, args.results_dir, out_dir)
        plot_curriculum_final(runs, args.results_dir, out_dir)
        plot_terrain_by_type(runs, args.results_dir, out_dir)


if __name__ == "__main__":
    main()
