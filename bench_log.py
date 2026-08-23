"""Per-run measurement logger for the thesis experiments.

Writes one self-contained folder per run so that runs can be compared later
without re-running anything:

    <run_dir>/iters.csv      one row per training iteration
    <run_dir>/meta.json      what was run (config, git commit, GPU, versions)
    <run_dir>/summary.json   aggregated timing / memory, warm-up excluded

Deliberately dependency-free apart from torch and the standard library, and it
touches no other module in this repository. That is on purpose: the same file
drops unchanged into the inherited commit (``1731453``) so the speed comparison
uses one measurement path for every code variant.

Usage (two inserted lines inside an existing training loop)::

    from bench_log import BenchLog

    bench = BenchLog(os.getenv("RUN_DIR"), meta={"variant": "cuda", "num_envs": B})
    for it in range(num_iters):
        bench.start()
        ...                                     # existing loop body, unchanged
        bench.stop(loss=float(loss), vx=vx_for_plot)
    bench.close()

``start``/``stop`` call ``torch.cuda.synchronize()`` so the recorded time is the
GPU work actually completed, not just the kernel-launch time -- CUDA is
asynchronous, so timing without the sync measures almost nothing.
"""

import csv
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime

import torch

DEFAULT_WARMUP = 20


def _git_commit():
    """Short git hash of the working tree, or None outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _gpu_name():
    """Name of the active CUDA device, or None on CPU-only machines."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        pass
    return None


def _percentile(sorted_vals, q):
    """Linear-interpolated percentile of an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class BenchLog:
    """Collects per-iteration timing plus arbitrary scalars, then writes a run folder.

    Args:
        run_dir: Output directory. ``None`` disables everything -- every method
            becomes a no-op, so the logger can stay wired into the training
            script permanently without affecting normal runs.
        meta: Anything worth recording about the run (mode, num_envs, seed,
            backend...). Merged with auto-detected environment information.
        warmup: Iterations excluded from the summary statistics. The first ones
            include CUDA context creation, cuDNN/cuBLAS autotuning, terrain
            construction and allocator growth, so they are not representative.
    """

    def __init__(self, run_dir=None, meta=None, warmup=DEFAULT_WARMUP):
        self.run_dir = run_dir
        self.enabled = bool(run_dir)
        self.warmup = int(warmup)
        self.rows = []
        self._t0 = None
        self._wall_start = time.perf_counter()
        self.meta = dict(meta or {})
        if not self.enabled:
            return
        os.makedirs(run_dir, exist_ok=True)
        self.meta.update({
            "warmup": self.warmup,
            "git_commit": _git_commit(),
            "gpu": _gpu_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "host": socket.gethostname(),
            "started": datetime.now().isoformat(timespec="seconds"),
        })
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"[bench] logging to {run_dir}")

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self):
        """Mark the beginning of one iteration."""
        if not self.enabled:
            return
        self._sync()
        self._t0 = time.perf_counter()

    def stop(self, **scalars):
        """Mark the end of one iteration and record it.

        Any keyword argument is stored as a column in ``iters.csv`` -- pass
        whatever the run should be judged on (loss, vx, terrain_level, ...).
        Values that are ``None`` or non-finite are written as empty cells.
        """
        if not self.enabled or self._t0 is None:
            return
        self._sync()
        dt = time.perf_counter() - self._t0
        self._t0 = None
        row = {"iter": len(self.rows), "t_iter_s": dt}
        if torch.cuda.is_available():
            row["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
        for k, v in scalars.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = None
        self.rows.append(row)

    def close(self):
        """Write ``iters.csv``, ``meta.json`` and ``summary.json``."""
        if not self.enabled or not self.rows:
            return

        fields = []
        for row in self.rows:
            for k in row:
                if k not in fields:
                    fields.append(k)

        with open(os.path.join(self.run_dir, "iters.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in self.rows:
                w.writerow({k: row.get(k, "") for k in fields})

        timed = [r["t_iter_s"] for r in self.rows[self.warmup:]]
        if not timed:                      # run shorter than the warm-up window
            timed = [r["t_iter_s"] for r in self.rows]
        ts = sorted(timed)
        t_med = _percentile(ts, 0.5)
        t_p10 = _percentile(ts, 0.10)
        t_p90 = _percentile(ts, 0.90)

        summary = {
            "iters_total": len(self.rows),
            "iters_timed": len(timed),
            "warmup": self.warmup,
            "t_iter_median_s": t_med,
            # Fast iterations give high it/s, so the percentiles cross over.
            "it_per_s_median": (1.0 / t_med) if t_med else None,
            "it_per_s_p10": (1.0 / t_p90) if t_p90 else None,
            "it_per_s_p90": (1.0 / t_p10) if t_p10 else None,
            "peak_mem_mb": max((r.get("peak_mem_mb") or 0.0) for r in self.rows) or None,
            "total_wall_s": time.perf_counter() - self._wall_start,
        }
        # Last recorded value of every extra scalar, so summary.json alone is
        # enough for a results table.
        for k in fields:
            if k in ("iter", "t_iter_s", "peak_mem_mb"):
                continue
            vals = [r.get(k) for r in self.rows if r.get(k) is not None]
            if vals:
                summary[f"final_{k}"] = vals[-1]
        summary.update({k: self.meta.get(k) for k in ("variant", "mode", "num_envs", "seed")
                        if self.meta.get(k) is not None})

        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self.meta["finished"] = datetime.now().isoformat(timespec="seconds")
        with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)

        ips = summary["it_per_s_median"]
        mem = summary["peak_mem_mb"]
        print(f"[bench] {self.run_dir}: {ips:.3f} it/s median over {len(timed)} timed iters"
              + (f", peak {mem:.0f} MB" if mem else ""))
