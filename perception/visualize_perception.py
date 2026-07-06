"""Visual sanity-check for the perception module.

Runs the depth camera and the height sampler on a handful of robot poses and
dumps PNGs (depth clean/noisy + height-map heatmap + a combined panel per pose),
so you can *see* that the module works before wiring it into training.

Two modes:

* **synthetic** (default) -- builds a small terrain with a few raised blocks and
  a pit entirely in numpy, so it needs **no Isaac Gym** and runs on ``cuda`` or
  ``cpu``. Ideal for a quick check on the dev machine.
* **--from-env** -- pulls live poses from a running :class:`env.RealQuadEnv`
  (needs Isaac Gym + a GPU; guarded so it is skipped where unavailable).

Because the collector consumes a terrain object by duck-typing, the synthetic
mode passes a :class:`~perception.terrain_mesh.TerrainField` straight through
(``from_terrain_data`` just reads the same attributes it exposes).

Examples::

    python -m perception.visualize_perception
    python -m perception.visualize_perception --device cpu --poses 4 --out ./viz
    python -m perception.visualize_perception --from-env --steps 50
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from .config import PerceptionCfg
from .collector import PerceptionCollector
from .terrain_mesh import TerrainField


def make_synthetic_terrain(size_m: float = 10.0, hs: float = 0.05) -> TerrainField:
    """Create a test terrain: flat ground with raised blocks, a wall and a pit.

    Args:
        size_m: Side length of the square terrain in metres.
        hs: Horizontal cell size in metres.

    Returns:
        A :class:`TerrainField` centred on the origin (both a heightfield for the
        height sampler and a matching trimesh for the camera).
    """
    n = int(round(size_m / hs))
    hf = np.zeros((n, n), dtype=np.float32)
    tx = ty = -size_m / 2.0

    def cell(x_m, y_m):
        """Return the (row, col) heightfield indices for world (x, y) in metres."""
        return int(round((x_m - tx) / hs)), int(round((y_m - ty) / hs))

    # Raised platform ahead of the origin (x in [1.5, 2.5], y in [-0.6, 0.6]).
    i0, j0 = cell(1.5, -0.6)
    i1, j1 = cell(2.5, 0.6)
    hf[i0:i1, j0:j1] = 0.40

    # A wall further out (x in [3.2, 3.4], full width).
    i0, _ = cell(3.2, 0.0)
    i1, _ = cell(3.4, 0.0)
    hf[i0:i1, :] = 0.60

    # A pit behind the origin (x in [-2.0, -1.2], y in [-0.6, 0.6]).
    i0, j0 = cell(-2.0, -0.6)
    i1, j1 = cell(-1.2, 0.6)
    hf[i0:i1, j0:j1] = -0.30

    return TerrainField.from_heightfield(hf, horizontal_scale=hs, x_offset=tx, y_offset=ty)


def yaw_to_quat_xyzw(yaw: float, device: str) -> torch.Tensor:
    """Quaternion (xyzw) for a pure yaw rotation about world z."""
    return torch.tensor(
        [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=torch.float32, device=device
    )


def build_synthetic_poses(num_poses: int, device: str):
    """A few hand-picked base poses aimed at the synthetic terrain features.

    Args:
        num_poses: How many poses to return (clamped to the built-in list).
        device: Torch device.

    Returns:
        (base_pos (P, 3), base_quat_xyzw (P, 4)) with the robot ~0.35 m above the
        flat ground near the origin, facing various directions.
    """
    specs = [
        ((0.0, 0.0, 0.35), 0.0),      # facing the platform/wall
        ((0.0, 0.0, 0.35), 0.4),      # yawed left
        ((0.3, 0.3, 0.35), -0.3),     # offset + yawed right
        ((-0.5, 0.0, 0.35), np.pi),   # facing the pit
    ]
    specs = specs[: max(1, min(num_poses, len(specs)))]
    pos = torch.tensor([s[0] for s in specs], dtype=torch.float32, device=device)
    quat = torch.stack([yaw_to_quat_xyzw(s[1], device) for s in specs], dim=0)
    return pos, quat


def save_frames(out_dir: str, result: dict, grid_shape, cfg: PerceptionCfg) -> None:
    """Write per-pose depth/height PNGs and a combined panel.

    Args:
        out_dir: Output directory (created if missing).
        result: The dict returned by :meth:`PerceptionCollector.collect`.
        grid_shape: (nx, ny) shape of the height map for reshaping.
        cfg: Perception config (for axis extents/labels).
    """
    os.makedirs(out_dir, exist_ok=True)
    depth_clean = result["depth_clean"].detach().cpu().numpy()   # (P, S, h, w)
    depth_noisy = result["depth_noisy"].detach().cpu().numpy()
    height = result["height_map"].detach().cpu().numpy()          # (P, n_pts)
    nx, ny = grid_shape
    hm_extent = [cfg.hm_y_min, cfg.hm_y_max, cfg.hm_x_min, cfg.hm_x_max]  # y horiz, x vert

    P = depth_clean.shape[0]
    for p in range(P):
        dc = depth_clean[p, 0]
        dn = depth_noisy[p, 0]
        hm = height[p].reshape(nx, ny)

        plt.imsave(os.path.join(out_dir, f"depth_clean_{p}.png"), dc, cmap="viridis")
        plt.imsave(os.path.join(out_dir, f"depth_noisy_{p}.png"), dn, cmap="viridis")
        plt.imsave(os.path.join(out_dir, f"heightmap_{p}.png"), hm, cmap="terrain")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(dc, cmap="viridis"); axes[0].set_title(f"depth clean #{p}")
        axes[1].imshow(dn, cmap="viridis"); axes[1].set_title(f"depth noisy #{p}")
        im = axes[2].imshow(hm, cmap="terrain", origin="lower", extent=hm_extent, aspect="auto")
        axes[2].set_title(f"height map #{p}")
        axes[2].set_xlabel("y left [m]"); axes[2].set_ylabel("x forward [m]")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
        for ax in axes[:2]:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"panel_{p}.png"), dpi=110)
        plt.close(fig)

    print(f"[perception] wrote {P} pose(s) x (depth_clean, depth_noisy, heightmap, panel) -> {out_dir}")


def run_synthetic(args) -> None:
    """Build a synthetic scene, collect perception, and dump PNGs."""
    device = args.device
    field = make_synthetic_terrain()

    # Enable a bit of noise so the noisy panel is visibly different.
    cfg = PerceptionCfg(device=device, noise_gaussian=0.05, noise_dropout=0.05)

    base_pos, base_quat = build_synthetic_poses(args.poses, device)
    # The collector duck-types the terrain; a TerrainField exposes the same
    # attributes as the env's TerrainData, so pass it straight through.
    collector = PerceptionCollector(field, cfg, num_envs=base_pos.shape[0], device=device)
    result = collector.collect(base_pos, base_quat)
    save_frames(args.out, result, collector.height_sampler.grid_shape, cfg)


def run_from_env(args) -> None:
    """Pull live poses from a running Isaac Gym env and dump PNGs (GPU box only)."""
    try:
        from env import RealQuadEnv, ISAAC_AVAILABLE
    except Exception as e:  # pragma: no cover - depends on Isaac Gym
        print(f"[perception] --from-env unavailable (import failed: {e!r})")
        return
    if not ISAAC_AVAILABLE:
        print("[perception] --from-env requires Isaac Gym; skipping.")
        return

    from config import EnvCfg
    cfg = EnvCfg()
    cfg.terrain_type = "rudin"
    cfg.use_perception = True
    cfg.use_gpu_pipeline = True

    env = RealQuadEnv(cfg)
    env.reset()
    for _ in range(max(0, args.steps)):
        env.step(torch.zeros(env.B, 12, device=env.device))

    result = env.perception.collect(env.root_state[:, 0:3], env.root_state[:, 3:7])
    save_frames(args.out, result, env.perception.height_sampler.grid_shape, cfg.perception)


def main() -> None:
    """Parse args and run the requested visualisation mode."""
    parser = argparse.ArgumentParser(description="Visualise perception module outputs.")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device, choices=["cuda", "cpu"])
    parser.add_argument("--poses", type=int, default=4, help="synthetic poses to render")
    parser.add_argument("--steps", type=int, default=30, help="env steps before capture (--from-env)")
    parser.add_argument("--out", default="./perception_viz_out", help="output directory")
    parser.add_argument("--from-env", action="store_true", help="use a live Isaac Gym env")
    args = parser.parse_args()

    if args.from_env:
        run_from_env(args)
    else:
        run_synthetic(args)


if __name__ == "__main__":
    main()
