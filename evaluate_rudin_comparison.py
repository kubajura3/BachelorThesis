# -*- coding: utf-8 -*-
"""
evaluate_rudin_comparison.py — deterministic evaluation of a trained DiffSim/SRBD policy on Rudin's
curriculum terrain, producing the metrics needed for a fair comparison against Rudin's PPO baseline.

This is the DiffSim side of the comparison. To get Rudin's numbers, run `legged_gym`'s own play/eval
with the SAME terrain config (num_rows=10, num_cols=20, terrain_proportions=[0.1,0.1,0.35,0.25,0.2],
max_init_terrain_level=5), the SAME command ranges (lin_vel_x/y ∈ [-1,1], ang_vel_yaw ∈ [-1,1]), the
SAME episode_length_s=20, the SAME seed and the SAME num_envs. The metrics below are defined to line
up with what legged_gym logs (notably `extras["episode"]["terrain_level"]`).

Metrics reported (averaged over the post-warmup window):
  * mean_terrain_level   — mean of env.terrain_levels (directly comparable to Rudin's terrain_level).
  * lin_vel_track_err    — mean |v_body_xy − cmd_xy|   (m/s), the primary task-tracking metric.
  * ang_vel_track_err    — mean |omega_z_body − yaw_cmd| (rad/s).
  * fall_rate            — falls / (falls + timeouts): fraction of resets that were falls.
  * mean_reset_distance  — mean distance walked from the cell origin at reset (m).
  * terrain_level_hist   — final histogram of terrain_levels over the difficulty rows.

Usage:
    python evaluate_rudin_comparison.py --weights results/quad_diffsim_srbd_align_multi_robot.pth \
        --num_envs 1024 --seed 0 --steps 30000 --warmup 10000 --tag diffsim
"""

import os
import csv
import json
import argparse

# ⚠️ CRITICAL: Isaac Gym must be imported BEFORE torch
try:
    from isaacgym import gymapi  # noqa: F401
except Exception:
    pass

import torch

from config import EnvCfg, PURE_PAPER_MODE
from env import RealQuadEnv
from policy import Policy
from utils_math import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_policy(weight_path: str, device: torch.device,
                dim_obs: int = 36, dim_action: int = 12) -> Policy:
    """Build Policy and load weights (mirrors play_many_dog.load_policy)."""
    policy = Policy(dim_obs, dim_action).to(device)
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        policy.load_state_dict(state)
        print(f"✅ Loaded policy weights from {weight_path}.")
    else:
        print(f"⚠️ Weight file {weight_path} not found, using randomly initialized policy.")
    policy.eval()
    return policy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=str,
                   default=os.path.join("results", "quad_diffsim_srbd_align_multi_robot.pth"),
                   help="Policy weight file path.")
    p.add_argument("--num_envs", type=int, default=1024,
                   help="Number of parallel robots (use a large batch so all 20 columns are filled).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (match the legged_gym run for a like-for-like comparison).")
    p.add_argument("--steps", type=int, default=30000,
                   help="Total env.step() calls to run (1 step = dt = 0.002 s of sim time).")
    p.add_argument("--warmup", type=int, default=10000,
                   help="Steps to ignore before accumulating metrics (lets the curriculum settle).")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", type=str, default="diffsim",
                   help="Label for the output files (eval_results_<tag>.json/.csv).")
    p.add_argument("--out", type=str, default=RESULTS_DIR,
                   help="Directory for the result files.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.warmup >= args.steps:
        raise SystemExit(f"--warmup ({args.warmup}) must be < --steps ({args.steps}).")

    device = torch.device(args.device)
    set_seed(args.seed)

    # -------- Build the Rudin-terrain comparison config --------
    cfg = EnvCfg()
    cfg.terrain_type = "rudin"
    cfg.rudin_terrain.dynamic_curriculum = True   # promote/demote across episodes
    cfg.num_envs = args.num_envs
    cfg.use_viewer = False
    cfg.use_gpu_pipeline = True

    env = RealQuadEnv(cfg, device=device)
    env.reset()
    policy = load_policy(args.weights, device)

    B = env.B
    num_rows = int(cfg.rudin_terrain.num_rows)

    # -------- Accumulators --------
    n_falls = 0
    n_timeouts = 0
    dist_sum = 0.0
    dist_count = 0
    lin_err_sum = 0.0
    ang_err_sum = 0.0
    level_sum = 0.0
    metric_steps = 0

    hx = None
    a_prev = torch.zeros(B, 12, device=device)
    hx_hold = None

    print(f"▶ Evaluating on Rudin terrain: B={B}, seed={args.seed}, "
          f"steps={args.steps}, warmup={args.warmup}, episode_length_s={cfg.episode_length_s}")

    for t in range(args.steps):
        s = env.get_obs().to(device)
        if (t % cfg.action_hold) == 0:
            a, hx = policy(s, hx)
            a_prev = a
            hx_hold = hx
        else:
            a = a_prev
            hx = hx_hold

        _, extra, _, _ = env.step(a)

        # ---- metrics for this step (state reflects the command followed this step; pre-reset) ----
        if t >= args.warmup:
            lin_err = (env.base_lin_body[:, 0:2] - env.cmd_rand[:, 0:2]).abs().mean().item()
            ang_err = (env.base_ang_body[:, 2] - env.cmd_rand[:, 2]).abs().mean().item()
            lin_err_sum += lin_err
            ang_err_sum += ang_err
            level_sum += env.terrain_levels.float().mean().item()
            metric_steps += 1

        # ---- resets (fall or timeout): measure distance-from-origin before respawning ----
        done = extra["done"]
        timed_out = extra["timeout"]
        reset_mask = done | timed_out
        if reset_mask.any():
            reset_ids = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
            d = torch.norm(env.base_pos[reset_ids, 0:2] - env.env_origins[reset_ids, 0:2], dim=1)
            dist_sum += float(d.sum().item())
            dist_count += int(reset_ids.numel())
            n_falls += int(done.sum().item())
            n_timeouts += int(timed_out.sum().item())
            env.reset_envs(reset_ids)

    # -------- Aggregate --------
    total_resets = n_falls + n_timeouts
    level_hist = torch.bincount(env.terrain_levels.clamp(0, num_rows - 1),
                                minlength=num_rows).cpu().tolist()
    results = {
        "tag": args.tag,
        "config": {
            "num_envs": B,
            "seed": args.seed,
            "steps": args.steps,
            "warmup": args.warmup,
            "episode_length_s": cfg.episode_length_s,
            "num_rows": num_rows,
            "num_cols": int(cfg.rudin_terrain.num_cols),
            "max_init_terrain_level": int(cfg.rudin_terrain.max_init_terrain_level),
            "terrain_proportions": list(cfg.rudin_terrain.terrain_proportions),
            "cmd_lin_vel_x": list(cfg.rudin_cmd_lin_vel_x),
            "cmd_lin_vel_y": list(cfg.rudin_cmd_lin_vel_y),
            "cmd_ang_vel_yaw": list(cfg.rudin_cmd_ang_vel_yaw),
        },
        "metrics": {
            "mean_terrain_level": level_sum / max(1, metric_steps),
            "lin_vel_track_err": lin_err_sum / max(1, metric_steps),
            "ang_vel_track_err": ang_err_sum / max(1, metric_steps),
            "fall_rate": n_falls / max(1, total_resets),
            "mean_reset_distance": dist_sum / max(1, dist_count),
            "n_falls": n_falls,
            "n_timeouts": n_timeouts,
            "terrain_level_hist": level_hist,
        },
    }

    # -------- Report + persist --------
    print("\n===== Rudin-terrain evaluation =====")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, f"eval_results_{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(args.out, f"eval_results_{args.tag}.csv")
    flat = {**{f"cfg_{k}": v for k, v in results["config"].items()},
            **{k: v for k, v in results["metrics"].items() if k != "terrain_level_hist"}}
    flat["terrain_level_hist"] = ";".join(str(x) for x in level_hist)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(flat.keys()))
        w.writerow(list(flat.values()))

    print(f"\n💾 Wrote {json_path}\n💾 Wrote {csv_path}")


if __name__ == "__main__":
    main()
