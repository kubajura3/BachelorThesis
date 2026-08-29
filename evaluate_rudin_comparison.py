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

# NOTE: Isaac Gym must be imported before torch (hard requirement of isaacgym).
try:
    from isaacgym import gymapi  # noqa: F401
except Exception:
    pass

import torch

from config import EnvCfg
from env import RealQuadEnv
from policy import Policy, VisionPolicy
from utils_math import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_policy(policy: torch.nn.Module, weight_path: str, device: torch.device) -> torch.nn.Module:
    """Load weights into an already-built policy (Policy or VisionPolicy).

    The caller constructs the right class/obs-dim for the chosen --obs-mode; this
    just loads the state dict (mirrors play_many_dog.load_policy otherwise).
    """
    policy = policy.to(device)
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        policy.load_state_dict(state)
        print(f"[eval] Loaded policy weights from {weight_path}.")
    else:
        print(f"[eval] WARNING: weight file {weight_path} not found, using randomly initialized policy.")
    policy.eval()
    return policy


def corrupt_channel(x, mode, perm):
    """Corrupt a terrain-perception channel for the THESIS_PLAN sec 6.8 saliency diagnostic.

    Invariance of the policy's behaviour to this corruption is direct evidence that it learned to
    disregard the terrain signal, which distinguishes "the information does not help" from "the
    information is present and structurally unusable" (THESIS_PLAN sec E.3).

    Args:
        x: (B, ...) terrain channel -- the height-scan slice of the obs, or the depth image.
        mode: "none" (identity), "zero", or "shuffle".
        perm: (B,) fixed batch permutation used by "shuffle". Drawn once per run so robot i
            persistently sees robot perm(i)'s terrain: a coherent-but-wrong terrain stream rather
            than per-step noise, which would conflate "uses the signal" with "is destabilised by
            jitter".

    Returns:
        The corrupted channel, same shape and dtype as ``x``.
    """
    if mode == "none":
        return x
    if mode == "zero":
        # For the height scan this is semantically clean: get_obs() emits clip(hm - h0, -1, 1),
        # so zero means "flat ground at nominal height" (env.py get_obs). For a depth image it
        # means "surface at zero range", which is NOT neutral -- for --obs-mode depth, shuffle is
        # the primary condition and zero only a sanity check. See the plan's depth caveat.
        return torch.zeros_like(x)
    if mode == "shuffle":
        # Substitutes each robot's terrain with another environment's, per sec 6.8. Preserves the
        # marginal distribution exactly, so it controls for "the policy just needs input of the
        # right magnitude" in a way that zeroing does not.
        return x[perm]
    raise ValueError(f"unknown corruption mode {mode!r}")


def parse_args():
    """Command-line arguments for the deterministic Rudin-terrain evaluation."""
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
    p.add_argument("--obs-mode", type=str, default="blind", choices=["blind", "height", "depth"],
                   help="Policy input: blind 36-D obs, 36+187 privileged height scan, or "
                        "36-D obs + depth image through the VisionPolicy CNN. Must match "
                        "how the weights were trained.")
    p.add_argument("--corrupt", type=str, default="none", choices=["none", "zero", "shuffle"],
                   help="Terrain-input corruption for the sec 6.8 saliency diagnostic: none "
                        "(clean baseline), zero, or shuffle (robot i sees robot perm(i)'s "
                        "terrain). Requires an obs-mode that has a terrain channel.")
    p.add_argument("--corrupt-seed", type=int, default=0,
                   help="Seeds the shuffle permutation independently of --seed, so the clean and "
                        "corrupted arms keep identical env/command streams.")
    p.add_argument("--selftest", action="store_true",
                   help="Wiring check: on step 0 compute the action with clean and with corrupted "
                        "input, print the resulting action delta, and exit without a full "
                        "rollout. Non-zero delta = the corruption reaches the policy.")
    p.add_argument("--tag", type=str, default="diffsim",
                   help="Label for the output files (eval_results_<tag>.json/.csv).")
    p.add_argument("--out", type=str, default=RESULTS_DIR,
                   help="Directory for the result files.")
    return p.parse_args()


@torch.no_grad()
def main():
    """Run the evaluation rollout and write eval_results_<tag>.json/.csv."""
    args = parse_args()
    if args.warmup >= args.steps:
        raise SystemExit(f"--warmup ({args.warmup}) must be < --steps ({args.steps}).")
    if args.corrupt != "none" and args.obs_mode == "blind":
        raise SystemExit(
            f"--corrupt {args.corrupt} needs a terrain channel, but --obs-mode blind has none. "
            "Use --obs-mode height or depth. (The blind policy is the control arm: run it with "
            "--corrupt none to get the terrain-blind floor.)")

    device = torch.device(args.device)
    set_seed(args.seed)

    # -------- Build the Rudin-terrain comparison config --------
    cfg = EnvCfg()
    cfg.terrain_type = "rudin"
    cfg.rudin_terrain.dynamic_curriculum = True   # promote/demote across episodes
    cfg.num_envs = args.num_envs
    cfg.use_viewer = False
    cfg.use_gpu_pipeline = True
    if args.obs_mode != "blind":
        cfg.use_perception = True
        if args.obs_mode == "height":
            cfg.use_height_obs = True
        else:
            cfg.use_depth_obs = True

    env = RealQuadEnv(cfg, device=device)
    env.reset()
    if args.obs_mode == "depth":
        policy = VisionPolicy(dim_obs=env.obs_dim, dim_action=12)
    else:
        policy = Policy(dim_obs=env.obs_dim, dim_action=12)
    policy = load_policy(policy, args.weights, device)

    B = env.B
    num_rows = int(cfg.rudin_terrain.num_rows)

    # Width of the height-scan tail of the obs vector (0 when the policy has no height input).
    # Read from the sampler rather than hardcoded, because env.obs_dim is built from this property.
    n_h = env.perception.height_sampler.num_points if getattr(cfg, "use_height_obs", False) else 0
    # Shuffle permutation, drawn from its OWN generator so that switching --corrupt does not touch
    # the global RNG stream. The clean and corrupted arms therefore see identical terrain, command
    # and spawn draws, and any difference in the metrics is attributable to the corruption alone.
    _g = torch.Generator().manual_seed(args.corrupt_seed)
    perm = torch.randperm(B, generator=_g).to(device)

    def _apply(obs, depth):
        """Corrupt whichever terrain channel this obs-mode carries. Returns (obs, depth, clean, dirty)."""
        if args.obs_mode == "depth":
            dirty = corrupt_channel(depth, args.corrupt, perm)
            return obs, dirty, depth, dirty
        if n_h:
            clean = obs[:, -n_h:]
            dirty = corrupt_channel(clean, args.corrupt, perm)
            return torch.cat([obs[:, :-n_h], dirty], dim=-1), depth, clean, dirty
        return obs, depth, None, None

    if args.selftest:
        # Wiring check: does the corruption actually reach the policy? Compares the action the
        # policy emits on step 0 with clean versus corrupted input. A delta of exactly 0 under
        # --corrupt zero/shuffle means the channel slice is wrong and the corruption is landing on
        # nothing, which would silently produce a fake null result in the sec 6.8 diagnostic.
        # All three modes are checked in this one process: building the Rudin trimesh is the
        # dominant cost of a short run, so looping here instead of re-invoking the script per
        # mode turns three terrain builds into one.
        s0 = env.get_obs().to(device)
        d0 = env.collect_perception()["depth_clean"] if args.obs_mode == "depth" else None
        if args.obs_mode == "depth":
            a_clean, _ = policy(s0, d0, None)
        else:
            a_clean, _ = policy(s0, None)
        print("")
        print(f"===== selftest: obs_mode={args.obs_mode} =====")
        ok = True
        for mode in ("none", "zero", "shuffle"):
            saved, args.corrupt = args.corrupt, mode
            s_c, d_c, clean, dirty = _apply(s0, d0)
            args.corrupt = saved
            if args.obs_mode == "depth":
                a_dirty, _ = policy(s_c, d_c, None)
            else:
                a_dirty, _ = policy(s_c, None)
            delta = (a_dirty - a_clean).abs().mean().item()
            chan = 0.0 if clean is None else clean.abs().mean().item()
            pert = 0.0 if clean is None else (dirty - clean).abs().mean().item()
            good = (delta == 0.0) if mode == "none" else (delta > 0.0)
            ok = ok and good
            print(f"  {mode:<8} channel_l1={chan:<12.6g} corrupt_l1={pert:<12.6g} "
                  f"|da|={delta:<12.6g} {'OK' if good else 'FAIL'}")
        print("  expected: none -> |da| exactly 0; zero/shuffle -> |da| > 0 "
              "(0 means the corruption never reaches the policy)")
        if not ok:
            raise SystemExit("selftest FAILED -- do not run the matrix until this passes")
        return

    # -------- Accumulators --------
    n_falls = 0
    n_timeouts = 0
    dist_sum = 0.0
    dist_count = 0
    lin_err_sum = 0.0
    ang_err_sum = 0.0
    level_sum = 0.0
    metric_steps = 0
    # Corruption magnitude over the metric window. Without these the sec 6.8 null is
    # uninterpretable: a policy that ignores the terrain and a corruption that changed nothing
    # (because the robots sat on level ~0, where there is almost no geometry -- CAMPAIGN_FINDINGS
    # sec 19.8) produce identical metrics. Reported so the two can be told apart.
    corrupt_l1_sum = 0.0
    channel_l1_sum = 0.0
    chan_steps = 0
    level_hist_start = None

    hx = None
    a_prev = torch.zeros(B, 12, device=device)
    hx_hold = None

    print(f"[eval] Evaluating on Rudin terrain: B={B}, seed={args.seed}, "
          f"steps={args.steps}, warmup={args.warmup}, episode_length_s={cfg.episode_length_s}")

    for t in range(args.steps):
        s = env.get_obs().to(device)
        if (t % cfg.action_hold) == 0:
            depth = env.collect_perception()["depth_clean"] if args.obs_mode == "depth" else None
            s, depth, clean, dirty = _apply(s, depth)
            if clean is not None and t >= args.warmup:
                channel_l1_sum += clean.abs().mean().item()
                corrupt_l1_sum += (dirty - clean).abs().mean().item()
                chan_steps += 1
            if args.obs_mode == "depth":
                a, hx = policy(s, depth, hx)
            else:
                a, hx = policy(s, hx)
            a_prev = a
            hx_hold = hx
        else:
            a = a_prev
            hx = hx_hold

        _, extra, _, _ = env.step(a)

        # ---- metrics for this step (state reflects the command followed this step; pre-reset) ----
        if t >= args.warmup:
            if level_hist_start is None:
                # Difficulty spread entering the metric window. The eval starts from
                # randint(0, max_init_terrain_level+1) and decays as robots fall, so this says
                # how much terrain geometry the window actually contained -- the context a null
                # corruption result has to be read against.
                level_hist_start = torch.bincount(
                    env.terrain_levels.clamp(0, num_rows - 1), minlength=num_rows).cpu().tolist()
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
            "obs_mode": args.obs_mode,
            "corrupt": args.corrupt,
            "corrupt_seed": args.corrupt_seed,
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
            "corrupt_l1": corrupt_l1_sum / max(1, chan_steps),
            "channel_l1": channel_l1_sum / max(1, chan_steps),
            "terrain_level_hist": level_hist,
            "terrain_level_hist_start": level_hist_start if level_hist_start is not None else [],
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
    hist_keys = ("terrain_level_hist", "terrain_level_hist_start")
    flat = {**{f"cfg_{k}": v for k, v in results["config"].items()},
            **{k: v for k, v in results["metrics"].items() if k not in hist_keys}}
    for k in hist_keys:
        flat[k] = ";".join(str(x) for x in results["metrics"][k])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(flat.keys()))
        w.writerow(list(flat.values()))

    print(f"\n[eval] Wrote {json_path}\n[eval] Wrote {csv_path}")


if __name__ == "__main__":
    main()
