# -*- coding: utf-8 -*-
"""play_many_dog.py - Visualise a trained policy on multiple robots in Isaac Gym.

Loads a trained policy and runs it live in the Isaac Gym viewer (no SRBD model,
no gradients -- pure playback with per-env resets on falls).

--terrain and --obs-mode mirror the training configuration, so a policy trained
on the Rudin curriculum with a height scan or a depth camera can be watched on
the terrain it was trained for. They must match how the weights were trained:
the observation size (and the policy class) is derived from them.

Usage examples:
    python play_many_dog.py                           # default: EnvCfg num_envs, viewer on
    python play_many_dog.py --num_envs 16            # 16 robots running together
    python play_many_dog.py --no_rand_cmd            # fixed velocity command
    python play_many_dog.py --gait_mode 1            # fixed trot gait
    python play_many_dog.py --weights your_ckpt.pth  # specify a weight file

    # a height-scan policy on the curriculum terrain (screenshots for the thesis)
    python play_many_dog.py --terrain rudin --obs-mode height --num_envs 64 \
        --weights results/train/height_s0/quad_diffsim_srbd_align_multi_robot.pth
"""

import os
import argparse

# NOTE: Isaac Gym must be imported before torch (hard requirement of isaacgym).
try:
    from isaacgym import gymapi
except Exception:
    pass

import torch

from config import EnvCfg, PURE_PAPER_MODE
from env import RealQuadEnv
from policy import Policy, VisionPolicy
from terrain import terrain_family_by_column
from evaluate_rudin_comparison import policy_action_dim, split_action


def load_policy(weight_path: str,
                device: torch.device,
                dim_obs: int = 36,
                dim_action: int = 12,
                policy: torch.nn.Module = None) -> torch.nn.Module:
    """Load weights from ``weight_path`` into ``policy`` (a fresh Policy by default).

    Pass ``policy`` to play back a VisionPolicy checkpoint; otherwise a blind
    ``Policy`` of the given size is built. Falls back to a randomly initialised
    network (with a warning) when the weight file does not exist, so the script
    stays usable for smoke tests.
    """
    policy = (policy if policy is not None else Policy(dim_obs, dim_action)).to(device)
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        policy.load_state_dict(state)
        print(f"[play] Loaded policy weights from {weight_path}.")
    else:
        print(f"[play] WARNING: weight file {weight_path} not found, using randomly initialized policy.")
    policy.eval()
    return policy


def _pin_spawn_cell(env, cfg, level, col):
    """Put every robot in one curriculum cell, for a like-for-like screenshot.

    The env randomises the difficulty row over 0..max_init_terrain_level at
    construction, so playback otherwise shows the policy on easy terrain no matter
    what training achieved. ``_assign_rudin_origins`` runs once in ``__init__``, so
    overriding here survives the ``reset()`` that follows.
    """
    num_rows = int(cfg.rudin_terrain.num_rows)
    num_cols = int(cfg.rudin_terrain.num_cols)
    families = terrain_family_by_column(num_cols, list(cfg.rudin_terrain.terrain_proportions))

    if level is not None:
        lvl = max(0, min(num_rows - 1, int(level)))
        if lvl != int(level):
            print(f"[play] clamped --terrain-level {level} to {lvl} (grid has {num_rows} rows)")
        env.terrain_levels[:] = lvl

    if col is not None:
        try:
            idx = max(0, min(num_cols - 1, int(col)))
        except ValueError:
            wanted = col.strip().lower().replace("-", " ")
            matches = [j for j, f in enumerate(families) if f == wanted]
            if not matches:
                raise SystemExit(
                    f"--terrain-col {col!r} matched no terrain family; "
                    f"available: {sorted(set(families))}")
            idx = matches[len(matches) // 2]      # middle column of that family
        env.terrain_types[:] = idx

    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    lvl0 = int(env.terrain_levels[0].item())
    col0 = int(env.terrain_types[0].item())
    if col is not None:
        print(f"[play] pinned every robot to row {lvl0} / column {col0} "
              f"({families[col0]}); promote-demote curriculum disabled")
        # Every robot now shares one 8x8 m cell and spawns within
        # +-rudin_spawn_jitter_m of its centre, so a large batch piles up on top of
        # itself. Pinning only the row instead spreads robots over the 20 columns.
        jitter = float(getattr(cfg, "rudin_spawn_jitter_m", 1.0))
        if env.B > 8:
            print(f"[play] WARNING: {env.B} robots share one cell, all spawning within "
                  f"+-{jitter:g} m of its centre -- they will overlap. Use --num_envs 4..8 "
                  f"for a single-cell shot, or drop --terrain-col to spread them across "
                  f"the row's 20 terrain types.")
    else:
        print(f"[play] pinned every robot to difficulty row {lvl0}, spread across all "
              f"{num_cols} terrain columns; promote-demote curriculum disabled")


def main():
    """Parse arguments, build the env + policy and run the playback loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join("results", "quad_diffsim_srbd_align_multi_robot.pth"),
        help="Policy weight file path (default: pth saved by training script)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel quadrupeds (if not specified, use EnvCfg default)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on: cuda / cpu"
    )
    parser.add_argument(
        "--no_rand_cmd",
        action="store_true",
        help="Disable random velocity commands, use the fixed cfg.cmd_fixed command"
    )
    parser.add_argument(
        "--gait_mode",
        type=int,
        default=None,
        help="Gait mode: -1=random per env, 0=stand, 1=trot, 2=pace, 3=bound"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum steps to run; 0 means run indefinitely until viewer is closed"
    )
    parser.add_argument(
        "--terrain",
        type=str,
        default=None,
        choices=["flat", "rough", "rudin"],
        help="Terrain to play on. Must match training: 'rudin' is the curriculum grid "
             "(and switches to the Rudin omnidirectional command ranges). "
             "Default: whatever EnvCfg specifies."
    )
    parser.add_argument(
        "--terrain-level",
        dest="terrain_level",
        type=int,
        default=None,
        help="Rudin terrain only: spawn every robot on this difficulty row (0 = easiest) "
             "instead of the random 0..max_init_terrain_level placement. This is what "
             "lets a screenshot show the policy at the difficulty training reached -- "
             "read it off final_state.json. Also disables the promote/demote curriculum "
             "so the robots stay on that row."
    )
    parser.add_argument(
        "--terrain-col",
        dest="terrain_col",
        type=str,
        default=None,
        help="Rudin terrain only: spawn every robot in this terrain column, given as an "
             "index (0..num_cols-1) or a family name ('stairs up', 'stairs down', "
             "'smooth slope', 'rough slope', 'discrete obstacles')."
    )
    parser.add_argument(
        "--obs-mode",
        dest="obs_mode",
        type=str,
        default="blind",
        choices=["blind", "height", "depth"],
        help="Policy input, must match how the weights were trained: 36-D blind obs, "
             "36+187 privileged height scan, or 36-D obs + depth image through the "
             "VisionPolicy CNN."
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # -------- Build EnvCfg --------
    cfg = EnvCfg()
    # Number of parallel dogs
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs

    # Play mode: must enable viewer
    cfg.use_viewer = True
    cfg.use_gpu_pipeline = True

    # Terrain + perception, mirroring train.py's mode table. env asserts that the
    # height/depth flags require use_perception and are mutually exclusive.
    if args.terrain is not None:
        cfg.terrain_type = args.terrain
    if args.obs_mode != "blind":
        cfg.use_perception = True
        if args.obs_mode == "height":
            cfg.use_height_obs = True
        else:
            cfg.use_depth_obs = True

    # Velocity command switch
    if args.no_rand_cmd:
        cfg.rand_cmd = False  # use cfg.cmd_fixed instead of per-env random commands
    # Gait mode (optional)
    if args.gait_mode is not None:
        cfg.gait_mode = args.gait_mode

    # Pinning a cell only means anything on the curriculum grid, and it has to be
    # decided before the env is built so the curriculum can be switched off.
    pin = args.terrain_level is not None or args.terrain_col is not None
    if pin and cfg.terrain_type != "rudin":
        raise SystemExit("--terrain-level / --terrain-col require --terrain rudin")
    if pin:
        cfg.rudin_terrain.dynamic_curriculum = False

    # -------- Build environment & policy --------
    env = RealQuadEnv(cfg, device=device)

    if pin:
        _pin_spawn_cell(env, cfg, args.terrain_level, args.terrain_col)

    env.reset()  # standing posture + random/fixed commands

    # env.obs_dim keeps the policy size consistent with the env configuration
    # (36 blind, 36+187 with the height scan).
    # Step 3 made dim_action variable (12, or 12+4/12+8 for a foothold-residual policy), so it
    # is read off the checkpoint's last layer rather than assumed -- otherwise loading one of
    # those is a shape error. Old 12-output checkpoints still report 12.
    dim_action = policy_action_dim(args.weights)
    net = (VisionPolicy(dim_obs=env.obs_dim, dim_action=dim_action)
           if args.obs_mode == "depth" else None)
    policy = load_policy(args.weights, device, dim_obs=env.obs_dim,
                         dim_action=dim_action, policy=net)
    B = env.B

    # Same action_hold / smoothing logic as train.py.
    hx = None
    a_prev = torch.zeros(B, dim_action, device=device)
    hx_hold = None

    t = 0
    steps_done = 0

    print("[play] Starting multi-robot policy playback")
    print(f"[play] Number of parallel robots B = {B}")
    print("[play] Move the camera freely in the Isaac Gym viewer; close the window to exit.")

    try:
        while True:
            # Observation: exactly same as training (B, 36)
            with torch.no_grad():
                s = env.get_obs().to(device)

                def act(obs, hidden):
                    """One policy call; in depth mode also captures the camera."""
                    if args.obs_mode == "depth":
                        return policy(obs, env.collect_perception()["depth_clean"], hidden)
                    return policy(obs, hidden)

                if PURE_PAPER_MODE:
                    # Paper version: only action_hold, no action smoothing
                    if (t % cfg.action_hold) == 0:
                        a, hx = act(s, hx)  # (B, 12)
                        a_prev = a
                        hx_hold = hx
                    else:
                        a = a_prev
                        hx = hx_hold
                else:
                    # Engineering version: action_hold + simple EMA smoothing
                    if (t % cfg.action_hold) == 0:
                        a_raw, hx = act(s, hx)  # (B, 12)
                        a_smooth = 0.7 * a_prev + 0.3 * a_raw
                        a_prev = a_smooth
                        hx_hold = hx
                        a = a_smooth
                    else:
                        a = a_prev
                        hx = hx_hold

            # Environment step forward (automatically calls IsaacGym simulate + viewer refresh)
            # Only the 12 joint offsets drive the simulator; a residual policy's extra
            # foothold outputs are a training-time signal to the gait planner.
            _, extra, q_err, q_ref = env.step(split_action(a))

            done = extra["done"]  # (B,)
            if done.any():
                fallen_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                print(f"[play] Partial reset envs: {fallen_ids.cpu().tolist()}")
                env.reset_envs(fallen_ids)
                # A future recurrent policy would need its hidden state zeroed
                # for these envs here; the feed-forward Policy has none.

            t += 1
            steps_done += 1

            # Exit if viewer is closed
            if env.viewer is None:
                print("Viewer closed, exiting play.")
                break

            # Limit maximum steps (optional)
            if args.max_steps > 0 and steps_done >= args.max_steps:
                print(f"Reached maximum steps {args.max_steps}, exiting play.")
                break

    except KeyboardInterrupt:
        print("Received Ctrl+C, exiting play.")


if __name__ == "__main__":
    main()
