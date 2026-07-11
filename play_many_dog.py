# -*- coding: utf-8 -*-
"""play_many_dog.py - Visualise a trained policy on multiple robots in Isaac Gym.

Loads a trained blind policy and runs it live in the Isaac Gym viewer (no SRBD
model, no gradients -- pure playback with per-env resets on falls).

Usage examples:
    python play_many_dog.py                           # default: EnvCfg num_envs, viewer on
    python play_many_dog.py --num_envs 16            # 16 robots running together
    python play_many_dog.py --no_rand_cmd            # fixed velocity command
    python play_many_dog.py --gait_mode 1            # fixed trot gait
    python play_many_dog.py --weights your_ckpt.pth  # specify a weight file
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
from policy import Policy


def load_policy(weight_path: str,
                device: torch.device,
                dim_obs: int = 36,
                dim_action: int = 12) -> Policy:
    """Build a Policy of the given size and load weights from ``weight_path``.

    Falls back to a randomly initialised policy (with a warning) when the
    weight file does not exist, so the script stays usable for smoke tests.
    """
    policy = Policy(dim_obs, dim_action).to(device)
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        policy.load_state_dict(state)
        print(f"[play] Loaded policy weights from {weight_path}.")
    else:
        print(f"[play] WARNING: weight file {weight_path} not found, using randomly initialized policy.")
    policy.eval()
    return policy


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

    # Velocity command switch
    if args.no_rand_cmd:
        cfg.rand_cmd = False  # use cfg.cmd_fixed instead of per-env random commands
    # Gait mode (optional)
    if args.gait_mode is not None:
        cfg.gait_mode = args.gait_mode

    # -------- Build environment & policy --------
    env = RealQuadEnv(cfg, device=device)
    env.reset()  # standing posture + random/fixed commands

    # env.obs_dim keeps the policy size consistent with the env configuration
    # (36 blind, 36+187 if a height-scan config were enabled).
    policy = load_policy(args.weights, device, dim_obs=env.obs_dim)
    B = env.B
    dim_action = 12

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

                if PURE_PAPER_MODE:
                    # Paper version: only action_hold, no action smoothing
                    if (t % cfg.action_hold) == 0:
                        a, hx = policy(s, hx)  # (B, 12)
                        a_prev = a
                        hx_hold = hx
                    else:
                        a = a_prev
                        hx = hx_hold
                else:
                    # Engineering version: action_hold + simple EMA smoothing
                    if (t % cfg.action_hold) == 0:
                        a_raw, hx = policy(s, hx)  # (B, 12)
                        a_smooth = 0.7 * a_prev + 0.3 * a_raw
                        a_prev = a_smooth
                        hx_hold = hx
                        a = a_smooth
                    else:
                        a = a_prev
                        hx = hx_hold

            # Environment step forward (automatically calls IsaacGym simulate + viewer refresh)
            obs, extra, q_err, q_ref = env.step(a)

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
