"""Training entry point: differentiable-SRBD policy optimisation.

Rolls the policy out for ``steps_per_iter`` control steps per iteration --
Isaac Gym provides the ground-truth physics, the SRBD model re-integrates the
estimated contact forces differentiably, and the two are blended by
alpha-alignment (value from Isaac, gradient corridor from SRBD). The loss is
assembled from the SRBD states and backpropagated through the whole rollout
into the policy (and, in depth mode, into the vision encoder).

Run modes, selected with the MODE environment variable:
    blind (default) -- flat terrain, 36-D observation, no perception
    blind_rudin     -- Rudin terrain, 36-D observation, no perception
    hobs            -- Rudin terrain, 36+187-D obs (height scan), no terrain loss
    hloss           -- Rudin terrain, 36-D obs, terrain-gradient losses
    height          -- Rudin terrain, 36+187-D obs + terrain-gradient losses
    depth           -- Rudin terrain, VisionPolicy (obs + depth CNN) + terrain losses

`hobs` and `hloss` are the two ablation cells that separate the terrain
*observation* path from the terrain *gradient* path.

Other environment variables:
    SEED       RNG seed (default 0)
    NUM_ENVS   parallel robots, overrides EnvCfg.num_envs
    ITERS      training iterations, overrides the default 1000
    RUN_DIR    write every output of this run into this folder, and record
               per-iteration timing / memory / metrics there (see bench_log.py)
"""

import json
import os

# NOTE: Isaac Gym must be imported before torch (hard requirement of isaacgym).
try:
    from isaacgym import gymapi
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_log import BenchLog
from config import (
    CUDA_KERNEL_SRBD,
    DEBUG_TRAIN,
    FORCE_DTYPE,
    EnvCfg,
    ONLY_ITERATE_NO_RESET,
    PURE_PAPER_MODE,
)
from env import RealQuadEnv
from policy import Policy, VisionPolicy
from terrain import terrain_family_by_column
from utils_math import (
    moving_average,
    quat_to_rot,
    set_seed,
)

# Directory for all training outputs (curves, metrics, model weights).
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Run modes: the EnvCfg fields each one sets, and the suffix its output files get.
# Anything not listed keeps the EnvCfg default (perception and terrain losses off).
MODE_CFG = {
    "blind":       (dict(terrain_type="flat"),                                                                   ""),
    "blind_rudin": (dict(terrain_type="rudin"),                                                                  "_blindr"),
    "hobs":        (dict(terrain_type="rudin", use_perception=True, use_height_obs=True),                        "_hobs"),
    "hloss":       (dict(terrain_type="rudin", use_perception=True, use_terrain_loss=True),                      "_hloss"),
    "height":      (dict(terrain_type="rudin", use_perception=True, use_height_obs=True, use_terrain_loss=True), "_height"),
    "depth":       (dict(terrain_type="rudin", use_perception=True, use_depth_obs=True, use_terrain_loss=True),  "_vision"),
}


def save_curriculum_snapshot(env, cfg, out):
    """Write where every robot ended up on the curriculum grid.

    The per-iteration ``terrain_level`` curve gives the mean over robots; this is
    the distribution behind that mean at the end of training, which is what
    distinguishes "everyone reached row 5" from "half at row 0, half at row 9".

    Writes ``final_state.npz`` (per-robot arrays) and ``final_state.json``
    (histogram over difficulty rows, per-terrain-family means, distance walked).
    """
    levels = env.terrain_levels.detach().cpu().numpy().astype(np.int64)
    types = env.terrain_types.detach().cpu().numpy().astype(np.int64)
    # Distance from the cell origin. NOTE: measured at whatever point each robot
    # is in its episode when training stops, so it is a partial-episode distance,
    # not a per-episode average -- useful as a spread, not as a headline metric.
    dist = torch.norm(env.base_pos[:, 0:2] - env.env_origins[:, 0:2], dim=1)
    dist = dist.detach().cpu().numpy()

    num_rows = int(cfg.rudin_terrain.num_rows)
    num_cols = int(cfg.rudin_terrain.num_cols)
    families = terrain_family_by_column(num_cols, list(cfg.rudin_terrain.terrain_proportions))

    hist = [int((levels == r).sum()) for r in range(num_rows)]
    by_family = {}
    for name in sorted(set(families)):
        cols = [j for j, f in enumerate(families) if f == name]
        mask = np.isin(types, cols)
        if not mask.any():
            continue
        by_family[name] = {
            "n_robots": int(mask.sum()),
            "mean_terrain_level": float(levels[mask].mean()),
            "max_terrain_level": int(levels[mask].max()),
            "mean_distance_m": float(dist[mask].mean()),
        }

    np.savez(out("final_state.npz"), terrain_level=levels, terrain_type=types,
             distance_from_origin=dist)
    summary = {
        "num_robots": int(levels.size),
        "num_rows": num_rows,
        "num_cols": num_cols,
        "mean_terrain_level": float(levels.mean()),
        "max_terrain_level": int(levels.max()),
        "terrain_level_hist": hist,
        "mean_distance_m": float(dist.mean()),
        "column_to_family": families,
        "by_terrain_family": by_family,
    }
    with open(out("final_state.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train] final curriculum: mean level {summary['mean_terrain_level']:.2f} "
          f"(max {summary['max_terrain_level']}), histogram over rows {hist}")
    for name, stats in by_family.items():
        print(f"[train]   {name:20s} n={stats['n_robots']:5d}  "
              f"mean level {stats['mean_terrain_level']:.2f}")


def train(num_iters=1000, steps_per_iter=24,
          device="cuda" if torch.cuda.is_available() else "cpu",
          seed: int = None, smooth_k: int = 25,
          mode: str = "blind",
          num_envs: int = None,
          run_dir: str = None,
          perception_terrain: bool = None,
          depth_policy: bool = None):
    """Train the policy and save weights, TorchScript export and curves.

    Args:
        num_iters: Outer training iterations (one optimiser step each).
        steps_per_iter: Control steps rolled out (and backpropagated through)
            per iteration.
        device: Torch device.
        seed: RNG seed (falls back to the SEED environment variable, then 0).
        smooth_k: Moving-average window for the smoothed curves.
        mode: One of ``MODE_CFG`` -- blind / blind_rudin / hobs / hloss /
            height / depth. See the module docstring.
        num_envs: Overrides ``EnvCfg.num_envs`` when given.
        run_dir: When set, every output of this run goes into this folder and
            per-iteration timing / memory / metrics are recorded there
            (``iters.csv`` / ``meta.json`` / ``summary.json``). When ``None``
            the previous behaviour is kept: outputs land in ``results/`` with a
            mode suffix, and nothing is measured.
        perception_terrain: Deprecated alias -- ``True`` selects ``height``.
        depth_policy: Deprecated alias -- ``True`` selects ``depth``.
    """
    # Deprecated boolean aliases, kept so older call sites and docs still work.
    if depth_policy:
        mode = "depth"
    elif perception_terrain:
        mode = "height"
    if mode not in MODE_CFG:
        raise SystemExit(f"unknown mode {mode!r}; expected one of {sorted(MODE_CFG)}")
    mode_fields, run_tag = MODE_CFG[mode]
    depth_policy = (mode == "depth")

    if seed is None: seed = int(os.getenv("SEED", 0))
    set_seed(seed)

    cfg = EnvCfg()
    cfg.trot_style = "normal"      # "normal" / "walk" / "run"
    cfg.train_no_aerial = True
    for field, value in mode_fields.items():
        setattr(cfg, field, value)
    if num_envs is not None:
        cfg.num_envs = int(num_envs)

    if cfg.terrain_type == "flat":
        # Flat-ground command: a fixed forward 0.5 m/s. On the Rudin terrain these
        # are ignored -- env._sample_command switches to the Rudin-matched
        # omnidirectional ranges whenever terrain_type == "rudin".
        cfg.rand_cmd = False
        cfg.vx_min = +0.5
        cfg.vx_max = +0.5

    if PURE_PAPER_MODE:
        cfg.use_paper_raibert = True

    env = RealQuadEnv(cfg, device=device)
    B = env.B

    # Per-run measurement folder. Disabled (all no-ops) when run_dir is None.
    bench = BenchLog(run_dir, meta={
        "mode": mode, "seed": seed, "num_envs": B, "iters": num_iters,
        "steps_per_iter": steps_per_iter,
        "srbd_backend": "cuda" if CUDA_KERNEL_SRBD else "torch",
        "force_dtype": FORCE_DTYPE,
        "pure_paper_mode": PURE_PAPER_MODE,
        "terrain_type": cfg.terrain_type,
        "dt": cfg.dt,
        # One iteration advances physics steps_per_iter times, so this converts an
        # iteration index into simulated seconds per robot -- the x-axis that makes
        # this comparable to legged_gym (24 * 4 * 0.005 = 0.48 s there).
        "sim_seconds_per_iter": cfg.dt * steps_per_iter,
        "device": str(device),
    })
    if depth_policy:
        model = VisionPolicy(dim_obs=env.obs_dim, dim_action=12).to(device)
    else:
        model = Policy(dim_obs=env.obs_dim, dim_action=12).to(device)

    def policy_act(s, hx):
        """One policy call; in depth mode also captures the camera this step.

        The CNN forward runs here, inside the grad-enabled rollout (not in the
        no_grad get_obs), so the encoder weights land on the autograd graph and
        receive gradients through the action -> SRBD -> loss chain. depth_clean
        is a fresh tensor per capture, so holding it across the BPTT window is
        safe even though the raw pixel buffer is reused in place.
        """
        if depth_policy:
            perc = env.collect_perception()
            return model(s, perc["depth_clean"], hx)
        return model(s, hx)

    # run_tag comes from MODE_CFG above: it keeps blind / height / vision runs from
    # overwriting each other's curves and weights when they share one results dir.
    opt = AdamW(model.parameters(), lr=1e-3)

    # Loss weights: velocity, height, angular velocity, control effort,
    # gravity projection, foot tracking, terrain clearance (terrain mode only).
    a1, a2, a3, a4, a5, a6 = 10, 1.0, 0.01, 0.01, 0.5, 5.0
    a7 = 3.0
    use_terrain_loss = getattr(cfg, "use_terrain_loss", False)
    # True only on the Rudin curriculum grid. Gates the terrain-relative height reference and
    # the commanded-yaw reference below, so flat / rough runs keep their previous behaviour
    # exactly (and the flat control run stays a valid regression test of both).
    on_terrain = (cfg.terrain_type == "rudin")

    pbar = tqdm(range(num_iters), ncols=92)
    losses = []; rewards = []
    vx_iter_track = []

    loss_v_hist_iter = []
    loss_h_hist_iter = []
    loss_omega_hist_iter = []
    loss_ctrl_hist_iter = []
    loss_gproj_hist_iter = []
    loss_foot_hist_iter = []
    loss_clear_hist_iter = []
    terrain_level_iter = []

    # Outer loop
    for it in pbar:
        bench.start()
        # α alignment scheduling
        if PURE_PAPER_MODE:
            env.cfg.alpha_align = 0.9
        else:
            alpha = 0.4 + 0.4 * (it / max(1, num_iters - 1))
            alpha = float(min(alpha, 0.8))
            env.cfg.alpha_align = alpha

        # Only do global reset at iter=0; afterwards rely on local reset_envs
        if it == 0:
            env.reset(it=it)
            model.reset()
        else:
            if not ONLY_ITERATE_NO_RESET:
                env.reset(it=it)
                model.reset()

        episodic_reward = 0.0

        v_world_hist = []
        q_body_hist  = []
        pz_hist  = []
        ureg_hist = []
        omega_hist, gproj_hist = [], []
        foot_ref_hist = []
        clearance_hist = []
        cmd_hist = []       # per-step high-level command [vx_cmd, vy_cmd, yaw_cmd]
        yawref_hist = []    # per-step commanded yaw reference (terrain only; see loss_yaw)


        hx = None
        hx_hold = None
        n_falls_iter = 0
        n_timeouts_iter = 0

        a_prev = torch.zeros(B, 12, device=device)

        # Inner loop: multiple simulation and training steps per iter
        for t in range(steps_per_iter):
            # Observation (B,36)
            s = env.get_obs().to(device)


            # RNN / action_hold logic (keep as is)
            if PURE_PAPER_MODE:
                if (t % cfg.action_hold) == 0:
                    a, hx = policy_act(s, hx)   # (B,12)
                    a_prev = a
                    hx_hold = hx.detach() if hx is not None else None
                else:
                    a = a_prev.detach()
                    hx = hx_hold
            else:
                if (t % cfg.action_hold) == 0:
                    a_raw, hx = policy_act(s, hx)
                    a_smooth = 0.7 * a_prev + 0.3 * a_raw
                    a_prev = a_smooth.detach()
                    hx_hold = hx.detach() if hx is not None else None
                    a = a_smooth
                else:
                    a = a_prev
                    hx = hx_hold

            # ------------------- Isaac Gym simulation, one control step -------------------
            _, extra, q_err, qref = env.step(a)

            # ------------------- gait plan for this step -------------------
            p_foot = env.foot_positions()                     # (B,4,3)

            # Phase (B,4)
            if hasattr(env, "leg_phase_offsets_B"):
                phase_offsets = env.leg_phase_offsets_B
            else:
                phase_offsets = env.leg_phase_offsets.view(1,4).repeat(B,1)
            phases = phase_offsets + env.phase.view(B,1)      # (B,4)
            # Stance/swing masks + foot targets from the gait planner (Raibert touchdown).
            pref, stance_mask, vref_foot = env.gait._update_foot_targets_from_command(
                phases, p_foot, return_vref=True
            )
            swing_mask  = 1.0 - stance_mask                   # (B,4,1)

            # Per-env swing-height target, reused by the clearance loss below.
            if hasattr(env, "swing_height_B"):
                h_tar = env.swing_height_B.view(B, 1, 1)                     # (B,1,1)
            else:
                h_tar = torch.full((B,1,1), env.cfg.swing_height, device=env.device)

            # ------------------- SRBD step + alpha alignment -------------------
            q12      = env.q[:, env.ctrl_idx_t]          # (B,12)
            qd12_now = env.qd[:, env.ctrl_idx_t]
            f_est = env.estimate_foot_forces(q_ref12=qref,
                                             q_now12=q12.detach(),
                                             qd_now12=qd12_now.detach(),
                                             stance_mask=stance_mask.detach())
            env.srbd._srbd_step(f_world=f_est, q_ref12=qref, dt=env.cfg.dt)

            # Alpha alignment: pin the SRBD state's *value* to Isaac's ground
            # truth while keeping the SRBD gradient corridor (scaled by alpha).
            alpha = env.cfg.alpha_align
            if env.cfg.use_strict_alpha_align:
                env.srbd_p = env.base_pos + alpha * (env.srbd_p - env.srbd_p.detach())
                env.srbd_v = env.base_lin + alpha * (env.srbd_v - env.srbd_v.detach())
                env.srbd_q = env.base_quat + alpha * (env.srbd_q - env.srbd_q.detach())
                env.srbd_q = env.srbd._quat_norm(env.srbd_q)
                env.srbd_w = env.base_ang_body + alpha * (env.srbd_w - env.srbd_w.detach())
            else:
                env.srbd_v = env.base_lin + alpha * (env.srbd_v - env.srbd_v.detach())
                env.srbd_p = env.srbd_p.clone()
                env.srbd_p[:,2] = env.base_pos[:,2] + alpha * (env.srbd_p[:,2] - env.srbd_p[:,2].detach())
                if use_terrain_loss:
                    # Terrain is sampled at srbd_p xy, so pin its *value* to the real
                    # (Isaac) position while keeping the SRBD gradient corridor -- else
                    # the SRBD xy drifts and the loss reads terrain at the wrong spot.
                    # The strict branch above already aligns the full position.
                    env.srbd_p[:, :2] = env.base_pos[:, :2] + alpha * (
                        env.srbd_p[:, :2] - env.srbd_p[:, :2].detach()
                    )

            v_hat3 = env.srbd_v.clone()           # (B,3)
            pz_hat = env.srbd_p[:, 2]            # (B,)
            if on_terrain:
                # Terrain-relative base height: (pz - terrain_z) is what loss_h compares to h0.
                # Unconditional on terrain, NOT gated on use_terrain_loss -- gating it there made
                # that flag switch two things at once (add the clearance term AND fix the height
                # loss), which confounded the 2x2 ablation: blind_rudin/hobs measured loss_h from
                # the world datum and so were told to stand h0 above z=0 regardless of the ground
                # under them. Grad-free lookup on purpose: the reference is the terrain, so no
                # terrain information leaks into the gradient and the blind arm stays blind. The
                # gradient still flows through pz itself, which is the quantity being controlled.
                # Sampled at the *real* base xy, not the SRBD xy: the SRBD xy is only pinned to
                # the real position when use_terrain_loss is set (see the alpha-align block
                # above), so on the blind arms it drifts and would read the ground at the wrong
                # spot. Being grad-free there is no reason to prefer the SRBD position, and this
                # makes loss_h measure exactly the height the fall check thresholds on.
                pz_hat = pz_hat - env._terrain_height(env.base_pos[:, :2])

            v_world_hist.append(v_hat3.clone())
            q_body_hist.append(env.srbd_q.clone())
            pz_hist.append(pz_hat.clone())
            cmd_hist.append(env.cmd_rand.clone())      # current [vx_cmd, vy_cmd, yaw_cmd]
            if on_terrain:
                # Where the yaw command says this robot should be pointing right now: the yaw it
                # was reset to, plus the commanded yaw rate integrated over its time since reset.
                # Captured per step rather than rebuilt after the loop so that a robot which
                # resets mid-window pairs its pre-reset yaw with its pre-reset elapsed time --
                # both fields change together on reset_envs.
                yawref_hist.append(
                    env.last_reset_yaw + env.cmd_rand[:, 2] * env.ep_len_buf.to(env.cmd_rand.dtype) * cfg.dt
                )
            ureg_hist.append(a.clone())

            p_foot_srbd = env.srbd.foot_positions_srbd(qref)  # (B,4,3)
            foot_err_vec = (p_foot_srbd - pref) * swing_mask
            foot_ref_hist.append(foot_err_vec.clone())

            if use_terrain_loss:
                # Swing-foot clearance: penalise swing feet below terrain + margin.
                # foot_tz is sampled (blurred field) at the differentiable SRBD foot
                # xy, so the gradient both lifts the foot (z) and pushes the foothold
                # away from riser edges (xy, via the terrain slope).
                foot_tz = env.terrain_height_diff(p_foot_srbd[..., :2])       # (B,4)
                # Required clearance follows the swing phase instead of being a constant
                # swing_height. _swing_parabola interpolates through (liftoff, apex, touchdown),
                # giving z(s) = z0 + (z1-z0)*s^2 + 4*h*s*(1-s) -- so between feet at equal height
                # it lifts exactly 4*h*s*(1-s): zero at both ends, h at mid-swing. A constant
                # margin demanded h even at the instants the foot must be on the ground, so the
                # term had a floor it could never reach and fought loss_foot (weight 5.0) at
                # every touchdown -- which is why loss_clear ran backwards in every terrain-loss
                # run. Modulated, it asks for that same arc but referenced to the terrain rather
                # than to last_contact_z, so it is positive exactly where the terrain-blind arc
                # under-clears: the s^2 base makes the swing sag below a straight line onto a
                # rising step, clipping a 0.20 m riser by ~0.08 m even when the target rises.
                s_swing = env.gait.swing_progress                              # (B,4)
                clear_margin = h_tar.view(B, 1) * 4.0 * s_swing * (1.0 - s_swing)
                clear_viol = torch.relu(foot_tz + clear_margin - p_foot_srbd[..., 2])
                clearance_hist.append(clear_viol * swing_mask.squeeze(-1))     # (B,4)

            # Angular velocity (env.srbd_w is already in the body frame)
            omega_hist.append(env.srbd_w.clone())

            # Gravity projection (body frame): g_body = R(q)^T @ g_world, batched
            q_b = env.srbd_q
            R_b = quat_to_rot(q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3], env.device)  # (B,3,3)
            g_w = torch.tensor([0.0, 0.0, -env.cfg.g], dtype=torch.float32, device=env.device)
            gproj_hist.append(torch.einsum('bji,j->bi', R_b, g_w))  # (B,3)

            done = extra["done"]              # (B,) falls
            # Episode timeout (Rudin dynamic curriculum); all-False on flat/rough so behaviour there
            # is unchanged (reset set == falls). Falls AND timeouts both reset, but only falls are
            # penalised — Rudin gives no terminal reward for time-outs.
            timeout = extra.get("timeout", torch.zeros_like(done))
            reset_mask = done | timeout

            # Local reset for fallen / timed-out robots (new command + gait for these envs).
            if reset_mask.any():
                reset_ids = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
                # Accumulated as tensors and read once per iteration, so this adds
                # no GPU->CPU sync inside the per-step loop.
                n_falls_iter = n_falls_iter + done.sum()
                n_timeouts_iter = n_timeouts_iter + timeout.sum()
                if done.any():
                    episodic_reward -= cfg.term_penalty * float(done.float().mean().item())
                env.reset_envs(reset_ids)

        # ====== Eq.(5) individual loss terms ======
        if v_world_hist:
            v_world_seq = torch.stack(v_world_hist)   # (T,B,3)
            q_seq = torch.stack(q_body_hist)         # (T,B,4)
            # q_seq: (T,B,4) in wxyz
            qw = q_seq[..., 0]
            qx = q_seq[..., 1]
            qy = q_seq[..., 2]
            qz = q_seq[..., 3]

            # yaw from quaternion (wxyz)
            siny_cosp = 2.0 * (qw*qz + qx*qy)
            cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
            yaw_seq = torch.atan2(siny_cosp, cosy_cosp)   # (T,B)

            # Reference yaw (no grad): the yaw at reset, advanced by the *commanded* yaw rate.
            # Holding the reset yaw is right only when the yaw command is zero; on the Rudin
            # terrain _sample_command draws it from +-1 rad/s, so the old fixed reference
            # penalised a robot for obeying its own command -- and since the reference was
            # pinned at reset and an episode is ~417 iterations, the penalty saturated near
            # pi^2 * yaw_w instead of decaying. Integrating the command makes this yaw tracking,
            # matching Rudin's tracking_ang_vel reward. Off the Rudin terrain nothing is
            # recorded, so flat / rough fall through to the previous expression unchanged.
            if yawref_hist:
                yaw_ref = torch.stack(yawref_hist).detach()                        # (T,B)
            else:
                yaw_ref = env.last_reset_yaw.detach().view(1, B).expand_as(yaw_seq)

            # wrap to [-pi, pi]
            yaw_err = torch.atan2(torch.sin(yaw_seq - yaw_ref), torch.cos(yaw_seq - yaw_ref))
            loss_yaw = (yaw_err ** 2).mean()


            T_steps = v_world_seq.shape[0]

            # World -> body frame linear velocity (batched over T and B)
            q_flat = q_seq.reshape(T_steps * B, 4)
            v_flat = v_world_seq.reshape(T_steps * B, 3)
            R_flat = quat_to_rot(q_flat[:, 0], q_flat[:, 1], q_flat[:, 2], q_flat[:, 3], device)  # (T*B,3,3)
            v_body_seq = torch.einsum('bji,bj->bi', R_flat, v_flat).reshape(T_steps, B, 3)


            # v_ref: the historical per-step command [vx_cmd, vy_cmd]
            vref_body = torch.zeros_like(v_body_seq)
            if cmd_hist:
                cmd_seq = torch.stack(cmd_hist)          # (T,B,3)
                vref_body[..., 0:2] = cmd_seq[..., 0:2]  # track vx, vy
            else:
                # Fallback: only use the current vx_star
                vref_body[..., 0] = env.vx_star.view(1,B).expand(T_steps,B)

            # Velocity tracking loss on (vx, vy)
            loss_v = ((v_body_seq[..., :2] - vref_body[..., :2]) ** 2).sum(-1).mean()

            # Average body vx per robot (time-averaged), for logging/plots.
            vx_env = v_body_seq[..., 0].mean(dim=0)         # (B,)
            vx_for_plot = float(vx_env.mean().item())
            if DEBUG_TRAIN:
                # .cpu() syncs and copies num_envs floats to the host every iteration;
                # at B=1024 that is 1024 formatted numbers printed per iteration.
                vx_env_np = vx_env.detach().cpu().numpy()
                print(f"[Iter {it}] vx_body per env:", np.round(vx_env_np, 3))
        else:
            loss_v = torch.tensor(0.0, device=device); vx_for_plot = 0.0

        # Periodic sanity check: body-frame velocity vs command direction (env 0).
        if DEBUG_TRAIN and it % 20 == 0:
            print(f"\n[DIRECTION CHECK] Iter {it}: Real_v_body_x={v_body_seq[0,0,0].item():+.3f}, Target_v_star={vref_body[0,0,0].item():+.3f}")
            print(f"                  World_v_x={v_world_seq[0,0,0].item():+.3f}")


        if pz_hist:
            pz_seq = torch.stack(pz_hist)  # (T,B)
            loss_h = (pz_seq - cfg.h0).abs().mean()
        else:
            loss_h = torch.tensor(0.0, device=device)

        if omega_hist:
            omega_seq = torch.stack(omega_hist)  # (T,B,3)

            # Roll/pitch regularization: keep roll/pitch angular velocity small.
            rollpitch_sq = (omega_seq[..., :2] ** 2).sum(dim=-1)   # (T,B)

            # Yaw-rate tracking: omega_z vs the commanded yaw rate.
            if cmd_hist:
                cmd_seq = torch.stack(cmd_hist)          # (T,B,3)
                yaw_cmd_seq = cmd_seq[..., 2]            # (T,B)
            else:
                yaw_cmd_seq = torch.zeros_like(omega_seq[..., 2])

            yaw_err_sq = (omega_seq[..., 2] - yaw_cmd_seq) ** 2    # (T,B)

            # Combine: both track yaw and penalize roll/pitch
            loss_omega = (rollpitch_sq + yaw_err_sq).mean()
        else:
            loss_omega = torch.tensor(0.0, device=device)


        if ureg_hist:
            u_seq = torch.stack(ureg_hist)  # (T,B,12)
            loss_ctrl = (u_seq ** 2).sum(dim=-1).mean()
        else:
            loss_ctrl = torch.tensor(0.0, device=device)

        if gproj_hist:
            gproj_seq = torch.stack(gproj_hist)  # (T,B,3)
            g_xy = gproj_seq[..., :2]
            # Normalise from m/s^2 to dimensionless so this term is not
            # naturally an order of magnitude larger than the others.
            g_xy_norm = g_xy / cfg.g
            loss_gproj = (g_xy_norm ** 2).sum(dim=-1).mean()
        else:
            loss_gproj = torch.tensor(0.0, device=device)

        if foot_ref_hist:
            foot_err_seq = torch.stack(foot_ref_hist)  # (T,B,4,3)
            loss_foot = (foot_err_seq ** 2).sum(dim=-1).mean()
        else:
            loss_foot = torch.tensor(0.0, device=device)

        if clearance_hist:
            clear_seq = torch.stack(clearance_hist)    # (T,B,4)
            loss_clearance = (clear_seq ** 2).mean()
        else:
            loss_clearance = torch.tensor(0.0, device=device)

        yaw_w = 0.1
        loss = (a1*loss_v +
                a2*loss_h +
                a3*loss_omega +
                a4*loss_ctrl +
                a5*loss_gproj +
                a6*loss_foot  +
                a7*loss_clearance +
                yaw_w * loss_yaw
                )

        opt.zero_grad(set_to_none=True)
        loss.backward()

        #------------------------------------------------
        # --- Gradient flow analysis patch (DEBUG_TRAIN only) ---
        # One GPU->CPU sync per parameter, so it stays out of campaign runs. Must run
        # BEFORE clip_grad_norm_, which rescales the gradients in place: these are the
        # pre-clip per-parameter norms.
        if DEBUG_TRAIN:
            grad_dict = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_dict[name] = param.grad.norm().item()
                else:
                    grad_dict[name] = None
            # Print gradient strength for key layers
            print(f"\n[Debug Iter {it}] Gradient Norms:")
            for name, norm in grad_dict.items():
                status = f"{norm:.8f}" if norm is not None else "MISSING (Zero/None)"
                print(f"  {name}: {status}")

            # If the input-most layer has no gradient, the connection from the physics model
            # (SRBD) back into the network is broken. Works for both Policy (net.0.weight)
            # and VisionPolicy (encoder.conv.0.weight -- the depth CNN's first conv).
            first_name = next(iter(grad_dict))
            if grad_dict[first_name] is not None and grad_dict[first_name] < 1e-9:
                print(f"[train] WARNING: gradient almost 0 at {first_name} -- physics-model gradient failed to propagate back into the network.")
        #------------------------------------------------

        # clip_grad_norm_ computes the total pre-clip gradient norm anyway and returns
        # it, so the always-on gradient-health diagnostic costs one sync per iteration
        # instead of one per parameter. Shown in the progress bar as |g|.
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 0.3))
        opt.step()

        # Detach the SRBD state at the iteration boundary (BPTT window ends here).
        env.srbd_p = env.srbd_p.detach()
        env.srbd_v = env.srbd_v.detach()
        env.srbd_q = env.srbd_q.detach()
        env.srbd_w = env.srbd_w.detach()

        loss_v_hist_iter.append(float(loss_v.detach().cpu()))
        loss_h_hist_iter.append(float(loss_h.detach().cpu()))
        loss_omega_hist_iter.append(float(loss_omega.detach().cpu()))
        loss_ctrl_hist_iter.append(float(loss_ctrl.detach().cpu()))
        loss_gproj_hist_iter.append(float(loss_gproj.detach().cpu()))
        loss_foot_hist_iter.append(float(loss_foot.detach().cpu()))
        loss_clear_hist_iter.append(float(loss_clearance.detach().cpu()))

        # Mean curriculum difficulty row across robots. This is the same quantity
        # legged_gym logs as extras["episode"]["terrain_level"], so it is the metric
        # the PPO comparison is made on. NaN on flat terrain, where it does not exist.
        levels = getattr(env, "terrain_levels", None)
        terrain_level = float(levels.float().mean().item()) if levels is not None else float("nan")
        terrain_level_iter.append(terrain_level)

        # Curriculum promotions / demotions this iteration. Read and zeroed here so the
        # counters cover exactly one iteration; without them a terrain_level collapse cannot
        # be attributed to falls rather than to the promote/demote rule itself.
        if levels is not None:
            n_move_up = float(env.n_move_up.item())
            n_move_down = float(env.n_move_down.item())
            env.n_move_up.zero_(); env.n_move_down.zero_()
        else:
            n_move_up = n_move_down = float("nan")

        bench.stop(loss=loss.item(), vx=vx_for_plot, grad_norm=grad_norm,
                   terrain_level=terrain_level,
                   loss_v=loss_v_hist_iter[-1], loss_clear=loss_clear_hist_iter[-1],
                   n_falls=float(n_falls_iter), n_timeouts=float(n_timeouts_iter),
                   n_move_up=n_move_up, n_move_down=n_move_down)

        vx_iter_track.append(vx_for_plot)
        losses.append(loss.item()); rewards.append(episodic_reward)
        pbar.set_description(f"Iter {it:4d} | loss {loss.item():.3f} | v_body_x {vx_for_plot:+.2f} | |g| {grad_norm:.3f}")


    # ===== Save curves =====
    # With RUN_DIR set every run has its own folder, so the mode suffix is dropped;
    # without it, outputs share results/ and the suffix keeps them apart.
    out_dir = run_dir or RESULTS_DIR
    out_suffix = "" if run_dir else run_tag
    os.makedirs(out_dir, exist_ok=True)
    def out(fn):
        stem, ext = os.path.splitext(fn)
        return os.path.join(out_dir, f"{stem}{out_suffix}{ext}")

    np.save(out("terrain_level_srbd_align.npy"),
            np.array(terrain_level_iter, dtype=np.float32))

    V = np.array(vx_iter_track, dtype=np.float32)
    S = steps_per_iter

    np.save(out("vx_curve_srbd_align.npy"), V)
    plt.figure(); plt.plot(V)
    plt.xlabel("Training Iteration"); plt.ylabel(f"Avg body vx over {S} steps (m/s)")
    plt.tight_layout(); plt.savefig(out("vx_curve_srbd_align.png"))

    plt.figure(); plt.plot(moving_average(V, smooth_k))
    plt.xlabel("Training Iteration"); plt.ylabel("Avg body vx (moving avg)")
    plt.tight_layout(); plt.savefig(out("vx_curve_srbd_align_smooth.png"))

    plt.figure(); plt.plot(losses); plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig(out("loss_curve_srbd_align.png"))

    loss_parts = {
        "loss_v":     np.array(loss_v_hist_iter,     dtype=np.float32),
        "loss_h":     np.array(loss_h_hist_iter,     dtype=np.float32),
        "loss_omega": np.array(loss_omega_hist_iter, dtype=np.float32),
        "loss_ctrl":  np.array(loss_ctrl_hist_iter,  dtype=np.float32),
        "loss_gproj": np.array(loss_gproj_hist_iter, dtype=np.float32),
        "loss_foot":  np.array(loss_foot_hist_iter,  dtype=np.float32),
        "loss_clear": np.array(loss_clear_hist_iter, dtype=np.float32),
    }

    for name, arr in loss_parts.items():
        np.save(out(f"{name}_srbd_align.npy"), arr)
        plt.figure()
        plt.plot(arr)
        plt.xlabel("Iteration")
        plt.ylabel(name)
        plt.tight_layout()
        plt.savefig(out(f"{name}_curve_srbd_align.png"))

    plt.figure()
    for name, arr in loss_parts.items():
        plt.plot(arr, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("loss_components_curve_srbd_align.png"))

    R = np.array(rewards, dtype=np.float32); np.save(out("rewards_srbd_align.npy"), R)
    plt.figure(); plt.plot(moving_average(R, smooth_k))
    plt.xlabel("Training Iteration"); plt.ylabel("Reward (moving avg)")
    plt.tight_layout(); plt.savefig(out("reward_curve_srbd_align.png"))

    torch.save(model.state_dict(), out("quad_diffsim_srbd_align_multi_robot.pth"))
    # ===== Additional TorchScript export (for ROS2 deployment) =====
    model.eval()

    class PolicyActOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            a, _ = self.m(x)
            return a

    class VisionActOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, d):
            a, _ = self.m(x, d)
            return a

    if depth_policy:
        wrapper = VisionActOnly(model).to(device)
        example_obs = (
            torch.zeros(1, env.obs_dim, device=device),
            torch.zeros(1, 1, cfg.perception.out_h, cfg.perception.out_w, device=device),
        )
    else:
        wrapper = PolicyActOnly(model).to(device)
        example_obs = torch.zeros(1, env.obs_dim, device=device)  # 36 blind, 36+187 with height obs
    traced = torch.jit.trace(wrapper, example_obs)
    traced.save(out("quad_diffsim_srbd_align_multi_robot.pt"))
    print(f"[train] Saved TorchScript: {out('quad_diffsim_srbd_align_multi_robot.pt')}")

    # Where every robot ended up on the curriculum -- the distribution behind the
    # mean terrain_level curve. Rudin terrain only; flat ground has no curriculum.
    if getattr(env, "terrain_levels", None) is not None:
        save_curriculum_snapshot(env, cfg, out)

    bench.close()
    print("[train] Training done (multi-robot SRBD + alpha-align, Eq.(5) loss, body-frame vx tracking).")
    
def _env_int(name, default=None, minimum=1):
    """Int from the environment, at least `minimum`; ignored with a warning if not.

    `minimum` is 1 for counts (NUM_ENVS, ITERS) and 0 for SEED, where 0 is a
    perfectly good seed and also the default.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value < minimum:
            raise ValueError
        return value
    except ValueError:
        print(f"[train] ignoring {name}={raw!r} (expected an integer >= {minimum})")
        return default


def _env_mode():
    """Run mode from MODE, falling back to the older PERCEPTION_TERRAIN spelling."""
    raw = os.getenv("MODE")
    if raw:
        return raw.strip().lower()
    legacy = os.getenv("PERCEPTION_TERRAIN", "0").strip().lower()
    return {"1": "height", "height": "height", "depth": "depth"}.get(legacy, "blind")


if __name__ == "__main__":
    train(num_iters=_env_int("ITERS", 1000),
          steps_per_iter=24,
          seed=_env_int("SEED", 0, minimum=0),
          mode=_env_mode(),
          num_envs=_env_int("NUM_ENVS"),
          run_dir=os.getenv("RUN_DIR"))

