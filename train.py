"""Training entry point: differentiable-SRBD policy optimisation.

Rolls the policy out for ``steps_per_iter`` control steps per iteration --
Isaac Gym provides the ground-truth physics, the SRBD model re-integrates the
estimated contact forces differentiably, and the two are blended by
alpha-alignment (value from Isaac, gradient corridor from SRBD). The loss is
assembled from the SRBD states and backpropagated through the whole rollout
into the policy (and, in depth mode, into the vision encoder).

Run modes (also selectable via the PERCEPTION_TERRAIN environment variable):
    blind (default)          -- flat terrain, 36-D observation
    perception_terrain=True  -- Rudin terrain, 36+187-D obs + terrain losses
    depth_policy=True        -- Rudin terrain, VisionPolicy (obs + depth CNN)
"""

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

from config import EnvCfg, ONLY_ITERATE_NO_RESET, PURE_PAPER_MODE
from env import RealQuadEnv
from policy import Policy, VisionPolicy
from utils_math import (
    moving_average,
    quat_to_rot,
    set_seed,
)

# Directory for all training outputs (curves, metrics, model weights).
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def train(num_iters=1000, steps_per_iter=24,
          device="cuda" if torch.cuda.is_available() else "cpu",
          seed: int = None, smooth_k: int = 25,
          perception_terrain: bool = False,
          depth_policy: bool = False):
    """Train the policy and save weights, TorchScript export and curves to results/.

    Args:
        num_iters: Outer training iterations (one optimiser step each).
        steps_per_iter: Control steps rolled out (and backpropagated through)
            per iteration.
        device: Torch device.
        seed: RNG seed (falls back to the SEED environment variable, then 0).
        smooth_k: Moving-average window for the smoothed curves.
        perception_terrain: Stage-1 mode -- Rudin terrain, privileged height
            scan in the observation, terrain-aware losses.
        depth_policy: Stage-2 mode -- Rudin terrain, VisionPolicy consuming the
            depth camera through a CNN, terrain-aware losses.
    """
    if seed is None: seed = int(os.getenv("SEED", 0))
    set_seed(seed)

    cfg = EnvCfg()
    cfg.trot_style = "normal"      # "normal" / "walk" / "run"
    cfg.rand_cmd = False           # fixed command below (set True for per-env random commands)
    cfg.vx_min = +0.5
    cfg.vx_max = +0.5
    cfg.train_no_aerial = True

    # Perception / hard-terrain training on the Rudin curriculum terrain with the
    # terrain-aware differentiable losses (terrain-relative height + swing-foot
    # clearance sampled at SRBD-predicted positions). Two policy-input variants:
    #   perception_terrain -> 36+187-D obs (privileged height scan, stage 1)
    #   depth_policy       -> 36-D obs + depth image through a CNN (stage 2)
    if perception_terrain or depth_policy:
        cfg.terrain_type = "rudin"
        cfg.use_perception = True
        cfg.use_terrain_loss = True
        if depth_policy:
            cfg.use_depth_obs = True
        else:
            cfg.use_height_obs = True

    if PURE_PAPER_MODE:
        cfg.use_paper_raibert = True

    env = RealQuadEnv(cfg, device=device)
    B = env.B
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

    # Tag output files by run mode so blind / height / vision runs don't overwrite
    # each other's curves and weights.
    run_tag = "_vision" if depth_policy else ("_height" if perception_terrain else "")
    opt = AdamW(model.parameters(), lr=1e-3)

    # Loss weights: velocity, height, angular velocity, control effort,
    # gravity projection, foot tracking, terrain clearance (terrain mode only).
    a1, a2, a3, a4, a5, a6 = 10, 1.0, 0.01, 0.01, 0.5, 5.0
    a7 = 3.0
    use_terrain_loss = getattr(cfg, "use_terrain_loss", False)

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

    # Outer loop
    for it in pbar:
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


        hx = None
        hx_hold = None

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
            obs, extra, q_err, qref = env.step(a)

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
            if use_terrain_loss:
                # Terrain-relative base height: (pz - terrain_z) is compared to h0 in
                # loss_h / r_stab below. terrain_z is sampled differentiably at the
                # SRBD xy, so the local terrain slope back-props into the policy.
                terr_z = env.terrain_height_diff(env.srbd_p[:, :2].unsqueeze(1)).squeeze(1)  # (B,)
                pz_hat = pz_hat - terr_z

            v_world_hist.append(v_hat3.clone())
            q_body_hist.append(env.srbd_q.clone())
            pz_hist.append(pz_hat.clone())
            cmd_hist.append(env.cmd_rand.clone())      # current [vx_cmd, vy_cmd, yaw_cmd]
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
                clear_margin = h_tar.view(B, 1)                                # swing apex target
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

            # reference yaw: the yaw at reset (no grad)
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
            vx_env_np = vx_env.detach().cpu().numpy()

            vx_for_plot = float(vx_env.mean().item())
            print(f"[Iter {it}] vx_body per env:", np.round(vx_env_np, 3))
        else:
            loss_v = torch.tensor(0.0, device=device); vx_for_plot = 0.0

        # Periodic sanity check: body-frame velocity vs command direction (env 0).
        if it % 20 == 0:
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
        # --- Gradient flow analysis patch ---
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_dict[name] = grad_norm
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


        nn.utils.clip_grad_norm_(model.parameters(), 0.3)
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

        vx_iter_track.append(vx_for_plot)
        losses.append(loss.item()); rewards.append(episodic_reward)
        pbar.set_description(f"Iter {it:4d} | loss {loss.item():.3f} | v_body_x {vx_for_plot:+.2f}")


    # ===== Save curves =====
    os.makedirs(RESULTS_DIR, exist_ok=True)
    def out(fn):
        stem, ext = os.path.splitext(fn)
        return os.path.join(RESULTS_DIR, f"{stem}{run_tag}{ext}")

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

    print("[train] Training done (multi-robot SRBD + alpha-align, Eq.(5) loss, body-frame vx tracking).")
    
if __name__ == "__main__":
    # PERCEPTION_TERRAIN=1|height -> Rudin terrain + privileged height-map obs (stage 1)
    # PERCEPTION_TERRAIN=depth    -> Rudin terrain + depth-CNN vision policy (stage 2)
    _mode = os.getenv("PERCEPTION_TERRAIN", "0").lower()
    train(num_iters=1000, steps_per_iter=24, seed=0,
          perception_terrain=_mode in ("1", "height"),
          depth_policy=_mode == "depth")

