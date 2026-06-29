import os

# ⚠️ CRITICAL: Isaac Gym must be imported BEFORE torch
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
from policy import Policy
from utils_math import (
    moving_average,
    project_gravity_to_body,
    quat_rotate_inverse_wxyz,
    quat_to_rot,
    set_seed,
)

# Directory for all training outputs (curves, metrics, model weights).
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def train(num_iters=1000, steps_per_iter=24,
          device="cuda" if torch.cuda.is_available() else "cpu",
          seed: int = None, smooth_k: int = 25):

    if seed is None: seed = int(os.getenv("SEED", 0))
    set_seed(seed)

    cfg = EnvCfg()
    cfg.trot_style = "normal"   # or "normal" / "run"
    cfg.rand_cmd = False          # ✅ Enable random velocity commands
    cfg.vx_min = +0.5        # You can change to paper's command range
    cfg.vx_max = +0.5
    cfg.train_no_aerial = True


    if PURE_PAPER_MODE:
        cfg.use_paper_raibert = True

    env = RealQuadEnv(cfg, device=device)
    B = env.B
    model = Policy(dim_obs=36, dim_action=12).to(device)
    opt = AdamW(model.parameters(), lr=1e-3)  # Further reduce learning rate
    
    a1, a2, a3, a4, a5, a6 = 10, 1.0, 0.01, 0.01 ,0.5, 5.0
     
  

    pbar = tqdm(range(num_iters), ncols=92)
    losses = []; rewards = []
    vx_iter_track = []

    loss_v_hist_iter = []
    loss_h_hist_iter = []
    loss_omega_hist_iter = []
    loss_ctrl_hist_iter = []
    loss_gproj_hist_iter = []
    loss_foot_hist_iter = []

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
        stuck_counter = 0

        v_world_hist = []
        q_body_hist  = []
        pz_hist  = []
        ureg_hist = []
        omega_hist, gproj_hist = [], []
        foot_ref_hist = []
        tilt_hist = []
        cmd_hist = []       # ★ Record high-level velocity command at each step [vx_cmd, vy_cmd, yaw_cmd]
        

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
                    a, hx = model(s, hx)   # (B,12)
                    a_prev = a
                    hx_hold = hx.detach() if hx is not None else None
                else:
                    a = a_prev.detach()
                    hx = hx_hold
            else:
                if (t % cfg.action_hold) == 0:
                    a_raw, hx = model(s, hx)
                    a_smooth = 0.7 * a_prev + 0.3 * a_raw
                    a_prev = a_smooth.detach()
                    hx_hold = hx.detach() if hx is not None else None
                    a = a_smooth
                else:
                    a = a_prev
                    hx = hx_hold

 #-------------------IsaacGym simulation one step----------------
            obs, extra, q_err, qref = env.step(a)


 #------------------SRBD step ----------------
            # Stuck detection (only watch robot 0)
            v_body_now0 = env.base_lin_body[0]
            if (v_body_now0.norm() < 0.02) and (env.base_pos[0,2] < 0.20):
                stuck_counter += 1
            else:
                stuck_counter = 0

            # Foot kinematics
            J = env.foot_jacobians()  # (B,4,3,dof)
            dof_offset = getattr(env, "jac_dof_offset", 0)
            J_ctrl = J[..., env.ctrl_idx_t + dof_offset]      # (B,4,3,12)
            qd12 = env.qd[:, env.ctrl_idx_t]                  # (B,12)
            v_foot = torch.einsum('bfkj,bj->bfk', J_ctrl, qd12)  # (B,4,3)

            p_foot = env.foot_positions()                     # (B,4,3)

            # Phase (B,4)
            if hasattr(env, "leg_phase_offsets_B"):
                phase_offsets = env.leg_phase_offsets_B
            else:
                phase_offsets = env.leg_phase_offsets.view(1,4).repeat(B,1)
            phases = phase_offsets + env.phase.view(B,1)      # (B,4)
            # ✅ Use new “stance+swing+Raibert” function to get foot target and stance_mask
            pref, stance_mask, vref_foot = env.gait._update_foot_targets_from_command(
                phases, p_foot, return_vref=True
            )
            swing_mask  = 1.0 - stance_mask                   # (B,4,1)

            # ===== Swing/stance loss =====
            # 1) Swing leg “should lift above swing_height”
            clearance = p_foot[:,:,2:3] - env.last_contact_z.unsqueeze(-1)  # (B,4,1)
            if hasattr(env, "swing_height_B"):
                h_tar = env.swing_height_B.view(B, 1, 1)                     # (B,1,1)
            else:
                h_tar = torch.full((B,1,1), env.cfg.swing_height, device=env.device)



            # SRBD step + α alignment
            q12      = env.q[:, env.ctrl_idx_t]          # (B,12)
            qd12_now = env.qd[:, env.ctrl_idx_t]
            f_est = env.estimate_foot_forces(q_ref12=qref,
                                             q_now12=q12.detach(),
                                             qd_now12=qd12_now.detach(),
                                             stance_mask=stance_mask.detach())
            env.srbd._srbd_step(f_world=f_est, q_ref12=qref, dt=env.cfg.dt)
#--------------------------------------------------------------------------------------

# After SRBD step, perform α alignment----------------------------------------------------------------
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

            v_hat3 = env.srbd_v.clone()           # (B,3)
            pz_hat = env.srbd_p[:, 2]            # (B,)

            v_world_hist.append(v_hat3.clone())
            q_body_hist.append(env.srbd_q.clone())
            pz_hist.append(pz_hat.clone())
            cmd_hist.append(env.cmd_rand.clone())      # ★ Record current [vx_cmd, vy_cmd, yaw_cmd]
            ureg_hist.append(a.clone())


            #p_foot_srbd = env.srbd.foot_positions_srbd(qref)  # (B,4,3)
            p_foot_srbd = env.srbd.foot_positions_srbd(qref)  # (B,4,3)
            foot_err_vec = (p_foot_srbd - pref) * swing_mask
            foot_ref_hist.append(foot_err_vec.clone())

            # Angular velocity (body frame) — env.srbd_w is already in body frame
            omega_hist.append(env.srbd_w.clone())

            # Gravity projection (body frame): g_body = R(q)^T @ g_world, batched
            q_b = env.srbd_q
            R_b = quat_to_rot(q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3], env.device)  # (B,3,3)
            g_w = torch.tensor([0.0, 0.0, -env.cfg.g], dtype=torch.float32, device=env.device)
            gproj_hist.append(torch.einsum('bji,j->bi', R_b, g_w))  # (B,3)

            tilt_hist.append(0.7*torch.abs(env.pitch) + 0.3*torch.abs(env.roll))  # (B,)

            # Reward (per-env, then average): v_body = R(q)^T @ v_world, batched
            v_body_dbg = torch.einsum('bji,bj->bi', R_b, v_hat3)  # (B,3)

            r_v = -(env.vx_star - v_body_dbg[:,0]).abs()      # (B,)
            r_u = -0.01 * a.detach().pow(2).mean(dim=1)       # (B,)
            r_stab = -0.8 * ((pz_hat - cfg.h0).abs() + tilt_hist[-1])



            done = extra["done"]              # (B,) falls
            # Episode timeout (Rudin dynamic curriculum); all-False on flat/rough so behaviour there
            # is unchanged (reset set == falls). Falls AND timeouts both reset, but only falls are
            # penalised — Rudin gives no terminal reward for time-outs.
            timeout = extra.get("timeout", torch.zeros_like(done))
            reset_mask = done | timeout

            # ★ Local reset for fallen / timed-out robots; also assign new vx_star + gait for these envs
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


            # ★ v_ref: use historical [vx_cmd, vy_cmd]
            vref_body = torch.zeros_like(v_body_seq)
            if cmd_hist:
                cmd_seq = torch.stack(cmd_hist)          # (T,B,3)
                vref_body[..., 0:2] = cmd_seq[..., 0:2]  # Track vx, vy
            else:
                # Fallback: only use current vx_star
                vref_body[..., 0] = env.vx_star.view(1,B).expand(T_steps,B)

            # ★ Velocity tracking loss: vx & vy
            #loss_v = ((v_body_seq - vref_body) ** 2).sum(dim=-1).mean()
            loss_v = ((v_body_seq[..., :2] - vref_body[..., :2]) ** 2).sum(-1).mean()

            # ==== New: average vx_body for each robot ====
            # v_body_seq: (T,B,3) -> average over time first -> (B,3)
            vx_env = v_body_seq[..., 0].mean(dim=0)         # (B,)
            vx_env_np = vx_env.detach().cpu().numpy()       # numpy for easy printing

            # Original overall average vx (all envs + all timesteps)
            vx_for_plot = float(vx_env.mean().item())
            # Optional: print vx for each env
            print(f"[Iter {it}] vx_body per env:", np.round(vx_env_np, 3))
        else:
            loss_v = torch.tensor(0.0, device=device); vx_for_plot = 0.0

        # ★★★ Put check code here ★★★
        if it % 20 == 0:
            # At this point v_body_seq is fully computed
            print(f"\n[DIRECTION CHECK] Iter {it}: Real_v_body_x={v_body_seq[0,0,0].item():+.3f}, Target_v_star={vref_body[0,0,0].item():+.3f}")
            print(f"                  World_v_x={v_world_seq[0,0,0].item():+.3f}")


        if pz_hist:
            pz_seq = torch.stack(pz_hist)  # (T,B)
            loss_h = (pz_seq - cfg.h0).abs().mean()
        else:
            loss_h = torch.tensor(0.0, device=device)

        if omega_hist:
            omega_seq = torch.stack(omega_hist)  # (T,B,3)

            # Roll/pitch regularization: want roll/pitch angular velocity not too large
            rollpitch_sq = (omega_seq[..., :2] ** 2).sum(dim=-1)   # (T,B)

            # ★ Yaw angular velocity tracking: omega_z vs yaw_cmd
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
            #g_xy = gproj_seq[..., :2]
            #loss_gproj = (g_xy ** 2).sum(dim=-1).mean()
            g_xy = gproj_seq[..., :2]
            # Normalization: convert unit from m/s^2 to dimensionless, avoid this term being naturally an order of magnitude larger than others
            g_xy_norm = g_xy / cfg.g   # cfg.g is typically 9.81
            loss_gproj = (g_xy_norm ** 2).sum(dim=-1).mean()
        else:
            loss_gproj = torch.tensor(0.0, device=device)

        if foot_ref_hist:
            foot_err_seq = torch.stack(foot_ref_hist)  # (T,B,4,3)
            loss_foot = (foot_err_seq ** 2).sum(dim=-1).mean()
        else:
            loss_foot = torch.tensor(0.0, device=device)

        yaw_w = 0.1
        loss = (a1*loss_v +
                a2*loss_h +
                a3*loss_omega +
                a4*loss_ctrl +
                a5*loss_gproj +
                a6*loss_foot  +
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

        # If first layer has no gradient, it means the connection from physics model (SRBD) to Policy is broken
        if grad_dict['net.0.weight'] is not None and grad_dict['net.0.weight'] < 1e-9:
            print("⚠️ Warning: Gradient almost 0! Physics model gradient failed to propagate back to neural network.")
        #------------------------------------------------


        nn.utils.clip_grad_norm_(model.parameters(), 0.3)  # Stricter gradient clipping
        opt.step()


        # Detach SRBD state
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

        vx_iter_track.append(vx_for_plot)
        losses.append(loss.item()); rewards.append(episodic_reward)
        pbar.set_description(f"Iter {it:4d} | loss {loss.item():.3f} | v_body_x {vx_for_plot:+.2f}")


    # ===== Save curves =====
    os.makedirs(RESULTS_DIR, exist_ok=True)
    def out(fn): return os.path.join(RESULTS_DIR, fn)

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

    wrapper = PolicyActOnly(model).to(device)

    example_obs = torch.zeros(1, 36, device=device)  # Your DiffSim obs_dim=36
    traced = torch.jit.trace(wrapper, example_obs)
    traced.save(out("quad_diffsim_srbd_align_multi_robot.pt"))
    print(f"✅ Saved TorchScript: {out('quad_diffsim_srbd_align_multi_robot.pt')}")
    # ===== Additional TorchScript export (for ROS2 deployment) =====

    print("✅ Training done (MULTI robot SRBD + α-align, Eq.(5) loss, body-frame vx tracking).")
    
if __name__ == "__main__":
    train(num_iters=1000, steps_per_iter=24, seed=0)

