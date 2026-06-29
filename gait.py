import math

# ⚠️ CRITICAL: Isaac Gym must be imported BEFORE torch
try:
    from isaacgym import gymapi
except Exception:
    pass

import torch


class GaitPlanner:
    def __init__(self, env):
        object.__setattr__(self, 'env', env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        if name == 'env':
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def _update_gait_from_cmd(self):
        """
        Fixed stride length + frequency follows velocity command (batch B version)
        - Use velocity magnitude of cmd_rand[:,0:2] as |v_cmd|
        - Set a fixed hip->landing distance L_land (m) for each gait
        - Derive f from Raibert formula delta = |v|/(4f) => f = |v|/(4*L_land)
        - Clamp to [step_freq_min, step_freq_max], and apply deadzone & rough terrain scaling
        """
        if not getattr(self.cfg, "step_freq_from_cmd", False):
            return

        dev = self.device
        B = self.B
        cfg = self.cfg

        # -------- Velocity command (you already maintain cmd_rand / vx_star) --------
        if not hasattr(self, "cmd_rand"):
            # Fallback: only use vx_star
            v_cmd_xy = torch.stack([self.vx_star, torch.zeros_like(self.vx_star)], dim=-1)
        else:
            v_cmd_xy = self.cmd_rand[:, 0:2]  # (B,2) in body frame (your convention)
        v_mag = torch.linalg.norm(v_cmd_xy, dim=1)  # (B,)
        v_hi = max(0.20, float(getattr(cfg, "vx_max", 0.5)))  # For normalization

        # -------- gait_id (B,) --------
        gait_ids = getattr(self, "gait_ids", torch.ones(B, dtype=torch.long, device=dev)).to(dev)

        # -------- deadzone --------
        dead = float(getattr(cfg, "cmd_deadzone", 0.05))  # m/s
        yaw_dead = float(getattr(cfg, "yaw_deadzone", dead))

        if hasattr(self, "cmd_rand"):
            v_mag = torch.linalg.norm(self.cmd_rand[:, 0:2], dim=1)   # (B,)
            yaw_mag = torch.abs(self.cmd_rand[:, 2])                  # (B,)
        else:
            v_mag = torch.abs(self.vx_star)
            yaw_mag = torch.zeros_like(v_mag)

        # As long as “velocity or turning” is significant, consider it needs stepping/advancing phase
        self.move_mask_B = ((v_mag >= dead) | (yaw_mag >= yaw_dead)).float()   # (B,)

        # -------- Fixed stride table: L_land here is “hip to landing point” forward distance (m) --------
        # You can adjust based on your feel; these values around 0.06~0.09m work well (corresponding to stride ~2*L_land magnitude)
        L_table = torch.tensor(
            #[0.0, 0.08, 0.065, 0.055, 0.085],   # Already tuned version
            [0.0, 0.08, 0.065, 0.055, 0.085],   #0.12
            dtype=torch.float32, device=dev
        )
        L_land = L_table[gait_ids].clamp(min=1e-3)  # (B,)

        # -------- rough terrain scaling (conservative: smaller step, slightly lower freq, higher lift) --------
        rough = 0.0 if getattr(cfg, "terrain_type", "flat") == "flat" else 1.0
        L_land = L_land * (1.0 - 0.10 * rough)               # Stride shorter
        freq_scale = (1.0 - 0.15 * rough)                    # Frequency more conservative (lower)
        #height_scale = (1.0 + 0.40 * rough)                  # Lift higher
        height_scale = 1.0                 # Lift higher

        # -------- Derive f from fixed stride: delta ≈ |v|/(4f)  =>  f ≈ |v|/(4*L_land) --------
        f_raw = (v_mag / (4.0 * L_land + 1e-6)) * freq_scale

        fmin = float(cfg.step_freq_min)
        fmax = float(cfg.step_freq_max)
        f = torch.clamp(f_raw, fmin, fmax)

        # deadzone: at low speed don't force “fixed stride”, just give minimum frequency + very small lift
        f = torch.where(v_mag < dead, torch.full((B,), fmin, device=dev), f)
        self.step_freq_B = f

        # -------- swing height: rises with speed + rough gain + deadzone reduction --------
        h0 = float(cfg.swing_height)
        #h_max = float(getattr(cfg, “swing_height_max”, 0.11))
        #h_max = float(getattr(cfg, “swing_height_max”, 0.015))

        # ✅ Fix: ensure h_max >= h0, and give reasonable default (0.05)
        h_max = float(getattr(cfg, "swing_height_max", 0.05))
        h_max = max(h_max, h0)
        s = torch.clamp(v_mag / (v_hi + 1e-6), 0.0, 1.0)
        h = (h0 + (h_max - h0) * s) * height_scale
        h = torch.where(v_mag < dead, torch.full((B,), 0.5 * h0, device=dev), h)
        self.swing_height_B = h


    @torch.no_grad()

    def _swing_parabola(self, p0, pm, p1, s):
        c = p0
        b = 4*(pm - (p0 + p1)/2.0)
        a = p1 - p0 - b
        return a*(s**2) + b*s + c

    # ---------------- swing trajectory (currently used version)----------------
    # Calculate foot target position based on velocity command and gait phase - equivalent to main function for foot trajectory generation

    def _update_foot_targets_from_command(self, phases, p_foot_now, return_vref: bool = False):
        """
        Strict Fig 9.2/9.3: stance/swing defined by duty factor β; running/bound/gallop allow aerial phase
        """
        dev, cfg, B = self.device, self.cfg, self.B

        # ---------- 0) contact (only used for last_contact_z etc; mask no longer mixes with contact by default) ----------
        contact = self.contact_flags(thresh=cfg.contact_thresh_n)  # (B,4,1)

        # ---------- 1) Get duty factor β & min_feet from gait ----------
        beta_B, min_feet_B, allow_aerial = self._get_beta_minfeet_allow_aerial()

        # ---------- 2) phase-based stance_mask (strict definition) ----------
        stance_mask = self._mix_stance(
            phases=phases,
            contact_flags=contact,
            beta_B=beta_B,
            min_feet_B=min_feet_B,
            w_phase=1.0,
            w_contact=0.0
        )  # (B,4,1)
        swing_mask = 1.0 - stance_mask

        # ---------- 2.5) Record liftoff moment start point (x0,y0,z0) ----------
        # liftoff: previous frame is stance, this frame becomes swing
        if not hasattr(self, "prev_stance_mask"):
            self.prev_stance_mask = torch.ones_like(stance_mask)
        # Important: when rolling out multiple steps then backward, any “state cache” must detach + clone,
        # and avoid in-place writes, otherwise autograd will report version mismatch
        with torch.no_grad():
            liftoff = (self.prev_stance_mask > 0.5) & (stance_mask < 0.5)          # (B,4,1)
            liftoff3 = liftoff.expand(-1, -1, 3)                                   # (B,4,3)
            self.last_liftoff_xyz = torch.where(
                liftoff3,
                p_foot_now.detach(),                                               # ★ detach
                self.last_liftoff_xyz
            ).clone()                                                               # ★ clone (break storage alias)


        # ---------- 3) Get high-level command (vx, vy, yaw_rate) ----------
        if hasattr(self, "cmd_rand"):
            v_cmd_body_xy = self.cmd_rand[:, 0:2]   # (B,2)
            yaw_rate_cmd  = self.cmd_rand[:, 2]     # (B,)
        else:
            v_cmd_body_xy = torch.stack([self.vx_star, torch.zeros_like(self.vx_star)], dim=-1)
            yaw_rate_cmd  = torch.zeros(B, device=dev)

        # ---------- stop: force all feet stance ----------
        dead = float(getattr(cfg, "cmd_deadzone", 0.05))
        v_mag = torch.linalg.norm(v_cmd_body_xy, dim=1)          # (B,)
        stop_env = (v_mag < dead)                                # (B,)
        stop3 = stop_env.view(B, 1, 1)
        v_cmd_body_xy = torch.where(stop_env[:, None], torch.zeros_like(v_cmd_body_xy), v_cmd_body_xy)
        yaw_rate_cmd  = torch.where(stop_env, torch.zeros_like(yaw_rate_cmd), yaw_rate_cmd)

        stance_mask = torch.where(stop3, torch.ones_like(stance_mask), stance_mask)
        swing_mask  = 1.0 - stance_mask

        # ---------- 4) body->world yaw rotation ----------
        yaw = self.yaw
        cy = torch.cos(yaw); sy = torch.sin(yaw)
        R_yaw = torch.stack(
            [torch.stack([cy, -sy], dim=-1),
             torch.stack([sy,  cy], dim=-1)], dim=1
        )  # (B,2,2)

        v_cmd_world_xy = torch.einsum("bij,bj->bi", R_yaw, v_cmd_body_xy)  # (B,2)

        # ---------- 5) STANCE branch (per Fig 9.2: stance foot stays fixed in world frame) ----------
        # Lock directly to the most recent “touchdown moment” foot position (world frame)
        # This way PD target won't drag the stance foot “backward”, significantly reducing slip/forward tilt
        p_stance = p_foot_now.clone()
        p_stance[..., 0:2] = self.last_contact_xy
        p_stance[..., 2]   = self.last_contact_z


        # ---------- 6) SWING branch: quadratic parabola ----------
        dt = cfg.dt
        step_freq = getattr(self, "step_freq_B", torch.full((B,), cfg.step_freq, device=dev))
        T = 1.0 / step_freq

        _, p_land_xy_world = self._raibert_touchdown_world(phases)   # (B,4,2)

        u = self._phase_u(phases)                                    # (B,4)
        beta4 = beta_B.view(B, 1).expand(B, 4)
        den = (1.0 - beta4).clamp_min(1e-6)
        raw = ((u - beta4) / den).clamp(0.0, 1.0)
        is_swing = (u >= beta4)
        swing_phase = torch.where(is_swing, raw, torch.zeros_like(u))  # (B,4)
        s = swing_phase.unsqueeze(-1)                                   # (B,4,1)

        # Start point p0 (world)
        p0 = self.last_liftoff_xyz.detach().clone()                     # (B,4,3)

        # End point p1 (world)
        p1 = torch.zeros_like(p0)
        p1[..., 0:2] = p_land_xy_world
        p1[..., 2]   = self.last_contact_z

        # Mid point pm (world)
        h_env = getattr(
            self,
            "swing_height_B",
            torch.full((B,), cfg.swing_height, device=dev)
        )                                                               # (B,)
        h_leg = h_env.view(B, 1).expand(-1, 4)                          # (B,4)

        pm = 0.5 * (p0 + p1)
        pm[..., 2] = 0.5 * (p0[..., 2] + p1[..., 2]) + h_leg

        # Quadratic parabola
        p_swing = self._swing_parabola(p0, pm, p1, s)                  # (B,4,3)

        v_foot_ref_world = torch.zeros_like(p_foot_now)
        # ---------- 7) Blend ----------
        p_foot_target = stance_mask * p_stance + (1.0 - stance_mask) * p_swing

        # stop env locks foot position
        p_foot_target = torch.where(stop3, p_foot_now, p_foot_target)
        stance_mask   = torch.where(stop3, torch.ones_like(stance_mask), stance_mask)

        # ---------- 8) Update prev_stance_mask (for next frame liftoff detection) ----------
        with torch.no_grad():
            self.prev_stance_mask = stance_mask.detach().clone()              # ★ clone to avoid alias/version issues

        if return_vref:
            return p_foot_target, stance_mask, v_foot_ref_world
        return p_foot_target, stance_mask

    # Foothold calculation (body frame Raibert version) - aligned with textbook, used in _update_foot_targets_from_command: currently used foothold formula

    def _raibert_touchdown_world(self, phases: torch.Tensor):
        """
        Strictly aligned with textbook/Fig 9.6/9.7 touchdown:
            p_land = p_hip(p) + v_now*(1-p)*T_swing + 0.5*T_stance*v_des + k*(v_now - v_des) + x_bias*fwd
        - p (=swing_phase) is each leg's own swing progress
        - v_now uses current estimated velocity (world), v_des comes from cmd (body->world)
        Returns:
            p_hip_xy_world  : (B,4,2)
            p_land_xy_world : (B,4,2)
        """
        dev, cfg, B = self.device, self.cfg, self.B

        beta_B, _, _ = self._get_beta_minfeet_allow_aerial()   # (B,)
        step_freq = getattr(self, "step_freq_B", torch.full((B,), cfg.step_freq, device=dev))
        T = 1.0 / step_freq                                    # (B,)
        T_stance = beta_B * T                                   # (B,)
        T_swing  = (1.0 - beta_B) * T                            # (B,)

        # u in [0,1), p in [0,1]
        u = self._phase_u(phases)                                # (B,4)
        beta4 = beta_B.view(B, 1).expand(B, 4)                   # (B,4)
        den = (1.0 - beta4).clamp_min(1e-6)
        p = ((u - beta4) / den).clamp(0.0, 1.0)                  # (B,4)
        T_left = ((1.0 - p) * T_swing.view(B, 1)).clamp_min(1e-3) # (B,4)

        # yaw -> body->world
        yaw = self.yaw                                           # (B,)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R_yaw = torch.stack(
            [torch.stack([cy, -sy], dim=-1),
             torch.stack([sy,  cy], dim=-1)], dim=1
        )                                                        # (B,2,2)

        # hip offsets in body frame
        hip_offsets_body = torch.tensor([
            [ +0.1934, +0.1420 ],
            [ +0.1934, -0.1420 ],
            [ -0.1934, +0.1420 ],
            [ -0.1934, -0.1420 ],
        ], dtype=torch.float32, device=dev).view(1,4,2).expand(B,4,2)

        base_xy = self.base_pos[:, 0:2]                          # (B,2)

        p_hip_xy_world = base_xy.view(B,1,2) + torch.matmul(hip_offsets_body, R_yaw.transpose(1,2))  # (B,4,2)


        # v_des: cmd in body -> world
        if hasattr(self, "cmd_rand"):
            v_des_body = self.cmd_rand[:, 0:2]                   # (B,2)
        else:
            v_des_body = torch.stack([self.vx_star, torch.zeros_like(self.vx_star)], dim=-1)
        v_des_world = torch.einsum("bij,bj->bi", R_yaw, v_des_body)  # (B,2)

        # v_now: current estimated base velocity (world)
        v_now_world = self.base_lin_world[:, 0:2]                # (B,2)

        term_predict = v_now_world.view(B,1,2) * T_left.unsqueeze(-1)                 # v*(1-p)T_swing

        #term_ff      = 0.5 * v_des_world.view(B,1,2) * T_stance.view(B,1,1)           # 0.5*T_stance*v_des
        term_ff      = 0.5 * v_des_world.view(B,1,2) * T_stance.view(B,1,1)

        # Feedback term: when v_now < v_des, foothold should move forward (positive direction) to generate more thrust
        # So use (v_des - v_now), this way when speed is insufficient term_fb is positive
        term_fb      = cfg.k_raibert * (v_des_world - v_now_world).view(B,1,2)        # +k*(v_des - v_now)
        fb_clip = float(getattr(cfg, "raibert_fb_clip", 0.15))
        term_fb = term_fb.clamp(min=-fb_clip, max=+fb_clip)

        # x_bias along body forward projected to world
        fwd_world = torch.stack([cy, sy], dim=1)                 # (B,2)
        term_bias = cfg.x_bias * fwd_world.view(B,1,2)
        #p_land_xy_world = p_hip_xy_world + term_predict + term_ff + term_fb + term_bias
        p_land_xy_world = p_hip_xy_world  + term_ff 
        return p_hip_xy_world, p_land_xy_world


    # ---------------- stance helpers ----------------

    def _phase_u(self, phases: torch.Tensor) -> torch.Tensor:
        """
        phases: (B,4) rad
        return u in [0,1): (B,4)
        """
        return torch.remainder(phases, 2 * math.pi) / (2 * math.pi)
    

    def _get_beta_minfeet_allow_aerial(self):
        """
        Per Fig 9.2/9.3: give duty factor β=r for each gait, and decide whether to allow aerial phase (min_feet=0)
        Returns:
        beta_B      : (B,) in (0,1]
        min_feet_B  : (B,) int64, 0/2/4
        allow_aerial: (B,) bool
        """
        dev, B = self.device, self.B
        cfg = self.cfg

        # Your existing gait_ids: 0 stand, 1 trot, 2 pace, 3 bound, 4 gallop
        gait_ids = getattr(self, "gait_ids", torch.ones(B, dtype=torch.long, device=dev)).to(dev)

        # ---- Default duty factors (can adjust per your textbook/paper) ----
        beta_stand  = 1.0
        beta_trot_n = 0.5   # Fig 9.2a
        beta_trot_w = 0.6   # Fig 9.2b: r>0.5 (walking)
        beta_trot_r = 0.4   # Fig 9.2c: r<0.5 (running -> has aerial)
        beta_pace   = 0.5   # Fig 9.3b text gives r=0.5
        beta_bound  = 0.4   # Fig 9.3a text gives r<0.5 (has aerial)
        beta_gallop = 0.35  # Textbook figure doesn't give specific value, commonly use smaller duty for obvious aerial
        # You can force which trot variant via cfg.trot_style: “normal” / “walk” / “run”
        trot_style = getattr(cfg, "trot_style", "normal")
        if trot_style not in ("normal", "walk", "run"):
            trot_style = "normal"

        beta_trot = {"normal": beta_trot_n, "walk": beta_trot_w, "run": beta_trot_r}[trot_style]

        # ---- Assemble beta table (by gait_id) ----
        beta_table = torch.tensor(
            [beta_stand, beta_trot, beta_pace, beta_bound, beta_gallop],
            dtype=torch.float32, device=dev
        )
        beta_B = beta_table[gait_ids].clamp(min=1e-3, max=1.0)  # (B,)

        # ---- min_feet: strictly decide by “whether to allow aerial phase” ----
        # walking/normal trot, pace: at least 2 feet support at any moment
        # stand: 4 feet
        # bound/gallop, running trot: allow aerial => min_feet=0
        allow_aerial = (gait_ids == 3) | (gait_ids == 4)  # bound/gallop
        # trot aerial decided by beta<0.5 (running trot)
        allow_aerial = allow_aerial | ((gait_ids == 1) & (beta_B < 0.5))

        # Training warm-up: disable aerial phase first, avoid initial “free fall + forward flip”
        if getattr(cfg, "train_no_aerial", False):
            allow_aerial = torch.zeros_like(allow_aerial, dtype=torch.bool)
            # ✅ Key fix: when disabling aerial, force duty >= 0.5
            # Otherwise bound/gallop's β<0.5 causes “nominal stance less than 2 feet”,
            # _mix_stance will use topk to force support feet -> fake support feet lock last_contact -> easy forward flip
            beta_B = torch.clamp(beta_B, min=0.5, max=1.0)

        min_feet_B = torch.full((B,), 2, dtype=torch.long, device=dev)
        min_feet_B = torch.where(gait_ids == 0, torch.full_like(min_feet_B, 4), min_feet_B)
        min_feet_B = torch.where(allow_aerial, torch.zeros_like(min_feet_B), min_feet_B)
        return beta_B, min_feet_B, allow_aerial

    # Calculate stance_mask based on phase & β (strict version): determine if current leg is stance or swing

    def _stance_phase_mask(self, phases: torch.Tensor, beta_B: torch.Tensor) -> torch.Tensor:
        """
        Strict duty-factor definition:
          u in [0,1), stance if u < β, swing otherwise
        phases: (B,4)
        beta_B: (B,)
        return: (B,4,1)
        """
        B = self.B
        u = self._phase_u(phases)                       # (B,4)
        beta = beta_B.view(B, 1)                        # (B,1)
        return (u < beta).float().unsqueeze(-1)         # (B,4,1)
        
    


    def _mix_stance(self,
                phases: torch.Tensor,
                contact_flags: torch.Tensor,
                beta_B: torch.Tensor,
                min_feet_B: torch.Tensor,
                w_phase: float = 1.0,
                w_contact: float = 0.0):
        """
        phases       : (B,4)
        contact_flags: (B,4,1)  (can be passed in, but in strict mode default w_contact=0)
        beta_B       : (B,)
        min_feet_B   : (B,) long, 0/2/4

        Returns stance_mask: (B,4,1)
        """
        dev, B = self.device, self.B

        phase_mask = self._stance_phase_mask(phases, beta_B)        # (B,4,1)

        # strict: w_contact=0, w_phase=1 => purely by figure definition
        mix = w_phase * phase_mask + w_contact * contact_flags
        mix = torch.clamp(mix, 0.0, 1.0)                            # (B,4,1)

        flat = mix.view(B, 4)
        # Per-env enforce minimum support feet count, vectorized (running/bound/
        # gallop have min_feet_B=0 => never forced => aerial phase preserved).
        # For envs with fewer than k stance feet, force the top-k legs to stance.
        # A stable descending sort breaks ties toward the lower leg index (FL<FR<
        # RL<RR), matching the per-env torch.topk fallback this replaces.
        min_feet_f = min_feet_B.to(flat.dtype)                          # (B,)
        needs = flat.sum(dim=1) < min_feet_f                            # (B,) ; k=0 -> never
        _, order = torch.sort(flat, dim=1, descending=True, stable=True)  # (B,4)
        rank = torch.zeros_like(order)
        rank.scatter_(1, order, torch.arange(4, device=dev).view(1, 4).expand(B, 4))
        forced = (rank < min_feet_B.view(B, 1)).to(flat.dtype)         # (B,4) one-hot top-k
        flat = torch.where(needs.view(B, 1), forced, flat)             # (B,4)
        return flat.view(B, 4, 1)

    # ---------------- PD -> foot forces ----------------

