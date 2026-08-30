"""Gait planning: phase-based stance/swing scheduling and foot targets.

``GaitPlanner`` turns the per-leg gait phase into stance/swing masks (duty
factor per gait), plans swing-foot trajectories (quadratic parabola between
liftoff and a Raibert-style touchdown point), and derives per-env step
frequency / swing height from the velocity command. The stance/swing and
touchdown formulations follow the standard legged-locomotion textbook
treatment (duty factor, Raibert heuristic).
"""

import math

# NOTE: Isaac Gym must be imported before torch (hard requirement of isaacgym).
try:
    from isaacgym import gymapi
except Exception:
    pass

import torch

from utils_math import swing_apex_z, swing_chord_points


class GaitPlanner:
    """Gait scheduling and foot-target generation for a batch of robots.

    Like :class:`srbd.SRBDModel`, this class holds no state of its own: it
    proxies attribute access onto the wrapped ``env`` so all gait state
    (``phase``, ``step_freq_B``, ``last_contact_*`` caches, ...) lives in
    ``RealQuadEnv``'s namespace. It is a mixin split into its own file.
    """

    def __init__(self, env):
        """Wrap the environment whose state this planner reads and writes."""
        object.__setattr__(self, 'env', env)

    def __getattr__(self, name):
        """Resolve unknown attributes on the wrapped env (shared state namespace)."""
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        """Write attributes onto the wrapped env (except the ``env`` reference itself)."""
        if name == 'env':
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def _update_gait_from_cmd(self):
        """Derive per-env step frequency and swing height from the velocity command.

        Only active when ``cfg.step_freq_from_cmd`` is set (fixed-stride mode):
        each gait gets a nominal hip-to-landing distance ``L_land``, and the
        Raibert relation ``delta = |v| / (4 f)`` is inverted to give
        ``f = |v| / (4 L_land)``, clamped to ``[step_freq_min, step_freq_max]``.
        Also maintains ``move_mask_B`` (whether the phase should advance at all,
        based on the command deadzone) and a speed-scaled swing height.
        """
        if not getattr(self.cfg, "step_freq_from_cmd", False):
            return

        dev = self.device
        B = self.B
        cfg = self.cfg

        v_hi = max(0.20, float(getattr(cfg, "vx_max", 0.5)))  # normalisation ceiling

        gait_ids = getattr(self, "gait_ids", torch.ones(B, dtype=torch.long, device=dev)).to(dev)

        # Command magnitudes and deadzone: a significant linear OR yaw command
        # means the robot should be stepping (phase advances).
        dead = float(getattr(cfg, "cmd_deadzone", 0.05))  # m/s
        yaw_dead = float(getattr(cfg, "yaw_deadzone", dead))

        if hasattr(self, "cmd_rand"):
            v_mag = torch.linalg.norm(self.cmd_rand[:, 0:2], dim=1)   # (B,)
            yaw_mag = torch.abs(self.cmd_rand[:, 2])                  # (B,)
        else:
            v_mag = torch.abs(self.vx_star)
            yaw_mag = torch.zeros_like(v_mag)

        self.move_mask_B = ((v_mag >= dead) | (yaw_mag >= yaw_dead)).float()   # (B,)

        # Per-gait nominal hip-to-landing distance (m); stride is ~2 * L_land.
        L_table = torch.tensor(
            [0.0, 0.08, 0.065, 0.055, 0.085],
            dtype=torch.float32, device=dev
        )
        L_land = L_table[gait_ids].clamp(min=1e-3)  # (B,)

        # Rough-terrain scaling: shorter stride, more conservative frequency.
        rough = 0.0 if getattr(cfg, "terrain_type", "flat") == "flat" else 1.0
        L_land = L_land * (1.0 - 0.10 * rough)
        freq_scale = (1.0 - 0.15 * rough)
        height_scale = 1.0

        # Invert the Raibert stride relation: delta = |v|/(4f)  =>  f = |v|/(4 L_land).
        f_raw = (v_mag / (4.0 * L_land + 1e-6)) * freq_scale

        fmin = float(cfg.step_freq_min)
        fmax = float(cfg.step_freq_max)
        f = torch.clamp(f_raw, fmin, fmax)

        # Inside the deadzone don't force a stride: minimum frequency, small lift.
        f = torch.where(v_mag < dead, torch.full((B,), fmin, device=dev), f)
        self.step_freq_B = f

        # Swing height rises with speed (h_max is clamped to at least h0 so the
        # interpolation below can never lower the lift under the configured default).
        h0 = float(cfg.swing_height)
        h_max = float(getattr(cfg, "swing_height_max", 0.05))
        h_max = max(h_max, h0)
        s = torch.clamp(v_mag / (v_hi + 1e-6), 0.0, 1.0)
        h = (h0 + (h_max - h0) * s) * height_scale
        h = torch.where(v_mag < dead, torch.full((B,), 0.5 * h0, device=dev), h)
        self.swing_height_B = h

    def _swing_parabola(self, p0, pm, p1, s):
        """Quadratic interpolation through (p0 at s=0, pm at s=0.5, p1 at s=1).

        Used for the swing-foot trajectory: start at the liftoff point, apex at
        the midpoint raised by the swing height, land at the touchdown target.
        All arguments broadcast; ``s`` is the swing progress in [0, 1].

        Deliberately **not** ``@torch.no_grad()`` (it was, before Step 3). The
        foothold residual reaches the training objective only through ``p1``, so
        under the old decorator the correction arrived as a constant and its
        policy outputs received no gradient at all. Removing it costs nothing
        when nothing upstream carries a graph: with the Step 3 flags off, ``p0``
        is detached, ``p1`` comes from ``last_contact_z`` and the Isaac-side
        Raibert point, and the arithmetic -- hence every value -- is unchanged.

        Worth knowing before tuning: ``pm`` is displaced from the chord midpoint
        only in z, so ``b_x = 0``, ``p_swing_x = (p1_x - p0_x)*s^2 + p0_x`` and
        ``d p_swing_x / d p1_x = s^2``. The residual has no authority at liftoff
        and full authority at touchdown, which is the profile it should have.
        """
        c = p0
        b = 4*(pm - (p0 + p1)/2.0)
        a = p1 - p0 - b
        return a*(s**2) + b*s + c

    def _update_foot_targets_from_command(self, phases, p_foot_now, return_vref: bool = False,
                                          foothold_res=None):
        """Compute per-foot PD targets and the stance mask for the current phase.

        The main foot-trajectory entry point, called once per control step.
        Stance/swing is defined strictly by the duty factor (running gaits may
        have an aerial phase): stance feet are locked to their last touchdown
        position in the world frame; swing feet follow a quadratic parabola
        from the liftoff point to the Raibert touchdown target.

        Args:
            phases: (B, 4) absolute leg phases in radians.
            p_foot_now: (B, 4, 3) current world-frame foot positions.
            return_vref: Also return the (currently zero) foot reference
                velocity, matching the training loop's call signature.
            foothold_res: (B, 4, 2) per-leg foothold correction in the **body**
                frame, metres, or None (Step 3). Rotated into the world frame by
                the base yaw alongside the Raibert feed-forward term, so it means
                "shift this foothold forward/back relative to where the robot is
                facing" rather than a compass direction.

        Returns:
            (p_foot_target (B,4,3), stance_mask (B,4,1)[, vref (B,4,3)]).

        Also publishes ``self.foothold_plan_xy`` (B,4,2), the planned foothold in
        world coordinates. Unlike ``self.swing_progress`` this one is **not**
        detached: it is the gradient path the foothold-quality loss samples the
        terrain along. It is the plan the returned target was built from, so a
        caller reading it gets exactly what the swing is aiming at.
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

        # ---------- 2.5) Record the liftoff start point (x0, y0, z0) ----------
        # liftoff edge: previous frame stance, this frame swing.
        if not hasattr(self, "prev_stance_mask"):
            self.prev_stance_mask = torch.ones_like(stance_mask)
        # When rolling out multiple steps before backward(), every state cache
        # must be detached and cloned (no in-place writes into aliased storage),
        # otherwise autograd reports a version mismatch.
        with torch.no_grad():
            liftoff = (self.prev_stance_mask > 0.5) & (stance_mask < 0.5)          # (B,4,1)
            liftoff3 = liftoff.expand(-1, -1, 3)                                   # (B,4,3)
            self.last_liftoff_xyz = torch.where(
                liftoff3,
                p_foot_now.detach(),
                self.last_liftoff_xyz
            ).clone()  # clone breaks the storage alias


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

        # ---------- 4) STANCE branch: stance foot stays fixed in the world frame ----------
        # Lock to the most recent touchdown position so the PD target does not
        # drag the stance foot backward (reduces slip and forward tilt).
        p_stance = p_foot_now.clone()
        p_stance[..., 0:2] = self.last_contact_xy
        p_stance[..., 2]   = self.last_contact_z

        # ---------- 5) SWING branch: quadratic parabola ----------
        # Step 3: the planned foothold, Raibert plus the policy's per-leg correction. Published
        # attached, because it is the gradient path the foothold-quality loss samples along.
        _, self.foothold_plan_xy = self._raibert_touchdown_world(phases, foothold_res=foothold_res)

        # The *target* the swing (and so loss_foot) is built from drops that graph by default:
        # with it attached, loss_foot has a degenerate minimum -- drag the target onto the foot
        # instead of moving the foot. See STEP3_FOOTHOLD.md 4.3.
        #
        # Detaching the sum is exactly equivalent to summing with a detached residual, because
        # the Raibert term itself is already grad-free (it is built from Isaac's base_pos and
        # yaw). Same floats either way, so this is one Raibert evaluation per step rather than
        # two, and there is no second code path to keep in step with the first.
        p_land_xy_world = self.foothold_plan_xy
        if foothold_res is not None and getattr(cfg, "foot_res_detach", True):
            p_land_xy_world = p_land_xy_world.detach()

        u = self._phase_u(phases)                                    # (B,4)
        beta4 = beta_B.view(B, 1).expand(B, 4)
        den = (1.0 - beta4).clamp_min(1e-6)
        raw = ((u - beta4) / den).clamp(0.0, 1.0)
        is_swing = (u >= beta4)
        swing_phase = torch.where(is_swing, raw, torch.zeros_like(u))  # (B,4)
        s = swing_phase.unsqueeze(-1)                                   # (B,4,1)
        # Published for the terrain-clearance loss, which modulates its required margin by the
        # same swing profile this parabola uses (see train.py). Derived from `phases`, which
        # carries no gradient, so caching it costs nothing and detaching changes nothing.
        self.swing_progress = swing_phase.detach()                      # (B,4)

        # Start point p0 (world)
        p0 = self.last_liftoff_xyz.detach().clone()                     # (B,4,3)

        # End point p1 (world). Landing height: `last_contact_z` is the height this leg last
        # touched down at, which is the correct proprioceptive estimate for a blind robot and
        # stale by exactly one riser on every stair step (CAMPAIGN_FINDINGS.md 22.6). Under
        # cfg.foot_z_terrain it becomes the actual ground under the landing point.
        #
        # smooth=False on purpose: this is a target *value*, and the blurred field is a 0.2 m
        # Gaussian against a 0.31 m tread, which would aim the foot halfway between tread and
        # riser near every edge. The blurred field is for gradients (loss_clear), not targets.
        if getattr(cfg, "foot_z_terrain", False):
            land_z = self.env.terrain_height_legs(p_land_xy_world, smooth=False)   # (B,4)
        else:
            land_z = self.last_contact_z                                           # (B,4)
        # torch.cat rather than zeros_like + in-place slice writes: p_land_xy_world carries a
        # graph once the foothold residual is on, and in-place writes into an aliased tensor
        # are what the liftoff cache above already had to work around.
        p1 = torch.cat([p_land_xy_world, land_z.unsqueeze(-1)], dim=-1)  # (B,4,3)

        # Mid point pm (world)
        h_env = getattr(
            self,
            "swing_height_B",
            torch.full((B,), cfg.swing_height, device=dev)
        )                                                               # (B,)
        h_leg = h_env.view(B, 1).expand(-1, 4)                          # (B,4)

        # Apex: the chord midpoint raised by the swing height, except that on a rising step the
        # ground between liftoff and touchdown sits above both endpoints and the parabola's s^2
        # base sags into the riser. utils_math.swing_apex_z keeps the old midpoint as a lower
        # bound, so this can only raise the apex and is bit-identical on flat ground.
        if getattr(cfg, "foot_apex_terrain", False):
            chord_xy = swing_chord_points(p0[..., 0:2], p1[..., 0:2],
                                          int(getattr(cfg, "foot_apex_n", 5)))      # (B,4,n,2)
            chord_z = self.env.terrain_height_legs(chord_xy, smooth=False)           # (B,4,n)
            pm_z = swing_apex_z(chord_z, p0[..., 2], p1[..., 2], h_leg)              # (B,4)
        else:
            pm_z = 0.5 * (p0[..., 2] + p1[..., 2]) + h_leg
        pm = torch.cat([0.5 * (p0[..., 0:2] + p1[..., 0:2]), pm_z.unsqueeze(-1)], dim=-1)

        # Quadratic parabola
        p_swing = self._swing_parabola(p0, pm, p1, s)                  # (B,4,3)

        v_foot_ref_world = torch.zeros_like(p_foot_now)
        # ---------- 6) Blend ----------
        p_foot_target = stance_mask * p_stance + (1.0 - stance_mask) * p_swing

        # stop env locks foot position
        p_foot_target = torch.where(stop3, p_foot_now, p_foot_target)
        stance_mask   = torch.where(stop3, torch.ones_like(stance_mask), stance_mask)

        # ---------- 7) Update prev_stance_mask (for next frame's liftoff detection) ----------
        with torch.no_grad():
            self.prev_stance_mask = stance_mask.detach().clone()  # clone avoids alias/version issues

        if return_vref:
            return p_foot_target, stance_mask, v_foot_ref_world
        return p_foot_target, stance_mask

    def _raibert_touchdown_world(self, phases: torch.Tensor, foothold_res=None):
        """Raibert-style touchdown targets in the world frame.

        The full textbook touchdown heuristic is

            p_land = p_hip + v_now*(1-p)*T_swing + 0.5*T_stance*v_des
                     + k*(v_des - v_now) + x_bias*fwd

        where p is each leg's swing progress, v_now the current base velocity
        (world) and v_des the commanded velocity (body -> world). The current
        implementation keeps only the feed-forward term ``0.5*T_stance*v_des``
        (the prediction, feedback and bias terms destabilised training; the
        feedback gain ``cfg.k_raibert`` defaults to 0 accordingly).

        Args:
            phases: (B, 4) absolute leg phases in radians.
            foothold_res: (B, 4, 2) body-frame per-leg correction in metres, or
                None (Step 3). Added after the same body->world yaw rotation the
                commanded velocity gets, which is the whole reason it is applied
                here rather than at the call site: R_yaw is already built.

        Returns:
            (p_hip_xy_world (B,4,2), p_land_xy_world (B,4,2)).
        """
        dev, cfg, B = self.device, self.cfg, self.B

        beta_B, _, _ = self._get_beta_minfeet_allow_aerial()   # (B,)
        step_freq = getattr(self, "step_freq_B", torch.full((B,), cfg.step_freq, device=dev))
        T = 1.0 / step_freq                                    # (B,)
        T_stance = beta_B * T                                   # (B,)

        # yaw -> body->world rotation
        yaw = self.yaw                                           # (B,)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R_yaw = torch.stack(
            [torch.stack([cy, -sy], dim=-1),
             torch.stack([sy,  cy], dim=-1)], dim=1
        )                                                        # (B,2,2)

        # hip offsets in the body frame (same geometry as srbd.foot_positions_srbd)
        hx, hy = float(cfg.hip_offset_x), float(cfg.hip_offset_y)
        hip_offsets_body = torch.tensor([
            [ +hx, +hy ],
            [ +hx, -hy ],
            [ -hx, +hy ],
            [ -hx, -hy ],
        ], dtype=torch.float32, device=dev).view(1,4,2).expand(B,4,2)

        base_xy = self.base_pos[:, 0:2]                          # (B,2)

        p_hip_xy_world = base_xy.view(B,1,2) + torch.matmul(hip_offsets_body, R_yaw.transpose(1,2))  # (B,4,2)

        # v_des: commanded velocity, body -> world
        if hasattr(self, "cmd_rand"):
            v_des_body = self.cmd_rand[:, 0:2]                   # (B,2)
        else:
            v_des_body = torch.stack([self.vx_star, torch.zeros_like(self.vx_star)], dim=-1)
        v_des_world = torch.einsum("bij,bj->bi", R_yaw, v_des_body)  # (B,2)

        # Feed-forward term only (see docstring for the full heuristic).
        term_ff = 0.5 * v_des_world.view(B,1,2) * T_stance.view(B,1,1)

        p_land_xy_world = p_hip_xy_world + term_ff

        # Step 3 foothold residual, body -> world through the same rotation as v_des.
        if foothold_res is not None:
            res_world = torch.einsum("bij,bnj->bni", R_yaw, foothold_res)   # (B,4,2)
            p_land_xy_world = p_land_xy_world + res_world

        return p_hip_xy_world, p_land_xy_world


    # ---------------- stance helpers ----------------

    def _phase_u(self, phases: torch.Tensor) -> torch.Tensor:
        """
        phases: (B,4) rad
        return u in [0,1): (B,4)
        """
        return torch.remainder(phases, 2 * math.pi) / (2 * math.pi)
    

    def _get_beta_minfeet_allow_aerial(self):
        """Per-gait duty factor, minimum stance-feet count and aerial-phase permission.

        Returns:
            beta_B: (B,) duty factor in (0, 1].
            min_feet_B: (B,) int64, minimum simultaneous stance feet (0/2/4).
            allow_aerial: (B,) bool, whether a full aerial phase is permitted.
        """
        dev, B = self.device, self.B
        cfg = self.cfg

        # gait ids: 0 stand, 1 trot, 2 pace, 3 bound, 4 gallop
        gait_ids = getattr(self, "gait_ids", torch.ones(B, dtype=torch.long, device=dev)).to(dev)

        # Duty factors per gait (textbook values; trot has walk/normal/run variants).
        beta_stand  = 1.0
        beta_trot_n = 0.5   # normal trot
        beta_trot_w = 0.6   # walking trot (duty > 0.5)
        beta_trot_r = 0.4   # running trot (duty < 0.5 -> aerial phase)
        beta_pace   = 0.5
        beta_bound  = 0.4   # duty < 0.5 -> aerial phase
        beta_gallop = 0.35  # small duty for a pronounced aerial phase
        # The trot variant is selected via cfg.trot_style: "normal" / "walk" / "run".
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

        # min_feet follows from the aerial-phase permission:
        #   walking/normal trot, pace -> at least 2 stance feet at any moment;
        #   stand -> 4 feet; bound/gallop and running trot -> aerial allowed (0).
        allow_aerial = (gait_ids == 3) | (gait_ids == 4)  # bound/gallop
        allow_aerial = allow_aerial | ((gait_ids == 1) & (beta_B < 0.5))  # running trot

        # Training warm-up: disable the aerial phase to avoid the initial
        # free-fall-and-forward-flip failure mode.
        if getattr(cfg, "train_no_aerial", False):
            allow_aerial = torch.zeros_like(allow_aerial, dtype=torch.bool)
            # With aerial disabled the duty factor must be >= 0.5: otherwise
            # bound/gallop's beta < 0.5 leaves fewer than 2 nominal stance feet,
            # _mix_stance force-fills them via top-k, and those fake stance feet
            # lock last_contact_* to bad positions (a forward-flip trigger).
            beta_B = torch.clamp(beta_B, min=0.5, max=1.0)

        min_feet_B = torch.full((B,), 2, dtype=torch.long, device=dev)
        min_feet_B = torch.where(gait_ids == 0, torch.full_like(min_feet_B, 4), min_feet_B)
        min_feet_B = torch.where(allow_aerial, torch.zeros_like(min_feet_B), min_feet_B)
        return beta_B, min_feet_B, allow_aerial

    def _stance_phase_mask(self, phases: torch.Tensor, beta_B: torch.Tensor) -> torch.Tensor:
        """Strict duty-factor stance mask: stance iff normalised phase u < beta.

        Args:
            phases: (B, 4) absolute leg phases in radians.
            beta_B: (B,) per-env duty factor.

        Returns:
            (B, 4, 1) float mask, 1 = stance, 0 = swing.
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
        """Blend the phase-based stance mask with contact flags, enforcing min feet.

        In the strict (default) configuration ``w_contact=0``, so the mask is
        purely the duty-factor definition; the contact input is kept for
        experimentation. Environments with fewer than ``min_feet_B`` stance
        feet get their top-k legs forced to stance (never triggered for
        aerial-permitted gaits, whose min_feet is 0).

        Args:
            phases: (B, 4) absolute leg phases in radians.
            contact_flags: (B, 4, 1) measured contact states.
            beta_B: (B,) duty factor.
            min_feet_B: (B,) int64 minimum stance feet (0/2/4).
            w_phase, w_contact: Blend weights for the two mask sources.

        Returns:
            (B, 4, 1) stance mask.
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
