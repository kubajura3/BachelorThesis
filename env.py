"""Isaac Gym environment for Go2 quadruped locomotion training.

``RealQuadEnv`` is the central hub of the project: it owns the Isaac Gym
simulation (robots, terrain, PD control, state tensors), the gait planner and
the differentiable SRBD model, plus the optional terrain-perception collector.
``GaitPlanner`` and ``SRBDModel`` are attribute-proxy mixins that operate
directly on this class's state (see their docstrings).

The Rudin-terrain placement and dynamic curriculum methods
(``_assign_rudin_origins``, ``_update_terrain_curriculum``) are ports of the
corresponding legged_gym logic (Rudin et al., BSD-3-Clause, ETH Zurich /
Nikita Rudin); see ``terrain.py`` for the full attribution note.
"""

# NOTE: Isaac Gym must be imported before torch (hard requirement of isaacgym).
try:
    from isaacgym import gymapi, gymtorch
    ISAAC_AVAILABLE = True
except Exception as e:
    ISAAC_AVAILABLE = False
    print("[Warning] Isaac Gym import failed:", repr(e))

import math
import os
from typing import Optional

import numpy as np
import torch

from config import (
    DBG_INIT_FALL,
    DBG_INIT_FALL_ENV,
    DBG_INIT_FALL_EVERY,
    DBG_INIT_FALL_STEPS,
    PURE_PAPER_MODE,
    EnvCfg,
)
from gait import GaitPlanner
from srbd import SRBDModel
from terrain import _setup_physx_stable, create_ground_plane, create_random_rough_terrain, create_rudin_terrain
from utils_math import (
    quat_from_rpy,
    quat_rotate_inverse_wxyz,
)


class RealQuadEnv:
    """Batched Isaac Gym environment for the Unitree Go2.

    Owns B parallel robots on one shared terrain surface, exposes the training
    interface (``reset`` / ``reset_envs`` / ``step`` / ``get_obs``), foot
    kinematics and contact sensing, and hosts the state written by the
    ``GaitPlanner`` / ``SRBDModel`` proxy mixins. Optional terrain perception
    (depth camera + height scan) is built when ``cfg.use_perception`` is set.
    """

    def __init__(self, cfg: EnvCfg, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Create the simulation, terrain, robots and all persistent buffers.

        Args:
            cfg: Environment configuration (see ``config.EnvCfg``).
            device: Torch device string; the GPU pipeline requires "cuda".
        """
        assert ISAAC_AVAILABLE, "Isaac Gym is required to run this real quadruped environment."

        self.cfg = cfg
        self.device = torch.device(device)
        self.B = int(cfg.num_envs)  # Number of parallel robots

        self._render_cnt = 0
        self.render_every = 10  # viewer refresh interval in sim steps

        # High-level velocity command: cmd_rand = [vx_cmd, vy_cmd, yaw_rate_cmd]
        self.cmd_rand = torch.zeros(self.B, 3, device=self.device)   # (B,3)
        self.vx_star  = torch.zeros(self.B, device=self.device)      # alias used by Raibert / loss

        # Rudin dynamic-curriculum bookkeeping (only used when terrain_type == "rudin" and
        # rudin_terrain.dynamic_curriculum). ep_len_buf counts env.step() calls since the last reset;
        # a robot that neither falls nor finishes resets after max_episode_length steps (= 20 s of sim
        # time at dt=0.002). init_done gates the curriculum so the very first reset doesn't promote.
        self.ep_len_buf = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.init_done = False
        self.max_episode_length_s = float(cfg.episode_length_s)
        self.max_episode_length = int(math.ceil(self.max_episode_length_s / cfg.dt))


        # thigh - upper leg; calf - lower leg
        # 12 controlled joint key names in order
        self.key12 = [
            "FL_hip","FL_thigh","FL_calf",
            "FR_hip","FR_thigh","FR_calf",
            "RL_hip","RL_thigh","RL_calf",
            "RR_hip","RR_thigh","RR_calf",
        ]

        # Default standing posture
        
        self.q_default_np = np.array([
             0.0, 0.86, -1.40,
            -0.0, 0.86, -1.40,
             0.0, 0.86, -1.40,
            -0.0, 0.86, -1.40,
        ], dtype=np.float32)


        # === Multi-gait: define phase patterns for different gaits (order: FL, FR, RL, RR) ===
        # 0: stand   - all four legs nearly synchronized
        # 1: trot    - diagonal gait (FL+RR, FR+RL)
        # 2: pace    - lateral gait (FL+RL, FR+RR)
        # 3: bound   - bounding (front legs sync, rear legs sync)
        # 4: gallop  - galloping (FL, FR, RL, RR phases incrementally)
        stand = torch.zeros(4, dtype=torch.float32, device=self.device)
        trot  = torch.tensor([0.0, math.pi, math.pi, 0.0], dtype=torch.float32, device=self.device)
        pace  = torch.tensor([0.0, math.pi, 0.0, math.pi], dtype=torch.float32, device=self.device)
        bound = torch.tensor([0.0, 0.0, math.pi, math.pi], dtype=torch.float32, device=self.device)

        # gallop: front pair and rear pair each in sync, rear lagging slightly, so the
        # sequence is "rear push-off -> airborne -> front landing" (unlike bound's 180 deg).
        gallop = torch.tensor(
            [ 0, 0.1*2* math.pi,  math.pi, 1.1 * math.pi],  # FL, FR, RL, RR
            dtype=torch.float32,
            device=self.device
        )

        self.gait_table = torch.stack([stand, trot, pace, bound, gallop], dim=0)   # (G,4)
        self.num_gaits = self.gait_table.shape[0]

        # Backward compatibility: default trot
        self.leg_phase_offsets = trot.clone()                              # (4,)

        # Read gait_mode from cfg
        self.gait_mode = getattr(cfg, "gait_mode", -1)

        # Initialize gait_id for each env
        if self.gait_mode < 0:
            # -1: each env random, actual sampling done in reset()
            self.gait_ids = torch.ones(self.B, dtype=torch.long, device=self.device)
        else:
            # Fixed: all envs use same gait
            gid = int(self.gait_mode)
            gid = max(0, min(self.num_gaits - 1, gid))    # clamp to [0, num_gaits-1]
            self.gait_ids = torch.full(
                (self.B,), gid, dtype=torch.long, device=self.device
            )

        # (B,4) current phase offsets for four legs of each env
        self.leg_phase_offsets_B = self.gait_table[self.gait_ids]


        # === Isaac Gym instance & physics parameters ===
        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.dt
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -self.cfg.g)
        sim_params.use_gpu_pipeline = self.cfg.use_gpu_pipeline
        _setup_physx_stable(sim_params, use_gpu=True)

        print("[env] creating PhysX simulation", flush=True)
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        print("[env] simulation created", flush=True)
        assert self.sim is not None, "create_sim failed"

        # ===== Terrain / ground: one shared surface, chosen by cfg switch =====
        if self.cfg.terrain_type == "rudin":
            print("[Terrain] terrain_type='rudin' -> using Rudin curriculum-grid landscape")
            self.terrain = create_rudin_terrain(self.gym, self.sim, self.cfg.rudin_terrain, self.B)
        elif self.cfg.terrain_type == "rough":
            print("[Terrain] terrain_type='rough' -> using random heightfield terrain")
            self.terrain = create_random_rough_terrain(self.gym, self.sim)
        else:
            print("[Terrain] terrain_type='flat' -> using flat ground plane")
            self.terrain = create_ground_plane(self.gym, self.sim)

        # Precompute device-side height samples (meters) for spawn-height lookup.
        # None on a flat plane -> _terrain_height() returns 0.
        if self.terrain is not None:
            self.height_samples = (
                torch.as_tensor(self.terrain.height_field_raw, device=self.device, dtype=torch.float32)
                * float(self.terrain.vertical_scale)
            )
            self._terr_h_scale = float(self.terrain.horizontal_scale)
            self._terr_x_offset = float(self.terrain.x_offset)
            self._terr_y_offset = float(self.terrain.y_offset)
        else:
            self.height_samples = None


        # Create multiple envs, one go2 per env
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.ceil(math.sqrt(self.B)))

        self.envs = []
        self.actor_handles = []
        self.actor_indices = []
        # env origins (origin of each env in SIM world coordinate system)
        self.env_origins = torch.zeros(self.B, 3, device=self.device, dtype=torch.float32)



        # Load asset (the URDF ships next to this file, so resolve it relative
        # to the repository instead of hardcoding a machine-specific path).
        ASSET_ROOT = os.path.dirname(os.path.abspath(__file__))
        ASSET_FILE = "go2_description.urdf"

        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = False
        asset_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        asset_opts.thickness = 0.01
        asset_opts.angular_damping = 0.01
        asset_opts.armature = 0.03
        for field in ("collapse_fixed_joints", "use_mesh_materials"):
            if hasattr(asset_opts, field):
                setattr(asset_opts, field, True if field=="collapse_fixed_joints" else False)
        for field in ("enable_self_collisions", "self_collisions", "use_self_collisions"):
            if hasattr(asset_opts, field):
                setattr(asset_opts, field, True)
                break
        if hasattr(asset_opts, "flip_visual_attachments"):
            asset_opts.flip_visual_attachments = True

        print(f"[env] loading asset {ASSET_FILE} from {ASSET_ROOT}", flush=True)
        self.robot_asset = self.gym.load_asset(self.sim, ASSET_ROOT, ASSET_FILE, asset_opts)
        assert self.robot_asset is not None, f"Failed to load {ASSET_FILE}, please check path."

        # Asset joint information
        self.dof_count = self.gym.get_asset_dof_count(self.robot_asset)
        raw_names = []
        for i in range(self.dof_count):
            n = self.gym.get_asset_dof_name(self.robot_asset, i)
            n = n.decode("utf-8") if isinstance(n, bytes) else n
            raw_names.append(n)

        # Asset -> key12 mapping
        def to_key(name: str):
            s = name.lower()
            if   s.startswith("fl_"): leg = "FL"
            elif s.startswith("fr_"): leg = "FR"
            elif s.startswith("rl_"): leg = "RL"
            elif s.startswith("rr_"): leg = "RR"
            else: return None
            if   "_hip_"   in s: joint = "hip"
            elif "_thigh_" in s: joint = "thigh"
            elif "_calf_"  in s: joint = "calf"
            else: return None
            return f"{leg}_{joint}"

        name2idx = {}
        for i, n in enumerate(raw_names):
            k = to_key(n)
            if k and (k not in name2idx): name2idx[k] = i

        print("\n[DBG] Asset DOFs in order:")
        for i, n in enumerate(raw_names):
            print(f"  {i:02d}: {n}")

        print("\n[DBG] Mapping to key12:")
        for k in self.key12:
            print(f"  {k:12s} -> asset dof idx {name2idx.get(k)}")

        ctrl_idx_list = [name2idx[k] for k in self.key12]
        self.ctrl_idx = np.array(ctrl_idx_list, dtype=np.int32)
        self.ctrl_idx_t = torch.as_tensor(self.ctrl_idx, device=self.device, dtype=torch.long)
        self.ctrl_sign = torch.ones(12, device=self.device, dtype=torch.float32)

        print("[DBG] ctrl_idx (key12 -> raw_names index):", self.ctrl_idx.tolist())
        self.ctrl_leg_tags = [k.split("_")[0].upper() for k in self.key12]

        def _leg_of_key(k: str) -> str:
            return k.split("_")[0].upper()

        legs_seq = [_leg_of_key(k) for k in self.key12]
        expected = ["FL"]*3 + ["FR"]*3 + ["RL"]*3 + ["RR"]*3
        if legs_seq != expected:
            print("[WARN] key12 order is not [FL*3, FR*3, RL*3, RR*3], current:", legs_seq)
            for leg in ("FL","FR","RL","RR"):
                leg3idxs = [i for i, k in enumerate(self.key12) if _leg_of_key(k) == leg]
                print(f"[HINT] {leg} indices in key12:", leg3idxs)
        else:
            print("[OK] key12 order is [FL*3, FR*3, RL*3, RR*3]")

        # Parse rigid body names, establish feet_local
        rb_count = self.gym.get_asset_rigid_body_count(self.robot_asset)
        self.rb_count = rb_count
        rb_names = []
        for j in range(rb_count):
            nm = self.gym.get_asset_rigid_body_name(self.robot_asset, j)
            nm = nm.decode("utf-8") if isinstance(nm, bytes) else nm
            rb_names.append((j, nm))
        rb_names_lower = [(j, nm.lower()) for (j, nm) in rb_names]

        def _pick_leg_body(leg_prefix: str):
            keys = ("foot", "toe", "sole", "ankle")
            cands = [(j, nm) for (j, nm) in rb_names_lower
                     if nm.startswith(leg_prefix + "_") and any(k in nm for k in keys)]
            for prefer in ("foot", "toe", "sole", "ankle"):
                for (j, nm) in cands:
                    if prefer in nm:
                        return j
            return cands[0][0] if cands else None

        leg_order = ["fl", "fr", "rl", "rr"]
        leg2body = {leg: _pick_leg_body(leg) for leg in leg_order}
        missing_feet = [leg for leg, idx in leg2body.items() if idx is None]
        if missing_feet:
            print("[ERR] Cannot find foot rigid bodies for these legs:", missing_feet)
            print("[HINT] Current rigid body names:", [nm for _, nm in rb_names])
            raise AssertionError("feet_local parsing failed, please improve rigid body naming matching rules.")

        self.feet_local = [leg2body["fl"], leg2body["fr"], leg2body["rl"], leg2body["rr"]]
        print("[INFO] feet_local rigid bodies (FL,FR,RL,RR idx):", self.feet_local)

        # Create env + actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, self.cfg.h0)
        yaw0 = 0.0
        pose.r = quat_from_rpy(0.0, 0.0, yaw0)

        actor_name = "go2"
        for env_id in range(self.B):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)
            
            o = self.gym.get_env_origin(env_ptr)  # gymapi.Vec3
            self.env_origins[env_id] = torch.tensor([o.x, o.y, o.z], device=self.device)

            actor_handle = self.gym.create_actor(env_ptr, self.robot_asset, pose, actor_name, env_id, 1)
            self.actor_handles.append(actor_handle)
            actor_index = self.gym.get_actor_index(env_ptr, actor_handle, gymapi.DOMAIN_SIM)
            self.actor_indices.append(actor_index)

        # Rudin terrain: override the env-grid origins with the curriculum-grid placement
        # (terrain_levels = difficulty rows, terrain_types = terrain-type columns). Runs before
        # the first reset so _sample_spawn_xy caches these origins as the spawn points.
        if self.cfg.terrain_type == "rudin" and self.terrain is not None \
                and getattr(self.terrain, "env_origins", None) is not None:
            self._assign_rudin_origins()

        # int32: Isaac Gym's indexed-tensor APIs require int32 actor indices.
        self.actor_indices_t = torch.as_tensor(self.actor_indices, device=self.device, dtype=torch.int32)

        # DOF PD properties (same for all env actors)
        props = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        props["driveMode"][:] = gymapi.DOF_MODE_POS
        kp, kd = self.cfg.pd_kp, self.cfg.pd_kd
        props["stiffness"][:] = 0.0
        props["damping"][:]   = 0.0
        props["stiffness"][self.ctrl_idx] = kp
        props["damping"][self.ctrl_idx]   = kd

        for env_ptr, actor_handle in zip(self.envs, self.actor_handles):
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, props)

        chk = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        print("driveMode unique (after set):", set(chk["driveMode"].tolist()))
        print("stiffness range:", float(chk["stiffness"].min()), "–", float(chk["stiffness"].max()))
        print("damping   range:", float(chk["damping"].min()), "–", float(chk["damping"].max()))
        lo12 = chk["lower"][self.ctrl_idx]; hi12 = chk["upper"][self.ctrl_idx]
        eff12 = chk["effort"][self.ctrl_idx] if "effort" in chk.dtype.names else None
        print("[DBG] lower[12] =", np.round(lo12, 6))
        print("[DBG] upper[12] =", np.round(hi12, 6))
        if eff12 is not None: print("[DBG] effort[12] =", np.round(eff12, 3))

        self.q_default_full = np.zeros(self.dof_count, dtype=np.float32)
        self.q_default_full[self.ctrl_idx] = self.q_default_np

        self.gym.prepare_sim(self.sim)

        # Global DOF count
        self.sim_dof_count = self.gym.get_sim_dof_count(self.sim)
        assert self.sim_dof_count == self.B * self.dof_count, \
            f"sim_dof_count={self.sim_dof_count}, B*dof_count={self.B*self.dof_count}"

        # DOF target tensor: 1D [B*dof_count], then view as (B,dof_count) for use
        self.pos_targets = torch.zeros(self.sim_dof_count, dtype=torch.float32, device=self.device).contiguous()
        self.pos_targets_batch = self.pos_targets.view(self.B, self.dof_count)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_targets))

        # State tensor wrapping (with batch view)
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state_t = gymtorch.wrap_tensor(_dof_state)
        self.dof_state_view = self.dof_state_t.view(self.B, self.dof_count, 2)

        _root = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(_root).view(self.B, 13)

        _rb = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_t = gymtorch.wrap_tensor(_rb).view(self.B, self.rb_count, 13)

        _jac = self.gym.acquire_jacobian_tensor(self.sim, actor_name)
        assert _jac is not None, "acquire_jacobian_tensor('go2') failed"
        self.jacobian = gymtorch.wrap_tensor(_jac)  # Will be reshaped in foot_jacobians

        _cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        assert _cf is not None, "acquire_net_contact_force_tensor() failed"
        self.net_cf = gymtorch.wrap_tensor(_cf).view(self.B, self.rb_count, 3)

        # Joint limits (asset level)
        props_now = chk
        self._lo_slice = torch.as_tensor(props_now["lower"], device=self.device, dtype=torch.float32)
        self._hi_slice = torch.as_tensor(props_now["upper"], device=self.device, dtype=torch.float32)

        # local_targets: (B, dof_count)
        base_local = torch.as_tensor(self.q_default_full, device=self.device, dtype=torch.float32)
        self.local_targets = base_local.view(1, -1).repeat(self.B, 1)
        self.pos_targets_batch[:] = self.local_targets
        self._commit_pos_targets()

        # Run for a short time to stabilize first
        for _ in range(self.cfg.settle_steps_init):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        self._sanity_check_io()
        self._last_action = torch.zeros(self.B, 12, device=self.device)

        self.last_reset_yaw = torch.zeros(self.B, device=self.device)

        # Viewer only watches robot 0
        self.viewer = None
        if self.cfg.use_viewer:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)
            self.gym.viewer_camera_look_at(self.viewer, None,
                                           gymapi.Vec3(2.0, 2.0, 1.2),
                                           gymapi.Vec3(0.0, 0.0, 0.3))

        self.cam_follow = False
        self.cam_dist   = 2.5
        self.cam_height = 1.0
        self.cam_smooth = 0.15
        self._cam_eye = None
        self._cam_tgt = None

        # Initialize cache + gait/SRBD
        self.gait = GaitPlanner(self)
        self.srbd = SRBDModel(self)
        self._update_cache()
        self.srbd._srbd_init_from_isaac()

        # ===== last-contact cache (world frame) =====
        # Most recent touchdown position per leg; stance feet are locked to it
        # so the PD target cannot drag them backward.
        self.last_contact_z  = torch.zeros(self.B, 4, device=self.device)     # (B,4)
        self.last_contact_xy = torch.zeros(self.B, 4, 2, device=self.device)  # (B,4,2)

        # ===== liftoff cache (world frame) =====
        # Most recent liftoff position per leg; used as the swing-trajectory start.
        self.last_liftoff_xyz = torch.zeros(self.B, 4, 3, device=self.device) # (B,4,3)
        # Previous timestep stance_mask (detects the stance -> swing liftoff edge).
        self.prev_stance_mask = torch.ones(self.B, 4, 1, device=self.device)  # (B,4,1)

        # ===== contact-edge cache =====
        # Detects touchdown edges (prev contact 0 -> 1) so ground scraping during
        # swing does not pollute last_contact_*.
        self.prev_contact_flags = torch.zeros(self.B, 4, 1, device=self.device)  # (B,4,1)

        # Initialize: fill last_contact_* with current foot positions
        with torch.no_grad():
            p_foot0 = self.foot_positions()           # (B,4,3)
            self.last_contact_xy[:] = p_foot0[..., 0:2]
            self.last_contact_z[:]  = p_foot0[..., 2]
            self.last_liftoff_xyz[:] = p_foot0
            self.prev_contact_flags[:] = self.contact_flags()

        # ===== stride debug (env0) =====
        self._stride_last_td_xy0 = torch.zeros(4, 2, device=self.device)          # (4,2)
        self._stride_have_td0    = torch.zeros(4, dtype=torch.bool, device=self.device)
        self._stride_count0      = torch.zeros(4, dtype=torch.long, device=self.device)

        # Optional: only print first N touchdowns (prevent spam); set None for unlimited
        self._stride_print_limit = 200

        self._stride_last_td_step0 = torch.full((4,), -10_000, device=self.device, dtype=torch.long)

        # ===== Perception (optional, gated) =====
        # Only build the terrain-perception collector when explicitly enabled, so the
        # blind/flat SRBD path never imports the warp/torch-heavy perception code.
        self.perception = None
        if (getattr(self.cfg, "use_height_obs", False)
                or getattr(self.cfg, "use_terrain_loss", False)
                or getattr(self.cfg, "use_depth_obs", False)):
            assert getattr(self.cfg, "use_perception", False), (
                "use_height_obs / use_terrain_loss / use_depth_obs require use_perception=True"
            )
        assert not (getattr(self.cfg, "use_height_obs", False)
                    and getattr(self.cfg, "use_depth_obs", False)), (
            "use_height_obs and use_depth_obs are mutually exclusive (flat obs vs obs+depth policy)"
        )
        if getattr(self.cfg, "use_perception", False):
            from perception import PerceptionCollector  # lazy: pulls in Warp
            if not getattr(self.cfg, "use_gpu_pipeline", False):
                print("[perception] use_perception=True with use_gpu_pipeline=False: base pose will "
                      "be uploaded host->device each step. Set use_gpu_pipeline=True for zero-copy.")
            self.perception = PerceptionCollector(
                self.terrain, self.cfg.perception, num_envs=self.B, device=str(self.device)
            )
            print(f"[perception] collector ready (B={self.B}, device={self.device}, "
                  f"terrain_type={self.cfg.terrain_type}).")

    def collect_perception(self):
        """Gather terrain perception for the current robot poses.

        Thin convenience wrapper: feeds the live root state (world position and
        xyzw quaternion) to the perception collector. Requires
        ``cfg.use_perception=True``.

        Returns:
            dict: ``depth_clean`` / ``depth_noisy`` / ``height_map`` / ``cam_pos``
            / ``cam_quat`` (see :meth:`perception.collector.PerceptionCollector.collect`).
        """
        assert self.perception is not None, "cfg.use_perception must be True to collect perception"
        return self.perception.collect(self.root_state[:, 0:3], self.root_state[:, 3:7])

    @property
    def obs_dim(self):
        """Observation dimension: 36 blind, +n height points when use_height_obs."""
        dim = 36
        if getattr(self.cfg, "use_height_obs", False):
            dim += self.perception.height_sampler.num_points
        return dim

    def terrain_height_diff(self, xy, smooth=True):
        """Differentiable terrain height at world (B, N, 2) points -> (B, N).

        Gradient-carrying sibling of :meth:`_terrain_height` (which rounds to the
        nearest cell and is grad-free): samples the perception heightfield with
        bilinear ``grid_sample``, so terrain slope back-propagates into ``xy``.
        Feed SRBD-predicted base/foot positions from the training loop.

        Args:
            xy: (B, N, 2) world-frame query points, metres.
            smooth: Use the Gaussian-blurred loss field (default; see
                ``PerceptionCfg.hm_loss_blur_cells``) instead of the exact one.
        """
        assert self.perception is not None, "cfg.use_perception must be True for terrain_height_diff"
        return self.perception.height_sampler.sample_points(xy, smooth=smooth)

    def _sanity_check_io(self):
        """One-time startup check: print Jacobian shapes/norm so IO problems fail loudly."""
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        J = self.foot_jacobians()          # (B, 4, 3, cols)
        print("[CHECK] foot_jacobians shape:", tuple(J.shape), " (B,4,3,cols)")

        # Jacobian DOF offset (6 for floating base)
        dof_offset = getattr(self, "jac_dof_offset", 0)
        ctrl_cols = self.ctrl_idx_t + dof_offset       # (12,)

        J12 = J[..., ctrl_cols]                       # (B,4,3,12)
        print("[CHECK] J12 shape:", tuple(J12.shape), " (B,4,3,12)")
        print("[CHECK] ||J12||:", float(J12.norm()))

    def _commit_pos_targets(self):
        """Push the current position-target buffer to the simulator."""
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.pos_targets)
        )

    def _limit_step(self, tgt_prev: torch.Tensor, tgt_new: torch.Tensor, max_step=0.04):
        """Rate-limit a target update: clamp the per-element change to +-max_step."""
        delta = torch.clamp(tgt_new - tgt_prev, min=-max_step, max=+max_step)
        return tgt_prev + delta

    @torch.no_grad()
    def contact_force_values(self):
        """Net contact-force magnitude per foot, (B, 4, 1) newtons."""
        self.gym.refresh_net_contact_force_tensor(self.sim)
        F = self.net_cf[:, self.feet_local, :].norm(dim=-1, keepdim=True)  # (B,4,1)
        return F

    @torch.no_grad()
    def contact_flags(self, thresh=None):
        """Hysteresis contact detector, (B, 4, 1) in {0, 1}.

        A foot switches to contact above ``cfg.contact_on_n`` newtons and back
        to no-contact below ``cfg.contact_off_n``; in between it keeps its
        previous state (suppresses chattering near the threshold).
        """
        cfg = self.cfg
        F = self.contact_force_values().squeeze(-1)  # (B,4)

        on  = float(getattr(cfg, "contact_on_n",  cfg.contact_thresh_n))
        off = float(getattr(cfg, "contact_off_n", cfg.contact_thresh_n * 0.6))

        if not hasattr(self, "_contact_state"):
            self._contact_state = (F > on).float()  # (B,4)

        # Rule: >on set to 1; <off set to 0; in between keep
        self._contact_state = torch.where(F > on,  torch.ones_like(self._contact_state), self._contact_state)
        self._contact_state = torch.where(F < off, torch.zeros_like(self._contact_state), self._contact_state)

        return self._contact_state.unsqueeze(-1)  # (B,4,1)

    # ============================================================
    # Per-env sampling helpers — shared by reset() (all envs) and
    # reset_envs() (a subset). Each takes a 1D LongTensor of env ids.
    # ============================================================
    def _as_env_ids(self, env_ids):
        """Normalize env_ids to a 1D LongTensor on device. None -> all envs."""
        dev = self.device
        if env_ids is None:
            return torch.arange(self.B, device=dev, dtype=torch.long)
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=dev, dtype=torch.long)
        return env_ids.view(-1)

    def _sample_gait(self, env_ids):
        """Assign gait id + leg phase offsets. gait_mode<0 -> random from cfg.gait_choices."""
        if not hasattr(self, "gait_table"):
            return
        dev = self.device
        n = env_ids.numel()
        if self.cfg.gait_mode < 0:
            choices = torch.as_tensor(
                getattr(self.cfg, "gait_choices", tuple(range(self.num_gaits))),
                device=dev, dtype=torch.long,
            ).clamp(0, self.num_gaits - 1)
            new_gids = choices[torch.randint(low=0, high=choices.numel(), size=(n,), device=dev)]
        else:
            gid = max(0, min(self.num_gaits - 1, int(self.cfg.gait_mode)))
            new_gids = torch.full((n,), gid, dtype=torch.long, device=dev)
        self.gait_ids[env_ids] = new_gids
        self.leg_phase_offsets_B[env_ids] = self.gait_table[new_gids]

    def _sample_command(self, env_ids):
        """Sample velocity command cmd_rand (+ vx_star alias). rand_cmd off -> cfg.cmd_fixed."""
        dev = self.device
        n = env_ids.numel()
        cfg = self.cfg
        if getattr(cfg, "terrain_type", "flat") == "rudin":
            # Rudin-matched omnidirectional command (fair comparison): vx, vy, yaw drawn from the
            # rudin_cmd_* ranges, then zero tiny planar commands so "stand still" is a valid target
            # (mirrors legged_robot._resample_commands' deadband).
            rx, ry, rw = cfg.rudin_cmd_lin_vel_x, cfg.rudin_cmd_lin_vel_y, cfg.rudin_cmd_ang_vel_yaw
            vx  = torch.empty(n, device=dev).uniform_(float(rx[0]), float(rx[1]))
            vy  = torch.empty(n, device=dev).uniform_(float(ry[0]), float(ry[1]))
            yaw = torch.empty(n, device=dev).uniform_(float(rw[0]), float(rw[1]))
            keep = (torch.sqrt(vx * vx + vy * vy) > float(cfg.rudin_cmd_deadband)).to(vx.dtype)
            vx = vx * keep
            vy = vy * keep
        elif cfg.rand_cmd:
            vx  = torch.empty(n, device=dev).uniform_(cfg.vx_min,  cfg.vx_max)
            vy  = torch.empty(n, device=dev).uniform_(cfg.vy_min,  cfg.vy_max)
            yaw = torch.empty(n, device=dev).uniform_(cfg.yaw_min, cfg.yaw_max)
        else:
            cmd_fixed = getattr(cfg, "cmd_fixed", (0.5, 0.0, 0.0))
            vx  = torch.full((n,), float(cmd_fixed[0]), device=dev)
            vy  = torch.full((n,), float(cmd_fixed[1]), device=dev)
            yaw = torch.full((n,), float(cmd_fixed[2]), device=dev)
        self.cmd_rand[env_ids, 0] = vx
        self.cmd_rand[env_ids, 1] = vy
        self.cmd_rand[env_ids, 2] = yaw
        self.vx_star[env_ids]     = vx   # alias still used by Raibert / loss

    def _sample_step_freq(self, env_ids):
        """Sample per-env step frequency. rand_step_freq off -> constant cfg.step_freq."""
        dev = self.device
        cfg = self.cfg
        if not hasattr(self, "step_freq_B"):
            self.step_freq_B = torch.full((self.B,), cfg.step_freq, device=dev)
        if getattr(cfg, "rand_step_freq", False):
            self.step_freq_B[env_ids] = torch.empty(env_ids.numel(), device=dev).uniform_(
                cfg.step_freq_min, cfg.step_freq_max
            )
        else:
            self.step_freq_B[env_ids] = cfg.step_freq

    def _sample_swing_height(self, env_ids):
        """Reset swing height to the configured default for env_ids."""
        if not hasattr(self, "swing_height_B"):
            self.swing_height_B = torch.full((self.B,), self.cfg.swing_height, device=self.device)
        self.swing_height_B[env_ids] = self.cfg.swing_height

    def _assign_rudin_origins(self):
        """Place robots on the Rudin curriculum grid (mirrors legged_gym _get_env_origins):
        random difficulty level (<= max_init_terrain_level) and terrain type spread evenly
        across the columns. Overrides self.env_origins with the per-robot cell origins."""
        cfg = self.cfg.rudin_terrain
        grid = torch.as_tensor(self.terrain.env_origins, device=self.device, dtype=torch.float32)  # (R,C,3)
        max_init = cfg.max_init_terrain_level if cfg.curriculum else cfg.num_rows - 1
        max_init = int(min(max_init, cfg.num_rows - 1))
        self.terrain_levels = torch.randint(0, max_init + 1, (self.B,), device=self.device)
        self.terrain_types = torch.div(
            torch.arange(self.B, device=self.device),
            (self.B / cfg.num_cols),
            rounding_mode="floor",
        ).long().clamp(0, cfg.num_cols - 1)
        # Persist the full grid + difficulty ceiling so the dynamic curriculum can re-index origins
        # after promote/demote (mirrors legged_gym's self.terrain_origins / self.max_terrain_level).
        self.terrain_origins = grid                 # (R,C,3)
        self.max_terrain_level = int(cfg.num_rows)
        self.env_origins[:] = grid[self.terrain_levels, self.terrain_types]

    def _update_terrain_curriculum(self, env_ids):
        """Rudin's game-inspired curriculum (port of legged_gym _update_terrain_curriculum).

        Called for each env being reset (fall or episode timeout), before respawn. Promotes a robot
        one difficulty row if it walked more than half a cell from its origin, demotes it if it
        covered less than half the distance its command implies over an episode, recycles robots that
        clear the hardest row to a random row, then re-indexes env_origins to the new cell.
        """
        # Don't change difficulty on the very first reset (origins still being established).
        if not self.init_done:
            return
        cfg = self.cfg.rudin_terrain
        # distance walked from the (fixed) cell origin this episode
        distance = torch.norm(self.base_pos[env_ids, 0:2] - self.env_origins[env_ids, 0:2], dim=1)
        # robots that walked far enough progress to harder terrain
        move_up = distance > cfg.terrain_length / 2
        # robots that covered less than half their commanded distance go to simpler terrain
        cmd_speed = torch.norm(self.cmd_rand[env_ids, 0:2], dim=1)
        move_down = (distance < cmd_speed * self.max_episode_length_s * 0.5) & (~move_up)
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        # robots that solve the last level are sent to a random one (else clamp at >= 0)
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def _terrain_height(self, xy):
        """World (N,2) -> shared-terrain surface height z (N,). Flat plane -> zeros."""
        if getattr(self, "height_samples", None) is None:
            return torch.zeros(xy.shape[0], device=self.device)
        hs = self.height_samples                      # (H, W) meters
        H, W = hs.shape
        ix = ((xy[:, 0] - self._terr_x_offset) / self._terr_h_scale).round().long().clamp(0, H - 1)
        iy = ((xy[:, 1] - self._terr_y_offset) / self._terr_h_scale).round().long().clamp(0, W - 1)
        return hs[ix, iy]

    def _sample_spawn_xy(self, env_ids):
        """Assign each robot's spawn (x,y) on the shared terrain.

        rand_spawn_xy -> uniform in +-spawn_area_half_m (re-scattered each reset);
        otherwise keep the current spots (defaults to the Isaac env-grid origins, so
        legacy placement is preserved when scattering is off).
        """
        dev = self.device
        cfg = self.cfg
        if not hasattr(self, "spawn_xy"):
            self.spawn_xy = self.env_origins[:, 0:2].clone()
        if getattr(cfg, "terrain_type", "flat") == "rudin":
            # Rudin placement: jitter around the assigned grid-cell origin each reset
            # (matches legged_gym _reset_root_states' +-1 m offset). env_origins already
            # holds the per-robot cell origins from _assign_rudin_origins().
            n = env_ids.numel()
            j = float(getattr(cfg, "rudin_spawn_jitter_m", 1.0))
            base = self.env_origins[env_ids, 0:2]
            self.spawn_xy[env_ids, 0] = base[:, 0] + torch.empty(n, device=dev).uniform_(-j, j)
            self.spawn_xy[env_ids, 1] = base[:, 1] + torch.empty(n, device=dev).uniform_(-j, j)
        elif getattr(cfg, "rand_spawn_xy", False):
            n = env_ids.numel()
            half = float(getattr(cfg, "spawn_area_half_m", 8.0))
            self.spawn_xy[env_ids, 0] = torch.empty(n, device=dev).uniform_(-half, half)
            self.spawn_xy[env_ids, 1] = torch.empty(n, device=dev).uniform_(-half, half)

    def _write_spawn_pose(self, env_ids):
        """Random yaw + root-state pose for env_ids, with terrain-aware spawn z and
        aerial-phase rejection (ensure >=2 feet would be in stance at the chosen phase)."""
        dev = self.device
        n = env_ids.numel()
        if not hasattr(self, "phase"):
            self.phase = torch.zeros(self.B, device=dev)

        # Random initial phase, avoiding landing mid-aerial (need >=2 stance feet)
        max_try = 50
        for _ in range(max_try):
            phase_try = 2 * math.pi * torch.rand(n, device=dev)              # (n,)
            phase_offsets = self.leg_phase_offsets_B[env_ids]               # (n,4)
            phases = phase_offsets + phase_try.view(n, 1)                   # (n,4)
            beta_B, _, _ = self.gait._get_beta_minfeet_allow_aerial()       # (B,)
            beta_sub = beta_B[env_ids]                                      # (n,)
            stance = (self.gait._phase_u(phases) < beta_sub.view(n, 1)).float()
            if (stance.sum(dim=1) >= 2).all():
                self.phase[env_ids] = phase_try
                break
        else:
            self.phase[env_ids] = phase_try

        # Random yaw + placement (x,y) + terrain-aware z
        self.gym.refresh_actor_root_state_tensor(self.sim)
        yaws = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
        self.last_reset_yaw[env_ids] = yaws.clone()

        self._sample_spawn_xy(env_ids)
        xy = self.spawn_xy[env_ids]                                         # (n,2)
        z  = self._terrain_height(xy) + self.cfg.h0                         # (n,)

        half = 0.5 * yaws
        self.root_state[env_ids, 0] = xy[:, 0]
        self.root_state[env_ids, 1] = xy[:, 1]
        self.root_state[env_ids, 2] = z
        # yaw-only quaternion, Isaac xyzw order: qx=qy=0, qz=sin(yaw/2), qw=cos(yaw/2)
        self.root_state[env_ids, 3] = 0.0
        self.root_state[env_ids, 4] = 0.0
        self.root_state[env_ids, 5] = torch.sin(half)
        self.root_state[env_ids, 6] = torch.cos(half)
        # zero linear + angular velocity
        self.root_state[env_ids, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))

    def reset(self, it: Optional[int] = None):
        """Global reset: re-sample gait/pose/command for every robot and settle.

        Re-samples gait ids, spawn poses (phase, yaw, terrain-aware placement),
        velocity commands and step frequency / swing height, restores the
        default joint posture, lets the sim settle for a few steps, then
        re-initialises the SRBD state and the contact caches.

        Args:
            it: Training iteration index (unused; kept for call-site clarity).
        """
        dev = self.device

        self.t = 0
        self.ep_len_buf[:] = 0
        all_ids = self._as_env_ids(None)   # all envs

        # 1) Gait, 2) spawn pose (phase + yaw + terrain-aware placement),
        # 3) velocity command, 4) step frequency / swing height — all shared with reset_envs()
        self._sample_gait(all_ids)
        self._write_spawn_pose(all_ids)
        self._sample_command(all_ids)
        self._sample_step_freq(all_ids)
        self._sample_swing_height(all_ids)

        # 4) Joint targets back to default posture
        base_local = torch.as_tensor(self.q_default_full, device=dev, dtype=torch.float32)
        self.local_targets = base_local.view(1, -1).repeat(self.B, 1)
        self.pos_targets_batch[:] = self.local_targets
        self._commit_pos_targets()

        for _ in range(self.cfg.settle_steps_reset):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        # 5) Engineering version: give a little initial forward velocity (batched)
        if not PURE_PAPER_MODE:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            v0_body = torch.full((self.B,), 0.10, device=dev)
            if hasattr(self, "stop_cmd_mask"):
                v0_body = torch.where(self.stop_cmd_mask.bool(), torch.zeros_like(v0_body), v0_body)
            yaw = self.last_reset_yaw
            self.root_state[:, 7] = v0_body * torch.cos(yaw)
            self.root_state[:, 8] = v0_body * torch.sin(yaw)
            self.root_state[:, 9] = 0.0
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))

        # 6) Refresh cache & SRBD initialization
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Clear all env DOFs: q=default, qd=0
        base_q = torch.as_tensor(self.q_default_full, device=dev, dtype=torch.float32)  # (dof_count,)
        self.dof_state_view[:, :, 0] = base_q.view(1, -1).repeat(self.B, 1)
        self.dof_state_view[:, :, 1] = 0.0

        # Write back to all actors (indexed needs int32)
        actor_ids = self.actor_indices_t.to(dtype=torch.int32)  # (B,)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state_t),
            gymtorch.unwrap_tensor(actor_ids),
            actor_ids.numel()
       )


        self._update_cache()
        self.srbd._srbd_init_from_isaac()

        self.srbd_p = self.srbd_p.detach()
        self.srbd_v = self.srbd_v.detach()
        self.srbd_q = self.srbd_q.detach()
        self.srbd_w = self.srbd_w.detach()

        # After reset: reset last_contact_*, avoid stance locking to “old episode” foot positions
        with torch.no_grad():
            p_foot = self.foot_positions()            # (B,4,3)
            self.last_contact_xy[:] = p_foot[..., 0:2]
            self.last_contact_z[:]  = p_foot[..., 2]
            # On reset: liftoff start point also reset to current foot (avoid previous episode residue)
            self.last_liftoff_xyz[:] = p_foot
            # prev stance set to 1, avoid first frame after reset being mistaken as “just lifted off”
            self.prev_stance_mask[:] = 1.0
            # contact edge cache sync
            if not hasattr(self, "prev_contact_flags"):
                self.prev_contact_flags = torch.zeros(self.B, 4, 1, device=self.device)
            self.prev_contact_flags[:] = self.contact_flags()

        # env0 stride debug cache reset

        self._stride_have_td0[:] = False
        self._stride_count0[:] = 0
        self._stride_last_td_step0[:] = -10000
        # Key: last_td_xy0 also needs to reset to current foot position, avoid first stride explosion
        p_foot0 = self.foot_positions()[0, :, 0:2].detach()
        self._stride_last_td_xy0[:] = p_foot0

        # The initial (global) reset is done; from here on per-env resets may run the dynamic
        # curriculum (legged_gym sets init_done at the end of construction for the same reason).
        self.init_done = True

    def reset_envs(self, env_ids):
        """Local reset: respawn only the robots in ``env_ids``.

        Same sampling pipeline as :meth:`reset` but restricted to a subset
        (used for per-env resets on falls / episode timeouts). On Rudin terrain
        the dynamic curriculum runs first, so a robot may be promoted/demoted
        to a different difficulty row before respawning.

        Args:
            env_ids: int / list[int] / numpy array / torch tensor of env indices.
        """
        env_ids = self._as_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        dev = self.device
        n = env_ids.numel()

        # 0) Rudin dynamic curriculum: promote/demote BEFORE respawning, using the distance walked
        # this episode and the still-current (just-ended) command. Updates self.env_origins so the
        # spawn helper below jitters around the new cell. Mirrors legged_gym reset_idx ordering.
        if self.cfg.terrain_type == "rudin" and self.cfg.rudin_terrain.dynamic_curriculum:
            self._update_terrain_curriculum(env_ids)
        self.ep_len_buf[env_ids] = 0

        # 1) gait, 2) spawn pose (phase + yaw + terrain-aware placement),
        # 3) velocity command, 4) step frequency / swing height — same helpers as reset()
        self._sample_gait(env_ids)
        self._write_spawn_pose(env_ids)
        self._sample_command(env_ids)
        self._sample_step_freq(env_ids)
        self._sample_swing_height(env_ids)

        # 6) Reset joint targets for these robots back to default standing posture
        base_local = torch.as_tensor(self.q_default_full, device=dev, dtype=torch.float32)
        # local_targets: (B, dof_count)
        self.local_targets[env_ids]      = base_local.unsqueeze(0)
        self.pos_targets_batch[env_ids]  = self.local_targets[env_ids]
        self._commit_pos_targets()

        # 7) Refresh cache & SRBD (for simplicity, do a global refresh)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        base_q = torch.as_tensor(self.q_default_full, device=dev)
        self.dof_state_view[env_ids, :, 0] = base_q.view(1, -1).expand(n, -1)
        self.dof_state_view[env_ids, :, 1] = 0.0

        actor_ids = self.actor_indices_t[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state_t),
            gymtorch.unwrap_tensor(actor_ids),
            actor_ids.numel()
        )

        self._update_cache()
        self.srbd._srbd_init_from_isaac()

        self.srbd_p = self.srbd_p.detach()
        self.srbd_v = self.srbd_v.detach()
        self.srbd_q = self.srbd_q.detach()
        self.srbd_w = self.srbd_w.detach()

        # After local reset: only update last_contact_* for these envs (otherwise stance will lock to old footprints)
        with torch.no_grad():
            p_foot = self.foot_positions()            # (B,4,3)
            self.last_contact_xy[env_ids] = p_foot[env_ids, :, 0:2]
            self.last_contact_z[env_ids]  = p_foot[env_ids, :, 2]
            # Synchronously reset liftoff start point
            self.last_liftoff_xyz[env_ids] = p_foot[env_ids]
            self.prev_stance_mask[env_ids] = 1.0
            # Synchronize contact edge cache
            if not hasattr(self, "prev_contact_flags"):
                self.prev_contact_flags = torch.zeros(self.B, 4, 1, device=self.device)
            self.prev_contact_flags[env_ids] = self.contact_flags()[env_ids]

        if (env_ids == 0).any():
            self._stride_have_td0[:] = False
            self._stride_count0[:] = 0
            self._stride_last_td_step0[:] = -10000
            # Critical: last_td_xy0 must also be reset to current foot position, avoid first stride explosion
            p_foot0 = self.foot_positions()[0, :, 0:2].detach()
            self._stride_last_td_xy0[:] = p_foot0


    def demo_trot(self, seconds=6.0, amp_thigh=0.25, amp_calf=0.45):
        """Policy-free smoke test: drive all envs with a scripted sinusoidal trot.

        Args:
            seconds: How long to run.
            amp_thigh, amp_calf: Sinusoid amplitudes added to the default posture.
        """
        steps = int(seconds / self.cfg.dt)
        self.phase = torch.zeros(self.B, device=self.device)
        for _ in range(steps):
            phases = self.leg_phase_offsets + self.phase[0]
            swing  = torch.clamp(torch.sin(phases), min=0.0)
            q_ref  = torch.as_tensor(self.q_default_np, device=self.device).clone()
            q_ref[1::3] += amp_thigh * swing
            q_ref[2::3] -= amp_calf  * swing

            target_slice = self.local_targets.clone()
            target_slice[:, self.ctrl_idx_t] = q_ref * self.ctrl_sign
            target_slice = torch.max(torch.min(target_slice, self._hi_slice), self._lo_slice)
            target_slice = self._limit_step(self.local_targets, target_slice, max_step=0.08)
            self.local_targets = target_slice
            self.pos_targets_batch[:] = self.local_targets
            self._commit_pos_targets()

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            if self.viewer is not None:
                self.gym.step_graphics(self.sim)
                self._update_chase_camera()
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)

            self.phase = (self.phase + 2*math.pi*self.cfg.step_freq*self.cfg.dt) % (2*math.pi)

    def _rpy_from_quat_wxyz(self):
        """Batched base orientation -> (roll, pitch, yaw), each (B,) radians."""
        qw, qx, qy, qz = self.base_quat[:, 0], self.base_quat[:, 1], self.base_quat[:, 2], self.base_quat[:, 3]

        sinr_cosp = 2.0 * (qw*qx + qy*qz)
        cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw*qy - qz*qx)
        sinp = torch.clamp(sinp, -1.0 + 1e-6, 1.0 - 1e-6)
        pitch = torch.asin(sinp)

        siny_cosp = 2.0 * (qw*qz + qx*qy)
        cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def _update_cache(self):
        """Refresh the cached base pose/velocity/joint tensors from Isaac Gym.

        Populates ``base_pos``, ``base_quat`` (wxyz), world/body linear and
        angular velocities, joint positions/velocities, and roll/pitch/yaw --
        the state everything else (gait, SRBD init, observations) reads.
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        base = self.root_state  # (B,13)

        self.base_pos  = base[:, 0:3]                      # (B,3)
        q_xyzw = base[:, 3:7]                              # (B,4)
        self.base_quat = torch.stack(                      # (B,4) wxyz
            [q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], dim=1
        )

        # World frame velocity / angular velocity
        self.base_lin_world = base[:, 7:10]                # (B,3)
        self.base_ang_world = base[:, 10:13]               # (B,3)



        # Body frame velocity / angular velocity (batched: quat_rotate_inverse_wxyz
        # already supports (B,4)/(B,3) input, so no per-env loop is needed)
        self.base_lin_body = quat_rotate_inverse_wxyz(self.base_quat, self.base_lin_world, self.device)  # (B,3)
        self.base_ang_body = quat_rotate_inverse_wxyz(self.base_quat, self.base_ang_world, self.device)  # (B,3)

        # world frame alias
        self.base_lin = self.base_lin_world
        self.base_ang = self.base_ang_world

        # Joints
        dof = self.dof_state_view  # (B,dof_count,2)
        self.q  = dof[..., 0]      # (B,dof_count)
        self.qd = dof[..., 1]      # (B,dof_count)

        # SRBD 2D legacy variables (maintain interface)
        self.p = torch.stack([self.base_pos[:, 0], self.base_pos[:, 2]], dim=1)  # (B,2)
        self.v = torch.stack([self.base_lin[:, 0], self.base_lin[:, 2]], dim=1)  # (B,2)

        self.roll, self.pitch, self.yaw = self._rpy_from_quat_wxyz()  # (B,)
        self.theta = self.pitch.clone()
        self.omega = self.base_ang[:, 1]  # (B,)


    def _yaw_from_quat(self) -> float:
        """Yaw of robot 0 (radians), used by the chase camera."""
        qw, qx, qy, qz = self.base_quat[0]
        return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

    def _update_chase_camera(self):
        """Smoothly track robot 0 with the viewer camera (when cam_follow is on)."""
        if self.viewer is None or not self.cam_follow: return
        px, py, pz = [float(v) for v in self.base_pos[0, :3]]
        yaw = self._yaw_from_quat()
        back = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])
        up   = np.array([0.0, 0.0, 1.0])
        desired_eye = np.array([px, py, pz]) + self.cam_dist * back + self.cam_height * up
        desired_tgt = np.array([px, py, pz + 0.30])
        if self._cam_eye is None:
            self._cam_eye = desired_eye; self._cam_tgt = desired_tgt
        else:
            a = float(self.cam_smooth)
            self._cam_eye = (1 - a) * self._cam_eye + a * desired_eye
            self._cam_tgt = (1 - a) * self._cam_tgt + a * desired_tgt
        self.gym.viewer_camera_look_at(
            self.viewer, None,
            gymapi.Vec3(*self._cam_eye.tolist()),
            gymapi.Vec3(*self._cam_tgt.tolist()),
        )

    # ---------------- helpers / sensors ----------------
    @torch.no_grad()
    def foot_positions(self):
        """Foot positions (world frame), (B,4,3)"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        pos = self.rb_state_t[:, self.feet_local, 0:3]  # (B,4,3)
        return pos

    @torch.no_grad()
    def foot_jacobians(self):
        """
        Foot Jacobian matrices, returns (B, 4, 3, cols)

        Note:
        - For floating base, Jacobian shape is (B, num_links, 6, num_dofs+6)
          First 6 columns correspond to base DOF, we need the remaining num_dofs part.
        """
        self.gym.refresh_jacobian_tensors(self.sim)
        J_flat = self.jacobian                    # Original tensor

        B_j, nb_j, six, cols = J_flat.shape
        assert B_j == self.B
        assert six == 6, "Jacobian 3rd dim must be 6 (linear + angular velocity)"

        # First call: automatically determine DOF offset based on column count
        if not hasattr(self, "jac_dof_offset"):
            if cols == self.dof_count + 6:
                # Floating base: first 6 columns are base DOF
                self.jac_dof_offset = 6
                print(f"[INFO] Jacobian detected as floating-base: cols={cols}, dof_count={self.dof_count}, offset=6")
            elif cols == self.dof_count:
                # Fixed base: exactly equals joint DOF count
                self.jac_dof_offset = 0
                print(f"[INFO] Jacobian detected as fixed-base: cols={cols}, dof_count={self.dof_count}, offset=0")
            else:
                # Non-standard case: treat last self.dof_count columns as joint DOFs
                self.jac_dof_offset = max(0, cols - self.dof_count)
                print(f"[WARN] Unexpected Jacobian shape {J_flat.shape}, "
                      f"treating last {self.dof_count} columns as joint DOFs, "
                      f"offset={self.jac_dof_offset}")

        # Only take linear velocity part 0:3
        J_lin = J_flat[:, :, 0:3, :]              # (B, nb_j, 3, cols)

        # Take rows corresponding to four foot rigid bodies
        assert nb_j > max(self.feet_local), "Jacobian num_links < feet_local index"
        J_feet = torch.stack(
            [J_lin[:, self.feet_local[i]] for i in range(4)],
            dim=1
        )                                         # (B, 4, 3, cols)

        return J_feet


    def estimate_foot_forces(self, q_ref12, q_now12, qd_now12, stance_mask):
        """Estimate ground-reaction forces from the PD torques via the Jacobian.

        Computes the joint torques the PD controller implies, maps them to foot
        forces with a damped pseudo-inverse of the stance-weighted Jacobian
        (solved in float64 for numerical stability), then applies Fz clamping
        and a friction-cone limit. Differentiable in ``q_ref12`` -- this is the
        input path through which policy gradients reach the SRBD step.

        Args:
            q_ref12: (B, 12) reference joint angles (carries gradients).
            q_now12, qd_now12: (B, 12) measured joint state (detached).
            stance_mask: (B, 4, 1) stance feet mask.

        Returns:
            (B, 4, 3) world-frame foot forces.
        """
        dev = self.device
        Kp, Kd = self.cfg.pd_kp, self.cfg.pd_kd

        B = self.B
        tau = Kp * (q_ref12 - q_now12) - Kd * qd_now12  # (B,12)
        tau = tau.view(B, 12, 1)

        J_all = self.foot_jacobians()                   # (B,4,3,cols)
        dof_offset = getattr(self, "jac_dof_offset", 0)
        J12 = J_all[..., self.ctrl_idx_t + dof_offset]  # (B,4,3,12) columns corresponding to joint DOFs

        # ---- Batched stance-weighted Jacobian (replaces the per-env loop) ----
        # Stack the 4 feet × 3 spatial rows into a (B,12,12) block, exactly like
        # the old torch.cat([Jw[i] for i in range(4)], dim=0) but for all envs.
        Jw   = J12 * stance_mask.view(B, 4, 1, 1)       # (B,4,3,12)
        Jbig = Jw.reshape(B, 12, 12)                    # (B,12,12)

        # ---- Damped-pseudo-inverse solve in float64 for precision ----
        # (.double()/.float() are differentiable, so the policy -> q_ref12 -> tau
        #  -> f autograd path is preserved; float64 removes batched-vs-loop drift.)
        JJt = (Jbig @ Jbig.transpose(-1, -2)).double()  # (B,12,12)
        rhs = (Jbig @ tau).double()                     # (B,12,1)
        eye = torch.eye(12, device=dev, dtype=torch.float64)            # broadcasts over batch
        U, S, Vh = torch.linalg.svd(JJt + 1e-9 * eye)   # (B,12,12)/(B,12)/(B,12,12)
        S = torch.clamp(S, min=1e-3)
        Ainv = U @ torch.diag_embed(1.0 / S) @ Vh       # (B,12,12)
        y = Ainv @ rhs                                  # (B,12,1)
        f = (-y.view(B, 4, 3)).float()                  # (B,4,3)

        # ---- Friction cone + Fz clamping (batched) ----
        fz = torch.clamp(f[..., 2:3], min=self.cfg.fz_min, max=self.cfg.fz_max) * stance_mask  # (B,4,1)
        ft = f[..., :2]                                 # (B,4,2)
        ft_norm = torch.linalg.norm(ft, dim=-1, keepdim=True)          # (B,4,1)
        ft_max = self.cfg.mu_tangent * fz                              # (B,4,1)
        scale = torch.clamp(ft_max / (ft_norm + 1e-6), max=1.0)        # (B,4,1)
        ft = ft * scale                                 # (B,4,2)
        f = torch.cat([ft, fz], dim=-1)                 # (B,4,3)
        return f

    def step(self, delta_q: torch.Tensor):
        """Advance the Isaac Gym simulation by one control step.

        Converts the policy output into PD joint targets (tanh squashing,
        per-joint scaling, joint limits, rate limiting), steps PhysX once,
        refreshes the cached state, maintains the touchdown caches and
        evaluates termination.

        Args:
            delta_q: (B, 12) raw policy output (joint-angle offsets before
                tanh/scaling).

        Returns:
            (obs, extra, q_err, q_ref12) where ``extra`` carries the ``done``
            (fall) and ``timeout`` (Rudin episode limit) masks, and ``q_ref12``
            is the gradient-carrying joint reference consumed by the SRBD step.
        """
        cfg = self.cfg

        # Update step frequency / swing height per robot from the current command.
        self.gait._update_gait_from_cmd()

        # Update phase using per-env step frequency: phase_{t+1} = phase_t + 2π f Δt
        # self.step_freq_B: (B,)
        move = getattr(self, "move_mask_B", torch.ones(self.B, device=self.device))
        self.phase = (self.phase + 2*math.pi*self.step_freq_B*cfg.dt*move) % (2*math.pi)


        scale12 = torch.as_tensor(self.cfg.delta_q_scale12, device=self.device, dtype=torch.float32).view(1, 12)
        ctrl_sign12 = self.ctrl_sign.view(1,12).to(self.device)   # (1,12)  ±1
        delta_q = torch.tanh(delta_q) * scale12


        #---------------------------------------
        # stop env detection
        dead = float(getattr(cfg, "cmd_deadzone", 0.05))
        v_mag = torch.linalg.norm(self.cmd_rand[:, 0:2], dim=1)
        stop_env = (v_mag < dead)  # (B,)
        # When stopped: don't allow policy to disturb standing posture
        delta_q = torch.where(stop_env[:, None], torch.zeros_like(delta_q), delta_q)
        #---------------------------------------

        q_default = torch.as_tensor(self.q_default_np, device=self.device).view(1,12).repeat(self.B,1)

        q_ref12 = q_default + delta_q


        # Target joints
        target_slice = self.local_targets.clone()   # (B,dof)
        # .detach(): local_targets only feeds Isaac's (non-differentiable) PD targets,
        # but it is rate-limited against its previous value every step, so without the
        # detach it chains an autograd graph across the whole run (never freed under
        # ONLY_ITERATE_NO_RESET). The gradient path lives in the returned q_ref12.
        target_slice[:, self.ctrl_idx_t] = (q_ref12 * ctrl_sign12).detach()   # policy -> sim target positions for 12 joint DOFs

        # Clamping & limits & rate of change
        target_slice = torch.max(torch.min(target_slice, self._hi_slice), self._lo_slice)
        target_slice = self._limit_step(self.local_targets, target_slice, max_step=0.05)
        self.local_targets = target_slice
        self.pos_targets_batch[:] = self.local_targets
        self._commit_pos_targets()

        # Isaac simulation one step
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if self.viewer is not None:
            self._render_cnt += 1
            if self._render_cnt % self.render_every == 0:
                self.gym.step_graphics(self.sim) 
                self._update_chase_camera() 
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if self.gym.query_viewer_has_closed(self.viewer):
                    self.gym.destroy_viewer(self.viewer); self.viewer = None

        self._update_cache()

        # ===== debug: world vs body velocity/position direction (env0) =====
        if self.t % 200 == 0:
            # pos: world frame
            px = float(self.base_pos[0, 0].item())
            py = float(self.base_pos[0, 1].item())

            # v_world: world frame (if you don't have base_lin_world, use base_lin - check your cache variable name)
            if hasattr(self, "base_lin_world"):
                vwx = float(self.base_lin_world[0, 0].item())
                vwy = float(self.base_lin_world[0, 1].item())
            else:
                vwx = float(self.base_lin[0, 0].item())
                vwy = float(self.base_lin[0, 1].item())
            # v_body: body frame
            vbx = float(self.base_lin_body[0, 0].item())
            vby = float(self.base_lin_body[0, 1].item())

            print(f"[DBG_DIR t={self.t:05d}] pos_w=({px:+.3f},{py:+.3f}) "
                  f"v_w=({vwx:+.3f},{vwy:+.3f}) v_b=({vbx:+.3f},{vby:+.3f})")



        # Update most recent ground contact height for each leg
        with torch.no_grad():
            p_foot = self.foot_positions()              # (B,4,3)
            c = self.contact_flags().squeeze(-1)        # (B,4)
            prev_c = getattr(self, "prev_contact_flags", None)
            if prev_c is None:
                self.prev_contact_flags = c.unsqueeze(-1).clone()
                prev_c = self.prev_contact_flags

            # Touchdown is any contact rising edge. Gating this on the stance
            # phase would miss scraping / early / terrain-bump touchdowns and
            # leave last_contact_* locked to stale positions (a stance-foot
            # lock at the wrong height produces forward-flip torque).
            touchdown = ((prev_c.squeeze(-1) <= 0.5) & (c > 0.5))  # (B,4)
            self.last_contact_xy = torch.where(
                touchdown.unsqueeze(-1),
                p_foot[..., 0:2],
                self.last_contact_xy
            )

            # ===== stride debug print (env0, per-leg touchdown-to-touchdown) =====
            b0 = 0
            if touchdown[b0].any():
                # Basic quantities: actual velocity, step frequency
                # Note: your cmd_rand here is in body frame
                v_body_x = float(self.base_lin_body[b0, 0].item())
                v_body_y = float(self.base_lin_body[b0, 1].item())
                f0 = float(self.step_freq_B[b0].item()) if hasattr(self, "step_freq_B") else float(self.cfg.step_freq)
                cmd0 = self.cmd_rand[b0].detach().cpu().numpy() if hasattr(self, "cmd_rand") else None

                # ===== yaw & world forward/left (env0) =====
                yaw0 = float(self.yaw[b0].item())  # env0 base yaw (rad), from _update_cache()
                cy0, sy0 = math.cos(yaw0), math.sin(yaw0)
                fwd_w = torch.tensor([cy0, sy0], device=self.device, dtype=torch.float32)      # (2,)
                left_w = torch.tensor([-sy0, cy0], device=self.device, dtype=torch.float32)    # (2,)

                # Optional: print yaw rate (body frame z)
                omega_z0 = float(self.base_ang_body[b0, 2].item())  # rad/s



                # Nominal stride (estimated from current f0 and actual v_body_x)
                stride_est = (v_body_x / (f0 + 1e-9))  # m

                leg_names = ["FL", "FR", "RL", "RR"]
                # ===== phase u (env0) for touchdown gating =====
                if hasattr(self, "leg_phase_offsets_B"):
                    phase_offsets = self.leg_phase_offsets_B
                else:
                    phase_offsets = self.leg_phase_offsets.view(1, 4).repeat(self.B, 1)

                phases = phase_offsets + self.phase.view(self.B, 1)    # (B,4)
                u0 = self.gait._phase_u(phases)[b0]                         # (4,) in [0,1)
                u_eps = 0.08

                for leg in range(4):
                    if bool(touchdown[b0, leg].item()):
                        # Phase gating: only accept touchdowns near the phase
                        # boundary (u close to 0) to filter jitter / double hits.
                        if not (float(u0[leg].item()) < u_eps):
                            continue

                        # Cooldown between touchdowns of the same leg: half the
                        # current gait period.
                        T0 = 1.0 / max(f0, 1e-6)
                        min_steps = int(0.5 * T0 / self.cfg.dt)
                        if (self.t - int(self._stride_last_td_step0[leg].item())) < min_steps:
                            continue  # too close -> jitter trigger, ignore
                        self._stride_last_td_step0[leg] = self.t


                        xy = p_foot[b0, leg, 0:2].detach()  # (2,)
                        if self._stride_have_td0[leg]:
                            dxy = (xy - self._stride_last_td_xy0[leg])                    # (2,)
                            stride_xy  = torch.linalg.norm(dxy).item()                    # Euclidean distance
                            stride_fwd = torch.dot(dxy, fwd_w).item()                     # Forward projection (most critical)
                            stride_lat = torch.dot(dxy, left_w).item()                    # Lateral drift (judge side drift)

                            self._stride_count0[leg] += 1

                            if stride_fwd > 0.45 or stride_xy > 0.60:
                                # Abnormal: don't print, don't count, but update cache to current point to prevent subsequent chain explosion
                                self._stride_last_td_xy0[leg] = xy
                                continue
                            # Control screen flooding: stop printing after limit (can disable)
                            if (self._stride_print_limit is None) or (int(self._stride_count0[leg].item()) <= int(self._stride_print_limit)):
                                print(
                                    f"[STRIDE env0 t={self.t:05d}] {leg_names[leg]} "
                                    f"stride_xy={stride_xy:.3f}  fwd={stride_fwd:.3f}  lat={stride_lat:.3f} | "
                                    f"v_body=({v_body_x:+.2f},{v_body_y:+.2f}) f={f0:.3f} v/f≈{stride_est:.3f} | "
                                    f"yaw={yaw0:+.2f} wz={omega_z0:+.2f} | "
                                    f"cmd={np.round(cmd0,3) if cmd0 is not None else None}"
                                )
                        else:
                            # First touchdown: only record, don't print
                            self._stride_have_td0[leg] = True

                        # Update this leg's last touchdown point
                        self._stride_last_td_xy0[leg] = xy

            self.last_contact_z = torch.where(touchdown, p_foot[..., 2], self.last_contact_z)

            # Update prev_contact
            self.prev_contact_flags = c.unsqueeze(-1).clone()
        self.t += 1
        self.ep_len_buf += 1

        # ===== Initial forward flip localization print (only watch env0) =====
        if DBG_INIT_FALL and (self.t <= DBG_INIT_FALL_STEPS) and (self.t % DBG_INIT_FALL_EVERY == 0):
             b = int(DBG_INIT_FALL_ENV)
             try:
                 beta_B, min_feet_B, allow_aerial_B = self.gait._get_beta_minfeet_allow_aerial()
                 beta0 = float(beta_B[b].item())
                 k0 = int(min_feet_B[b].item())
                 allow0 = bool(allow_aerial_B[b].item()) if torch.is_tensor(allow_aerial_B) else bool(allow_aerial_B)

                 # phases -> original phase stance (no topk) vs mix stance (with topk filling)
                 if hasattr(self, "leg_phase_offsets_B"):
                     phase_offsets = self.leg_phase_offsets_B
                 else:
                     phase_offsets = self.leg_phase_offsets.view(1, 4).repeat(self.B, 1)
                 phases = phase_offsets + self.phase.view(self.B, 1)   # (B,4)
                 u = self.gait._phase_u(phases)                              # (B,4)
                 raw_phase_stance = (u[b] < beta_B[b].view(1)).float()  # (4,)
 
                 stance_mix = self.gait._mix_stance(
                     phases=phases[b:b+1],
                     contact_flags=self.contact_flags().detach()[b:b+1],
                     beta_B=beta_B[b:b+1],
                     min_feet_B=min_feet_B[b:b+1],
                     w_phase=1.0,
                     w_contact=0.0,
                 ).squeeze(0).squeeze(-1)                               # (4,)
 
                 forced = (raw_phase_stance.sum() < float(k0))

                 # Basic state
                 cmd0 = self.cmd_rand[b].detach().cpu().numpy() if hasattr(self, "cmd_rand") else None
                 sf0 = float(getattr(self, "step_freq_B", torch.tensor([self.cfg.step_freq], device=self.device))[b].item())
                 sh0 = float(getattr(self, "swing_height_B", torch.tensor([self.cfg.swing_height], device=self.device))[b].item())
                 c0 = self.contact_flags().detach()[b].squeeze(-1).cpu().numpy()
                 z_foot0 = p_foot[b, :, 2].detach().cpu().numpy()
                 z_lc0 = self.last_contact_z[b].detach().cpu().numpy()
 
                 print(
                     f"[DBG_INIT_FALL t={self.t:04d}] "
                     f"gait={int(self.gait_ids[b].item()) if hasattr(self,'gait_ids') else -1} "
                     f"beta={beta0:.2f} minFeet={k0} allowAerial={allow0} forcedTopK={bool(forced)} | "
                     f"cmd(vx,vy,yaw)={np.round(cmd0,3) if cmd0 is not None else None} "
                     f"f={sf0:.2f}Hz hSwing={sh0:.3f} | "
                     f"z={float(self.base_pos[b,2].item()):.3f} "
                     f"roll={float(self.roll[b].item()):+.3f} pitch={float(self.pitch[b].item()):+.3f} | "
                     f"contact={np.round(c0,0).astype(int).tolist()} "
                     f"stanceRaw={np.round(raw_phase_stance.cpu().numpy(),0).astype(int).tolist()} "
                     f"stanceMix={np.round(stance_mix.cpu().numpy(),0).astype(int).tolist()} | "
                     f"zFoot={np.round(z_foot0,3).tolist()} zLast={np.round(z_lc0,3).tolist()}"
                 )
             except Exception:
                 # Never let a debug print break training.
                 pass

        fallen_height = (self.base_pos[:, 2] < 0.16)           # (B,)
        fallen_tilt   = (torch.abs(self.roll) > 0.9) | (torch.abs(self.pitch) > 0.9)
        done = fallen_height | fallen_tilt                     # (B,)

        # Episode timeout (Rudin dynamic curriculum only). Robots that neither fell nor finished are
        # reset after max_episode_length steps so the curriculum can promote them. Kept all-False for
        # flat/rough so plain-ground DiffSim training is unchanged (no timeouts, reset on fall only).
        if self.cfg.terrain_type == "rudin" and self.cfg.rudin_terrain.dynamic_curriculum:
            timed_out = self.ep_len_buf > self.max_episode_length      # (B,)
        else:
            timed_out = torch.zeros(self.B, dtype=torch.bool, device=self.device)

        obs = self.get_obs()
        extra = {
            "done": done,
            "timeout": timed_out,
            "muN": torch.ones(self.B, device=self.device),
            "q_err_norm": torch.zeros(self.B, device=self.device),
        }
        q_now_sim = self.q[:, self.ctrl_idx_t]                      # sim convention
        q_now_pol = q_now_sim * ctrl_sign12                         # sim -> policy convention
        q_err = q_ref12 - q_now_pol                                 # both sides in policy convention

        return obs, extra, q_err, q_ref12

    @torch.no_grad()
    def get_obs(self):
        """Assemble the policy observation from the current Isaac state.

        Layout (36-D): 3 command + 8 phase sin/cos + 3 body linear velocity +
        4 base quaternion (wxyz) + 3 body angular velocity + 12 joint deltas
        from the default posture + 3 body-frame gravity projection. With
        ``cfg.use_height_obs`` the 187-point height scan is appended (see
        ``obs_dim``). Deliberately grad-free: observations are input leaves;
        the gradient path into the policy runs through the SRBD losses.

        Returns:
            (B, obs_dim) observation tensor.
        """
        cfg, dev = self.cfg, self.device
        B = self.B

        # cmd: [vx_cmd, vy_cmd, yaw_rate_cmd] directly from high-level command sampled at reset
        if hasattr(self, "cmd_rand"):
            cmd = self.cmd_rand.clone()                       # (B,3)
        else:
            # Safety fallback: compatible writing when only vx_star exists
            cmd = torch.stack([
                self.vx_star,
                torch.zeros_like(self.vx_star),
                torch.zeros_like(self.vx_star)
            ], dim=1)

        # Per-leg phases from the multi-gait phase table.
        if hasattr(self, "leg_phase_offsets_B"):
            phase_offsets = self.leg_phase_offsets_B                         # (B,4)
        else:
            phase_offsets = self.leg_phase_offsets.view(1,4).repeat(B,1)     # (B,4)

        phases = phase_offsets + self.phase.view(B,1)  # (B,4)

        sincos = torch.stack([torch.sin(phases), torch.cos(phases)], dim=2).reshape(B, 8)

        v_b = self.base_lin_body                             # (B,3) body-frame linear velocity
        q_wxyz = self.base_quat                              # (B,4)
        w_b  = self.base_ang_body                            # (B,3)

        # Gravity projection (B,3): g_body = R(q)^T @ g_world, which is exactly
        # what the batched quat_rotate_inverse_wxyz computes -> no per-env loop.
        g_w = torch.tensor([0.0, 0.0, -cfg.g], dtype=torch.float32, device=dev).view(1, 3).expand(B, 3)
        g_proj = quat_rotate_inverse_wxyz(q_wxyz, g_w, dev)  # (B,3)


        q_now_sim = self.q[:, self.ctrl_idx_t].to(dev)      # sim convention
        ctrl_sign12 = self.ctrl_sign.view(1,12).to(dev)

        q_now_pol = q_now_sim * ctrl_sign12                 # -> policy convention
        q_default_pol = torch.from_numpy(self.q_default_np).to(dev).view(1,12)

        q_delta = q_now_pol - q_default_pol                 # This is the paper's (q - q_default)

        # Total dimension: 3(cmd) + 8(phase) + 3(v_b) + 4(q) + 3(w_b) + 12(q_delta) + 3(g_proj) = 36
        obs = torch.cat([cmd, sincos, v_b, q_wxyz, w_b, q_delta, g_proj], dim=-1)  # (B,36)

        # Optional Rudin-style height scan: 187 local terrain heights relative to the
        # base (hm_subtract_base_z=True), centred on the nominal height and clipped to
        # +-1 m so it is roughly zero-mean when standing on flat ground. Sampled at the
        # Isaac root state -- the obs is an input leaf (whole method is no_grad); the
        # gradient path to the terrain lives in the loss terms, not here.
        if getattr(cfg, "use_height_obs", False):
            root = self.root_state
            qx, qy, qz, qw = root[:, 3], root[:, 4], root[:, 5], root[:, 6]  # xyzw
            yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            hm = self.perception.height_sampler.sample(root[:, 0:3], yaw)    # (B, n_pts)
            obs = torch.cat([obs, torch.clip(hm - cfg.h0, -1.0, 1.0)], dim=-1)

        return obs

# ---------------- Training ----------------
