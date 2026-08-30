import os
from dataclasses import dataclass, field

# PerceptionCfg is lightweight (stdlib dataclass only); importing it here does
# NOT pull in torch/warp -- those load lazily only when use_perception is enabled.
from perception.config import PerceptionCfg

def resolve_cmd_style(cfg):
    """Which velocity-command distribution a config selects: "rudin" | "rand" | "fixed".

    ``cfg.cmd_style`` pins it explicitly; ``None`` (the default) falls back to the
    terrain-derived choice ``_sample_command`` used before that knob existed, so every run
    predating it reproduces exactly.

    Module-level on purpose: ``train.py`` needs the same answer to decide whether a run carries
    a live yaw command, and resolving it independently there is precisely how ``blind_omni``
    ended up drawing Rudin yaw commands while the loss still measured yaw against a fixed reset
    heading (CAMPAIGN_FINDINGS.md 19.9).
    """
    style = getattr(cfg, "cmd_style", None)
    if style is not None:
        return style
    if getattr(cfg, "terrain_type", "flat") == "rudin":
        return "rudin"
    return "rand" if cfg.rand_cmd else "fixed"


def heading_command_active(cfg):
    """True when the yaw command is derived from a target heading rather than held constant.

    Gated on the "rudin" command style as well as the flag: the forward-only and rand styles
    have no yaw command to redefine.
    """
    return bool(getattr(cfg, "heading_command", False)) and resolve_cmd_style(cfg) == "rudin"


# ===============================
# Pure Paper Mode Switch
# ===============================
PURE_PAPER_MODE = True
# True  = Pure paper version (no engineering tricks)
# False = Engineering version (with initial velocity, action smoothing, etc.)

def _flag(name, default):
    """Read a boolean switch from the environment, falling back to `default`.

    Env vars are strings, and bool("0") is True, so the value is parsed
    explicitly. Unset -> `default`, i.e. exactly the literal written below, so a
    plain `python train.py` behaves identically to before these overrides existed.
    Set before the process starts (e.g. `DEBUG_TRAIN=1 python train.py`), because
    config.py is imported at module scope, long before any argument parsing.
    """
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


# ===============================
# Verbose Training Diagnostics Switch  (env var: DEBUG_TRAIN=1)
# ===============================
DEBUG_TRAIN = _flag("DEBUG_TRAIN", False)
# Gates the per-iteration / per-step diagnostic output that forces GPU->CPU
# synchronisation inside the hot loop: the per-parameter gradient dump and the
# per-env velocity print in train.py, and the per-leg stride print in env.step().
# Off by default -- those syncs are fixed overhead per iteration, so they inflate
# the small-batch end of any throughput measurement. Turn on for debugging a
# single short run; keep off for training campaigns and benchmarks.

# ===============================
# Initial Fall Debug Print Switch (only print first N steps of env0)
# ===============================
DBG_INIT_FALL = True
DBG_INIT_FALL_STEPS = 250     # Only print first N env.step() calls (t starts from 0 after each reset)
DBG_INIT_FALL_EVERY = 5       # Print every N steps
DBG_INIT_FALL_ENV = 0         # Only watch which robot (env index)

# ===============================
# Fall-rule diagnostic overrides  (env vars: FALL_H_THRESH=<m>, DIAG_NO_TERM=1)
# ===============================
FALL_H_THRESH = float(os.getenv("FALL_H_THRESH", "0.16"))
DIAG_NO_TERM = _flag("DIAG_NO_TERM", False)
# Unset -> exactly the literals that used to be hard-coded in env.step(), so a plain
# run is unchanged. They exist for one diagnostic: on Rudin terrain the base height
# falls 0.35 -> 0.16 in ~0.19 s (within 3% of free fall) and the robot is terminated
# before it ever lands, so we never see the height it would have settled at -- and that
# height is what separates "spawned above the collision mesh" from "legs collapsing".
# DIAG_NO_TERM=1 disables BOTH halves of the rule (height and tilt) so the run keeps
# falling and the answer becomes readable. Never set either for a real training run.

# ===============================
# Soft tilt barrier  (env vars: TILT_W=<weight>, TILT_ON=<radians>)
# ===============================
TILT_W = float(os.getenv("TILT_W", "0.0"))
TILT_ON = float(os.getenv("TILT_ON", "0.6"))
# Step 1(c): the differentiable stand-in for `term_penalty`, which only ever reached
# episodic_reward and was never backpropagated -- so the objective said nothing about
# falling over while every termination on the Rudin curriculum was a tilt fall
# (CAMPAIGN_FINDINGS.md 18.4, 20.8). train.py hinges relu(cos(TILT_ON) - cos(tilt))^2 off
# the gravity projection it already computes. TILT_W = 0 leaves the loss expression
# untouched, so a plain run is unchanged, and the term's raw value is still logged --
# which is what lets a weight be calibrated from a run that did not use one.
# TILT_ON = 0.6 rad (34 deg) sits above the ~16 deg lean the trained policy walks with, so
# the term is inert in normal walking rather than a re-weighting of loss_gproj, and below
# the 0.9 rad (51.6 deg) fall_tilt_thresh, so it engages while righting is still possible.
# Deliberately env-var-only and NOT in train.py's MODE_CFG: a mode dict is applied after
# EnvCfg() and would silently win over the variable.

# Whether to only reset at iter=0
ONLY_ITERATE_NO_RESET = True
# True: no reset
# False: reset

# Note: terrain selection lives in EnvCfg.terrain_type ("flat" | "rough" | "rudin") so all
# randomization switches live in one place. See EnvCfg and RudinTerrainCfg below.

# ===============================
# SRBD CUDA Kernel Switch
# ===============================
CUDA_KERNEL_SRBD = _flag("CUDA_KERNEL_SRBD", True)
# True  = Use custom CUDA kernel for _srbd_step (requires: python setup.py build_ext --inplace)
# False = Use PyTorch implementation (default, always works)
# Env var CUDA_KERNEL_SRBD=0/1 overrides the literal above, so the PyTorch-vs-CUDA
# speed comparison needs no source edit between runs. srbd.py reads this at import
# time, which is why it has to be an environment variable and not a CLI flag.

# ===============================
# Ground-Reaction-Force Solver Precision  (env var: FORCE_DTYPE=fp32|fp64)
# ===============================
FORCE_DTYPE = os.getenv("FORCE_DTYPE", "fp32").strip().lower()
# Precision of the damped-pseudo-inverse solve in env.estimate_foot_forces, which
# runs once per physics step (24 per training iteration) regardless of the SRBD
# backend -- the fused CUDA kernel is pure fp32 and does not touch it.
#
# fp32 (default) is what the inherited implementation (commit 1731453) used, and it
# is what ships now. fp64 was adopted for a while together with the batched rewrite
# (commit 7a82eca) so that batched and per-env-loop results agreed tightly enough
# for a strict parity assertion -- a *testing* decision, not a physics one -- and it
# is expensive: consumer GPUs run fp64 at 1/32 of fp32 (e.g. GTX 1660 Ti), and this
# solve happens 24 times per training iteration.
#
# The accuracy cost is negligible. clamp(S, min=1e-3) on the singular values caps
# the effective condition number of the applied inverse, so the fp32-vs-fp64
# difference measures ~1e-5 N (tests/test_vectorization.py prints it) against foot
# forces of order 10-100 N that are then clamped to [20, 250] N and friction-cone
# projected.
#
# fp64 remains available for verification: tests/test_vectorization.py uses it as
# the reference both for the vectorisation parity check and for bounding the fp32
# error. Never change this midway through a training campaign -- it changes the
# numbers the policy trains on.


@dataclass
class RudinTerrainCfg:
    """Mirror of Rudin et al. legged_gym terrain config (defaults match the paper code).

    Used when EnvCfg.terrain_type == "rudin" to build the curriculum-grid landscape
    (rows = increasing difficulty, columns = terrain-type variety). See terrain.Terrain.
    """
    mesh_type: str = "trimesh"          # only "trimesh" is supported here
    horizontal_scale: float = 0.1       # [m] heightfield pixel resolution
    vertical_scale: float = 0.005       # [m] per height unit
    border_size: float = 25.0           # [m] flat border around the grid
    curriculum: bool = True             # rows = increasing difficulty
    selected: bool = False              # use a single hand-picked terrain instead of the grid
    terrain_kwargs: object = None       # kwargs for the selected terrain (when selected=True)
    terrain_length: float = 8.0         # [m] cell size along x (difficulty axis)
    terrain_width: float = 8.0          # [m] cell size along y (type axis)
    num_rows: int = 10                  # difficulty levels
    num_cols: int = 20                  # terrain-type columns
    # [smooth slope, rough slope, stairs, discrete obstacles, stepping stones]
    terrain_proportions: list = field(default_factory=lambda: [0.1, 0.1, 0.35, 0.25, 0.2])
    slope_treshold: float = 0.75        # (sic) slopes above this become vertical in trimesh
    max_init_terrain_level: int = 5     # difficulty ceiling for initial placement
    # Dynamic "game-inspired" curriculum (Rudin _update_terrain_curriculum): promote a robot one
    # difficulty row when it walks > terrain_length/2 from its cell origin, demote it when it covers
    # less than half its commanded distance, recycle robots that clear the hardest row. Runs on every
    # per-env reset (fall or episode timeout). True = Rudin's behaviour; False = static placement only
    # (the previous integration's behaviour). Only active when EnvCfg.terrain_type == "rudin".
    dynamic_curriculum: bool = True


@dataclass
class EnvCfg:
    # physics
    g: float = 9.81
    h0: float = 0.35   # 0.30
    dt: float = 0.002  # 500 Hz

    # control
    use_gpu_pipeline: bool = True
    use_viewer: bool = False  # Render during training? True - render but much slower; False - no render

    # perception (terrain vision) -- gated; blind training is unchanged when False.
    # Gathers a forward depth image + a differentiable local height map each step
    # (see perception/ and env.collect_perception). use_gpu_pipeline=True (the default)
    # keeps the pose input zero-copy.
    use_perception: bool = False
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    # Append the 187-point local height map to the observation (obs 36 -> 36+187).
    # Requires use_perception=True. Policies must be built with dim_obs=env.obs_dim.
    use_height_obs: bool = False
    # Enable the terrain-aware differentiable loss terms in train.py (terrain-relative
    # height + swing-foot clearance, sampled at SRBD-predicted positions so the terrain
    # slope back-propagates into the policy). Requires use_perception=True.
    use_terrain_loss: bool = False
    # Vision-policy path in train.py: the depth camera image is encoded by a CNN and
    # fed to the policy alongside the 36-D proprio obs (obs itself stays 36-D; depth is
    # a separate input). Requires use_perception=True; mutually exclusive with
    # use_height_obs (a policy takes either the flat obs or obs+depth, not both).
    use_depth_obs: bool = False
    action_hold: int = 5      # 100 Hz control

    pd_kp: float = 60      # 60
    pd_kd: float = 2       # 2
    q_default: tuple = (0.0, 0.9, -1.20)
    # q_default: tuple = (0.0, 0.8, -1.50)

    # gait
    use_paper_raibert: bool = True # True=original paper formula, False=engineering enhanced version (body vx tracking)

    step_freq: float = 1.6 # 1.5 seems to be optimal

    # Random step frequency switch - True: on; False: off (fixed step_freq)
    rand_step_freq: bool = False       # True: sample step_freq_B on reset/reset_envs; False: use constant step_freq
    step_freq_min: float = 1.0     # 1.4
    step_freq_max: float = 4.0     # 3.2

    # Step frequency from velocity command switch - True: on; False: off (fixed step_freq_B = 2.2)
    step_freq_from_cmd: bool = False  # True: override step_freq_B each step from |vx_star|

    # Raibert parameters
    # swing_height: float = 0.012        # Fixed swing height: 0.025
    k_raibert: float = 0
    x_bias: float = 0.0

    swing_height: float = 0.12
    swing_height_max: float = 0.12

    raibert_fb_clip: float = 0.15       # m, touchdown feedback term clipping per axis

    # Training warm-up: disable aerial phase first, enable after stable (bound/gallop/running-trot)
    train_no_aerial: bool = True

    # gait selection switch:
    # -1: random gait per env (stand / trot / pace / bound / gallop)
    #  0: stand
    #  1: trot
    #  2: pace
    #  3: bound
    #  4: gallop
    gait_mode: int = 1
    # When gait_mode < 0 (random per env), sample only from these gait ids.
    # e.g. (1, 2, 3, 4) excludes "stand"; (0,1,2,3,4) allows all.
    gait_choices: tuple = (0, 1, 2, 3, 4)
    trot_style: str = "normal"   # "normal" | "walk" | "run"
    # trot_style = "normal"  # β=0.5 (Fig 9.2a)
    # trot_style = "walk"    # β=0.6 (Fig 9.2b)
    # trot_style = "run"     # β=0.4 (Fig 9.2c)

    z_time_constant: float = 0.08   # Stance foot height convergence time constant, smaller = faster to target height

    # settle steps
    settle_steps_init: int = 60
    settle_steps_reset: int = 40

    # SRBD params

    m: float = 7.0       # Close to URDF's 6.921
    Ixx: float = 0.024
    Iyy: float = 0.098
    Izz: float = 0.107

    # Go2 leg geometry used by the SRBD foot kinematics (srbd.foot_positions_srbd)
    # and the Raibert foothold planner (gait._raibert_touchdown_world).
    hip_offset_x: float = 0.1934   # [m] hip forward offset from the base origin
    hip_offset_y: float = 0.1420   # [m] hip lateral offset from the base origin
    leg_l1: float = 0.213          # [m] thigh link length
    leg_l2: float = 0.213          # [m] calf link length


    alpha_align: float = 0.9
    use_strict_alpha_align: bool = True

    # termination
    term_penalty: float = 200.0
    # Fall rule, previously hard-coded in env.step(). Same values, now in one place so a
    # diagnostic run can move them from the environment (see FALL_H_THRESH / DIAG_NO_TERM).
    fall_height_thresh: float = FALL_H_THRESH   # [m] base height above the ground beneath it
    fall_tilt_thresh: float = 0.9               # [rad] |roll| or |pitch|
    # Soft tilt barrier (Step 1c); defaults come from the TILT_W / TILT_ON env vars above.
    # tilt_w = 0 makes the term inert, tilt_on is the tilt angle where it starts pushing.
    tilt_w: float = TILT_W
    tilt_on: float = TILT_ON

    # Episode length (seconds of sim time). Only used by the Rudin dynamic curriculum: a robot that
    # neither falls nor finishes the episode is reset after this long, which is what lets the
    # promote/demote logic advance (matches Rudin's episode_length_s = 20). max_episode_length (in
    # env.step() units) is derived as ceil(episode_length_s / dt) in env.py. No effect on flat/rough.
    episode_length_s: float = 20.0

    #============================================
    # Random velocity command switch - True: on; False: off (use cmd_fixed)
    rand_cmd: bool = False       # True: sample cmd_B on reset/reset_envs; False: use constant cmd
    cmd_fixed: tuple = (0.5, 0.0, 0.0)   # (vx, vy, yaw_rate) used when rand_cmd=False
    # trot/pace    0.5 - 1 m/s
    # bound/gallop 1 - 2 m/s
    # Increase velocity command: previous 0.1-0.3 too slow, Raibert foothold displacement too small
    vx_min: float = +0.4   # Increase minimum velocity
    vx_max: float = +0.8   # Increase maximum velocity
    # Lateral velocity and yaw-rate command ranges (all 0 by default; widen to
    # enable lateral movement / turning in the random-command mode)
    vy_min: float = 0
    vy_max: float = 0
    yaw_min: float = 0        # [rad/s]
    yaw_max: float = 0
    #============================================
    # Rudin-matched omnidirectional command ranges, used ONLY when terrain_type == "rudin" (for the
    # fair DiffSim-vs-PPO comparison). The flat/rough path keeps the forward-only vx/vy/yaw_* above.
    # Mirrors legged_robot_config.commands.ranges (lin_vel_x/y = [-1,1] m/s, ang_vel_yaw = [-1,1] rad/s)
    # and the deadband that zeros tiny commands (legged_robot._resample_commands).
    rudin_cmd_lin_vel_x: tuple = (-1.0, 1.0)   # [m/s]
    rudin_cmd_lin_vel_y: tuple = (-1.0, 1.0)   # [m/s]
    rudin_cmd_ang_vel_yaw: tuple = (-1.0, 1.0) # [rad/s]
    rudin_cmd_deadband: float = 0.2            # zero (vx,vy) when |v_xy| < this (Rudin)

    # Command style, decoupled from terrain_type so the two can be varied independently. The
    # Exp-2 runs cannot say whether the fall rate comes from the rough ground or from the
    # omnidirectional command set, because picking "rudin" terrain also picks the rudin command
    # ranges -- this knob is what lets the two be measured apart (flat terrain + rudin commands).
    # None = follow terrain_type, which reproduces every previous run bit-for-bit; the explicit
    # values force one style regardless of the terrain.
    cmd_style: str = None                      # None | "rudin" | "rand" | "fixed"

    # Heading-based yaw command (legged_gym's commands.heading_command, which is True in every
    # config the PPO baseline actually runs: legged_robot_config.py:70, MGDP
    # random_dog_config_stage1.py:96 and stage2.py:103). legged_gym samples a *target heading*
    # and derives the yaw-rate command from the heading error every step
    # (legged_robot.py:327-330), so the command decays to zero once the robot faces its target.
    # This repo instead sampled a yaw *rate* and held it for the whole 20 s episode, which is a
    # strictly harder task than the baseline's -- an unintended mismatch in the Exp-1
    # DiffSim-vs-PPO comparison rather than a deliberate choice. Only active on the "rudin"
    # command style. Default False so every run already on disk reproduces bit-for-bit.
    heading_command: bool = False
    cmd_heading_range: tuple = (-3.141592653589793, 3.141592653589793)  # [rad], legged_gym's heading
    # Proportional gain turning heading error into a yaw-rate command. legged_gym hardcodes 0.5.
    heading_to_yaw_gain: float = 0.5


    # contact & friction
    contact_thresh_n: float = 8.0
    mu_tangent: float = 0.6
    fz_min: float = 20.0
    fz_max: float = 250.0

    # Number of parallel environments = number of parallel robots
    num_envs: int = 16

    # terrain (one shared surface for all robots; see env._terrain_height / _sample_spawn_xy)
    terrain_type: str = "flat"          # "flat" = ground plane | "rough" = random heightfield | "rudin" = curriculum grid
    rand_spawn_xy: bool = False         # True = re-scatter each robot's (x,y) across terrain on reset ("rough" mode)
    spawn_area_half_m: float = 8.0      # half-extent (m) of the scatter region (well within terrain bounds)
    # "rudin" mode: per-cell config + per-reset spawn jitter around the assigned grid origin
    rudin_terrain: RudinTerrainCfg = field(default_factory=RudinTerrainCfg)
    rudin_spawn_jitter_m: float = 1.0   # +-m jitter around the assigned cell origin (matches legged_gym)

    cmd_deadzone: float = 0.05   # m/s, threshold for “stop”

    contact_on_n: float  = 20.0
    contact_off_n: float = 10.0

    # Per-joint action scale (hip, thigh, calf) x 4 legs: policy outputs pass
    # through tanh and are multiplied by these before being added to q_default.
    delta_q_scale12: tuple = (0.10, 0.30, 0.30) * 4
