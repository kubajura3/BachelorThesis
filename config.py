from dataclasses import dataclass, field

# PerceptionCfg is lightweight (stdlib dataclass only); importing it here does
# NOT pull in torch/warp -- those load lazily only when use_perception is enabled.
from perception.config import PerceptionCfg

# ===============================
# ⭐ Pure Paper Mode Switch
# ===============================
PURE_PAPER_MODE = True
# True  = Pure paper version (no engineering tricks)
# False = Engineering version (with initial velocity, action smoothing, etc.)

# ===============================
# ⭐ Initial Fall Debug Print Switch (only print first N steps of env0)
# ===============================
DBG_INIT_FALL = True
DBG_INIT_FALL_STEPS = 250     # Only print first N env.step() calls (t starts from 0 after each reset)
DBG_INIT_FALL_EVERY = 5       # Print every N steps
DBG_INIT_FALL_ENV = 0         # Only watch which robot (env index)

# ⭐ Whether to only reset at iter=0
ONLY_ITERATE_NO_RESET = True
# True: no reset
# False: reset

# Note: terrain selection lives in EnvCfg.terrain_type ("flat" | "rough" | "rudin") so all
# randomization switches live in one place. See EnvCfg and RudinTerrainCfg below.

# ===============================
# ⭐ SRBD CUDA Kernel Switch
# ===============================
CUDA_KERNEL_SRBD = True
# True  = Use custom CUDA kernel for _srbd_step (requires: python setup.py build_ext --inplace)
# False = Use PyTorch implementation (default, always works)


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


    alpha_align: float = 0.9
    use_strict_alpha_align: bool = True

    # termination
    term_penalty: float = 200.0

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
    # Lateral velocity and yaw rate command range (default all 0, change later if you want lateral movement / turning)
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

    delta_q_scale12 = (0.10, 0.30, 0.30) * 4
