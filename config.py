from dataclasses import dataclass

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

# Note: terrain selection moved into EnvCfg.use_complex_terrain (flat field) so all
# randomization switches live in one place. See EnvCfg below.

# ===============================
# ⭐ SRBD CUDA Kernel Switch
# ===============================
CUDA_KERNEL_SRBD = True
# True  = Use custom CUDA kernel for _srbd_step (requires: python setup.py build_ext --inplace)
# False = Use PyTorch implementation (default, always works)


@dataclass
class EnvCfg:
    # physics
    g: float = 9.81
    h0: float = 0.35   # 0.30
    dt: float = 0.002  # 500 Hz

    # control
    use_gpu_pipeline: bool = True
    use_viewer: bool = False  # Render during training? True - render but much slower; False - no render
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


    # contact & friction
    contact_thresh_n: float = 8.0
    mu_tangent: float = 0.6
    fz_min: float = 20.0
    fz_max: float = 250.0

    # Number of parallel environments = number of parallel robots
    num_envs: int = 16

    # terrain (one shared surface for all robots; see env._terrain_height / _sample_spawn_xy)
    use_complex_terrain: bool = False   # True = random rough heightfield; False = flat ground plane
    rand_spawn_xy: bool = False         # True = re-scatter each robot's (x,y) across terrain on reset
    spawn_area_half_m: float = 8.0      # half-extent (m) of the scatter region (well within terrain bounds)

    cmd_deadzone: float = 0.05   # m/s, threshold for “stop”

    contact_on_n: float  = 20.0
    contact_off_n: float = 10.0

    delta_q_scale12 = (0.10, 0.30, 0.30) * 4
