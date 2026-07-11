# Quadruped Robot Gait Training System

Isaac Gym-based quadruped robot (Unitree Go2) gait control training system using a differentiable SRBD (Single Rigid Body Dynamics) model and a neural network policy for end-to-end training — on flat ground (blind) and on hard terrain using terrain perception (privileged height scan or a depth-camera CNN).

## Project Structure

```
BachelorThesis/
├── config.py                    # Global configuration and environment parameters
├── utils_math.py                # Math utilities (quaternions, rotation matrices, etc.)
├── terrain.py                   # Terrain creation (flat / rough / Rudin curriculum grid)
├── policy.py                    # Neural network policies (MLP + depth-CNN VisionPolicy)
├── gait.py                      # Gait planner (GaitPlanner)
├── srbd.py                      # Differentiable rigid body dynamics (SRBDModel)
├── env.py                       # Isaac Gym simulation environment (RealQuadEnv)
├── train.py                     # Training main loop (blind / height / depth modes)
├── play_many_dog.py             # Policy playback script
├── evaluate_rudin_comparison.py # Deterministic eval vs Rudin's PPO baseline
├── setup.py                     # Build script for the SRBD CUDA extension
├── go2_description.urdf         # Unitree Go2 robot model
├── perception/                  # Terrain perception (depth camera + height sampler)
│   ├── config.py                #   PerceptionCfg (camera, height grid, noise)
│   ├── collector.py             #   PerceptionCollector: one collect() entry point
│   ├── warp_camera.py           #   Warp ray-cast depth camera (CUDA-graph captured)
│   ├── warp_kernels/cam_kernel.py #  depth kernel (vendored from MGDP, see Attribution)
│   ├── height_sampler.py        #   differentiable height map (grid_sample)
│   ├── terrain_mesh.py          #   terrain adapter + Warp mesh construction
│   ├── preprocessing.py         #   depth clip/resize/normalise/noise
│   └── visualize_perception.py  #   offline sanity-check renderer (no Isaac Gym)
├── tests/
│   ├── test_srbd_kernel.py      # CUDA-kernel vs PyTorch parity (needs GPU + built ext)
│   ├── test_vectorization.py    # loop-vs-batched parity (CPU)
│   ├── test_perception_grad.py  # differentiable terrain sampling + gradients (CPU)
│   └── test_vision_policy.py    # VisionPolicy shapes/gradients/TorchScript (CPU)
└── src/
    ├── srbd_ext.cpp             # pybind11 bindings for the CUDA kernel
    └── srbd_cuda.cu             # Fused CUDA kernel: foot FK + SRBD dynamics step
```

## Features

### Supported Gaits
- **Stand**: All four legs synchronized
- **Trot**: Diagonal gait (FL+RR, FR+RL)
- **Pace**: Lateral gait (FL+RL, FR+RR)
- **Bound**: Front and rear legs synchronized
- **Gallop**: Four legs with sequential phase increments

### Core Technologies
- **SRBD Model**: Simplified single rigid body dynamics for differentiable physics simulation
- **α-Alignment Mechanism**: Blends real physics and SRBD predictions (default α=0.9)
- **Raibert Foothold Planning**: Adaptive foothold calculation based on velocity feedback
- **Multi-Environment Parallel Training**: Supports training multiple robots simultaneously (default 16, scales to ~1000+)
- **Domain Randomization** (optional, independent per-env switches): velocity command (`rand_cmd`), gait (`gait_mode = -1` + `gait_choices`), step frequency (`rand_step_freq`), and terrain + spawn placement (`terrain_type`, `rand_spawn_xy`). All robots share one terrain surface and spawn at the correct local terrain height under their own (x, y)
- **GPU Acceleration**: Uses Isaac Gym's GPU physics pipeline
- **Custom CUDA Kernel for SRBD** (optional): Fused kernel that replaces the per-step PyTorch ops with a single launch; toggled via `CUDA_KERNEL_SRBD` in `config.py`. Backward compatibility with autograd is preserved (see [SRBD CUDA Kernel](#srbd-cuda-kernel-optional))
- **Terrain Perception** (optional, `use_perception`): a forward Warp ray-cast depth camera and a differentiable 187-point local height scan gathered every step (`env.collect_perception()`), all GPU-resident and CUDA-graph captured
- **Three training modes** (one switch, see [Training](#training)): blind 36-D observation on flat ground; privileged height-scan observation (36+187-D) on the Rudin curriculum terrain; or a depth-camera **VisionPolicy** whose CNN encoder is trained end-to-end through the SRBD rollout

### Loss Functions
Training uses a weighted combination of multiple losses:
- `loss_v`: Velocity tracking (vx, vy)
- `loss_h`: Height maintenance (target 0.35 m; terrain-*relative* in the perception modes, sampled differentiably at the SRBD position so terrain slope back-propagates)
- `loss_omega`: Angular velocity regularization + yaw-rate tracking
- `loss_ctrl`: Control input regularization
- `loss_gproj`: Gravity projection (keep body level)
- `loss_foot`: Foot position tracking
- `loss_clear`: Swing-foot terrain clearance (perception modes only): penalises swing feet below terrain + margin, sampled at differentiable SRBD foot positions on a Gaussian-blurred heightfield
- `loss_yaw`: Yaw angle maintenance

## Requirements

### Dependencies
- Python 3.8+
- PyTorch 1.10+ (built with CUDA support)
- Isaac Gym Preview 4
- NumPy
- Matplotlib
- tqdm
- warp-lang (only for the perception / vision modes: `pip install warp-lang`)

### Optional (only for the custom SRBD CUDA kernel)
- CUDA Toolkit matching your PyTorch build (`nvcc --version` must work)
- A C++ toolchain (gcc/clang on Linux, MSVC on Windows) — the same one PyTorch was built against

### Installing Isaac Gym
```bash
# Download Isaac Gym Preview 4
# Extract and enter directory
cd isaacgym/python
pip install -e .
```

⚠️ **Important**: Isaac Gym must be imported before PyTorch. This is already handled in the code.

## Usage

### Training

```bash
# Blind baseline: flat ground, 36-D observation (1000 iterations, 24 steps each)
python3 train.py

# Stage 1: Rudin curriculum terrain + privileged height-scan observation (36+187-D)
#          + terrain-aware differentiable losses. Outputs get the "_height" tag.
PERCEPTION_TERRAIN=1 python3 train.py

# Stage 2: Rudin terrain + VisionPolicy (36-D obs + depth image through a CNN,
#          trained end-to-end through the SRBD rollout). Outputs get "_vision".
PERCEPTION_TERRAIN=depth python3 train.py

# Files generated in results/ (tag = "", "_height" or "_vision" by mode):
# - results/quad_diffsim_srbd_align_multi_robot<tag>.pth  (model weights)
# - results/quad_diffsim_srbd_align_multi_robot<tag>.pt   (TorchScript export)
# - results/ training curves (loss_*<tag>.png, vx_curve_*<tag>.png, ...)
```

### Evaluation (comparison against Rudin's PPO baseline)

Deterministic rollout on the Rudin curriculum terrain, reporting the metrics
legged_gym logs (`mean_terrain_level`, tracking errors, fall rate, ...):

```bash
python3 evaluate_rudin_comparison.py --obs-mode blind  --weights results/quad_diffsim_srbd_align_multi_robot.pth         --tag diffsim_blind
python3 evaluate_rudin_comparison.py --obs-mode height --weights results/quad_diffsim_srbd_align_multi_robot_height.pth  --tag diffsim_height
python3 evaluate_rudin_comparison.py --obs-mode depth  --weights results/quad_diffsim_srbd_align_multi_robot_vision.pth  --tag diffsim_vision
```

`--obs-mode` must match how the weights were trained. Results are written to
`results/eval_results_<tag>.json/.csv`.

### Perception sanity check (no Isaac Gym required)

```bash
python3 -m perception.visualize_perception                    # synthetic scene, auto device
python3 -m perception.visualize_perception --device cpu       # CPU-only
```

### Playing Trained Policy

```bash
# Default playback (4 dogs, random velocity commands)
python3 play_many_dog.py

# 16 dogs running together
python3 play_many_dog.py --num_envs 16

# Fixed velocity command (0.5 m/s)
python3 play_many_dog.py --no_rand_cmd

# Specify gait (1=trot)
python3 play_many_dog.py --gait_mode 1

# Specify weight file
python3 play_many_dog.py --weights your_model.pth

# Run for specified steps then stop
python3 play_many_dog.py --max_steps 1000
```

## Configuration

Main configuration in `EnvCfg` class in `config.py`:

### Physics Parameters
- `g = 9.81`: Gravity acceleration
- `h0 = 0.35`: Target height (meters)
- `dt = 0.002`: Simulation timestep (500 Hz)

### Control Parameters
- `action_hold = 5`: Control frequency (100 Hz)
- `pd_kp = 60`: PD controller proportional gain
- `pd_kd = 2`: PD controller derivative gain

### Gait Parameters
- `step_freq = 1.6`: Step frequency (Hz) — used as the constant cadence when the two switches below are off
- `rand_step_freq = False`: Randomize step frequency per env in `[step_freq_min, step_freq_max]`; constant `step_freq` when off
- `step_freq_from_cmd = False`: Derive step frequency from `|velocity command|` instead (fixed-stride mode); overrides `step_freq` each step when on
- `swing_height = 0.12`: Swing height (meters)
- `gait_mode = 1`: Gait mode (-1=random per-env, 0=stand, 1=trot, 2=pace, 3=bound, 4=gallop)
- `gait_choices = (0,1,2,3,4)`: Gait ids eligible when `gait_mode < 0` (e.g. `(1,2,3,4)` excludes stand)
- `rand_cmd = False`: Randomize velocity command per env; `cmd_fixed = (vx,vy,yaw)` is used when off

### Terrain & placement (all robots share one terrain surface)
- `terrain_type = "flat"`: `"flat"` = ground plane, `"rough"` = random rough heightfield,
  `"rudin"` = Rudin et al. curriculum-grid landscape (rows = increasing difficulty, columns = terrain type)
- `rand_spawn_xy = False` (`"rough"` mode): `True` = re-scatter each robot's (x,y) across the terrain on every reset;
  robots spawn at the local terrain height under their own (x,y) either way
- `spawn_area_half_m = 8.0`: half-extent (m) of the scatter region (kept well within the terrain bounds)
- `rudin_terrain` (`"rudin"` mode): grid config mirroring legged_gym (`num_rows`, `num_cols`,
  `terrain_proportions`, `border_size`, `max_init_terrain_level`, …); robots are placed on the grid like
  Rudin (random difficulty level ≤ `max_init_terrain_level`, terrain type spread across columns)
- `rudin_spawn_jitter_m = 1.0`: ±m jitter around each robot's assigned cell origin on reset (matches legged_gym)

> Note (`"rudin"` mode): use a large `num_envs` (hundreds–thousands). The grid spreads robots one
> column at a time (`terrain_types = arange(num_envs) // (num_envs / num_cols)`), so with
> `num_envs < num_cols` (e.g. default 16 vs 20) most columns stay empty — Rudin's scheme only fills
> the landscape at the massively-parallel batch sizes it was designed for.

### Training Parameters
- `num_envs = 16`: Number of parallel environments
- `alpha_align = 0.9`: SRBD alignment coefficient
- `train_no_aerial = True`: Disable aerial phase during training (warm-up)

### Perception & terrain-gradient switches (EnvCfg)
- `use_perception = False`: build the terrain-perception collector (depth camera + height sampler); required by the three flags below
- `use_height_obs = False`: append the 187-point height scan to the observation (obs 36 -> 223)
- `use_depth_obs = False`: vision-policy mode — depth image is a separate CNN input (obs stays 36-D); mutually exclusive with `use_height_obs`
- `use_terrain_loss = False`: terrain-relative height loss + swing-foot clearance loss, sampled differentiably at SRBD-predicted positions
- `perception`: a `PerceptionCfg` with camera intrinsics/mounting, height-grid extent and the loss-field blur (`hm_loss_blur_cells`)

The `PERCEPTION_TERRAIN` environment variable in `train.py` sets these
consistently per mode, so they rarely need to be touched by hand.

### Global Switches
- `PURE_PAPER_MODE = True`: Pure paper version (no engineering tricks)
- `ONLY_ITERATE_NO_RESET = True`: Only reset on first iteration
- `CUDA_KERNEL_SRBD` (default `True` in `config.py`): Use the custom fused CUDA kernel for `_srbd_step`. Requires `python setup.py build_ext --inplace` once; if the extension is missing the code prints a warning and falls back to the pure-PyTorch path automatically, so the flag is always safe. Set `False` to force the PyTorch reference implementation (see [SRBD CUDA Kernel](#srbd-cuda-kernel-optional)).

## Training Output

After training completes, the following files are generated in the `results/` folder
(this folder is git-ignored — only source code, the README and the URDF are tracked):

### Model Files
- `results/quad_diffsim_srbd_align_multi_robot<tag>.pth`: PyTorch model weights
- `results/quad_diffsim_srbd_align_multi_robot<tag>.pt`: TorchScript model (for ROS2 deployment)

`<tag>` is empty for blind runs, `_height` for the height-scan mode and
`_vision` for the depth-CNN mode, so the three modes never overwrite each other.

### Training Curves
- `results/loss_curve_srbd_align.png`: Total loss curve
- `results/loss_components_curve_srbd_align.png`: Individual loss components
- `results/vx_curve_srbd_align.png`: Body forward velocity curve
- `results/vx_curve_srbd_align_smooth.png`: Smoothed velocity curve
- `results/reward_curve_srbd_align.png`: Reward curve

### Data Files
- `results/*.npy`: NumPy arrays of various metrics (for post-analysis)

## Code Architecture

### Module Responsibilities

- **config.py**: Centralized management of all configuration parameters and global switches
- **utils_math.py**: Provides quaternion, rotation matrix, gravity projection and other math utilities
- **terrain.py**: Creates flat / rough / Rudin-curriculum terrain; non-flat terrain returns a `TerrainData` (heightfield + scales/offsets + world-frame mesh) so the env can look up surface height at any (x, y) and perception can ray-cast the exact PhysX surface
- **policy.py**: Neural network policies — blind MLP (`Policy`) and depth-CNN vision policy (`DepthEncoder` + `VisionPolicy`)
- **gait.py**: `GaitPlanner` class, handles gait planning, phase management, foothold calculation
- **srbd.py**: `SRBDModel` class, implements differentiable rigid body dynamics forward propagation
- **env.py**: `RealQuadEnv` class, wraps Isaac Gym simulation environment (+ optional perception collector)
- **perception/**: Self-contained terrain-perception package (Warp depth camera + differentiable height sampling); no Isaac Gym dependency, works standalone
- **train.py**: Training main loop, includes loss calculation, backpropagation, model saving
- **evaluate_rudin_comparison.py**: Deterministic evaluation producing metrics directly comparable to Rudin's legged_gym logs

### Key Design Patterns

#### GaitPlanner (Gait Planner)
Uses `__getattr__` and `__setattr__` to proxy access to environment attributes, avoiding circular dependencies:
```python
self.gait = GaitPlanner(self)
pref, stance_mask = self.gait._update_foot_targets_from_command(phases, p_foot)
```

#### SRBDModel (Simplified Rigid Body Dynamics)
Also uses proxy pattern to implement differentiable physics propagation:
```python
self.srbd = SRBDModel(self)
self.srbd._srbd_step(f_world=f_est, q_ref12=qref, dt=dt)
```

#### α-Alignment Mechanism
In each training step, blends Isaac Gym's real physics with SRBD predictions:
```python
env.srbd_p = env.base_pos + alpha * (env.srbd_p - env.srbd_p.detach())
env.srbd_v = env.base_lin + alpha * (env.srbd_v - env.srbd_v.detach())
```

## SRBD CUDA Kernel (optional)

The per-step centroidal dynamics in `SRBDModel._srbd_step` can run through either of two backends, selected by a single switch in `config.py`:

```python
# config.py
CUDA_KERNEL_SRBD = True    # Custom fused CUDA kernel (default; falls back to PyTorch if not built)
CUDA_KERNEL_SRBD = False   # Pure PyTorch reference path (always works, no build step)
```

When `True`, `srbd.py` imports the compiled `srbd_cuda_ext` module and dispatches both directions through a `torch.autograd.Function` wrapper (`SRBDStepFunction`):
- **Forward** is a single fused kernel in `src/srbd_cuda.cu` (foot FK + force/torque accumulation + Newton-Euler dynamics + quaternion integration), one thread per environment.
- **Backward** is a hand-written analytic adjoint kernel in the same file — no PyTorch replay, no `torch.autograd.grad` re-execution. It recomputes the forward intermediates and applies the chain rule directly, returning gradients for all six tensor inputs (`p, v, q, w, f_world, q_ref12`).

`loss.backward()` works identically under both backends. Gradients propagate through the SRBD state across the full rollout, matching the pure-PyTorch behavior. The kernel is built with `--use_fast_math`, so gradients agree with the PyTorch path to ≈ 1e-4 absolute (1-2 ULP per `sinf`/`cosf`/`rsqrtf` call, accumulated through the chain).

If the extension is not built, the code prints a warning and silently falls back to the PyTorch path, so toggling the flag is always safe.

### Building the extension

The extension uses `torch.utils.cpp_extension.CUDAExtension`. By default `setup.py` cross-compiles for every major NVIDIA architecture from Pascal (sm_60) through Hopper (sm_90), plus PTX for forward-compatibility with future GPUs (≈ 3–8 minutes the first time):

```bash
# From the project root, with your conda environment active
python setup.py build_ext --inplace
```

This produces `srbd_cuda_ext*.so` (Linux) / `srbd_cuda_ext*.pyd` (Windows) in the project root.

For faster iteration during development, restrict the build to the GPU on the current machine (~30 s):

```bash
# Linux
TORCH_CUDA_ARCH_LIST="native" python setup.py build_ext --inplace
```

```powershell
# Windows PowerShell
$env:TORCH_CUDA_ARCH_LIST = "native"
python setup.py build_ext --inplace
```

### Running training with the CUDA kernel

```bash
# 1) Set the flag in config.py:
#    CUDA_KERNEL_SRBD = True
# 2) Run training as usual:
python train.py
```

On the first import you should see:

```
[SRBD] Custom CUDA kernel active (CUDA_KERNEL_SRBD=True in config.py).
```

If the extension is missing or fails to import, you will instead see:

```
[SRBD] WARNING: CUDA_KERNEL_SRBD=True but extension not found (...).
[SRBD]          Falling back to PyTorch implementation.
[SRBD]          Run: python setup.py build_ext --inplace
```

### Testing the kernel

After every rebuild, run the parity test to verify the CUDA kernel matches the PyTorch reference path on the same inputs:

```bash
python tests/test_srbd_kernel.py
```

The test runs three checks and exits with code 0 on success:

1. **Forward parity** — compares `srbd_step_forward` (CUDA) against an inline copy of the inline PyTorch path from `srbd.py:_srbd_step` on a random batch (B = 16). Tolerance `atol=1e-4, rtol=1e-3`.
2. **Backward parity** — builds random upstream gradients, runs `torch.autograd.backward` on the PyTorch reference, calls `srbd_step_backward` (CUDA) directly, and compares all six input gradients (`p, v, q, w, f_world, q_ref12`). Tolerance `atol=1e-3, rtol=1e-2`.
3. **Autograd wrapper round-trip** — calls `SRBDStepFunction.apply(...)` end-to-end, runs `.backward()` on a weighted sum of outputs, and compares `.grad` of each input against the PyTorch reference. This catches bugs in how the wrapper plumbs `ctx`/`needs_input_grad`, not just in the kernel.

Expected output:

```
[1/3] Forward parity (atol=1e-4, rtol=1e-3)
  [OK ] p_new        max abs 1.xx e-06  max rel 1.xx e-06
  ...
[2/3] Backward parity, direct ext call (atol=1e-3, rtol=1e-2)
  [OK ] g_p          max abs 5.xx e-06  max rel 1.xx e-05
  ...
[3/3] SRBDStepFunction.apply round-trip (atol=1e-3, rtol=1e-2)
  ...
All SRBD kernel tests passed.
```

Per-tensor max absolute and max relative diff is printed for every check, so a failing line tells you which gradient drifted and by how much. Concrete numbers depend on GPU + driver; **orders of magnitude are what matter**. Drift around `1e-3` on `g_q_ref12` is the expected fast-math hit on the per-foot `sinf`/`cosf` chain — it is not a regression. Drift above the printed tolerances indicates one of:

- The extension wasn't rebuilt after a `.cu`/`.cpp` change (re-run `python setup.py build_ext --inplace`).
- A real math error was introduced in the kernel.
- You want bit-tighter parity than fast-math allows — drop `--use_fast_math` from the `nvcc` flags in `setup.py` and rebuild; the test should then pass with much smaller residuals (cost: marginally slower forward/backward).

The test requires the extension to be built and a CUDA-capable GPU. It does **not** depend on Isaac Gym and does **not** read `config.py` — the dispatch toggle is bypassed internally so the kernel itself is always exercised.

### When to enable it

- **Small `num_envs` (≤ 64)**: PyTorch is usually fine; the kernel-launch overhead of the many small ops doesn't dominate.
- **Large `num_envs` (a few hundred to a few thousand)**: the fused kernel becomes substantially faster than the PyTorch path because it replaces dozens of small dispatched ops per env with a single launch, and the per-env working set fits entirely in L2.
- **For debugging / numerical comparison**: keep `CUDA_KERNEL_SRBD = False`. The PyTorch path is the reference implementation.

### Requirements (CUDA kernel only)

- A working PyTorch CUDA install (`python -c "import torch; print(torch.cuda.is_available())"` returns `True`).
- A CUDA Toolkit on `PATH` matching your PyTorch build (`nvcc --version`).
- A C++ compiler compatible with that PyTorch build (gcc/clang on Linux, MSVC Build Tools on Windows).

## Profiling & Finding Bottlenecks (Linux)

This project has two layers worth profiling separately:

1. **The whole pipeline** — the Python training loop, Isaac Gym stepping, and all the PyTorch
   ops (`train.py`, `env.py`, `gait.py`, the PyTorch SRBD path). This tells you *where* time goes:
   CPU vs GPU, simulation vs policy vs loss/backward.
2. **The custom CUDA kernels** — `foot_positions_kernel`, `srbd_step_kernel`,
   `srbd_step_backward_kernel` in `src/srbd_cuda.cu` (active only when `CUDA_KERNEL_SRBD = True`).
   This tells you *why* a kernel is slow (memory- vs compute-bound, occupancy).

Work top-down: triage → whole-pipeline profile → zoom into the worst kernel. Don't start in
Nsight Compute.

### Three rules that make GPU profiling trustworthy

- **Warm up first.** The first ~10–20 iterations include CUDA context creation, cuDNN/cuBLAS
  autotuning, and lazy allocation. Always skip them, or your "hot spot" is just startup.
- **CUDA is asynchronous.** Wall-clock timing around a GPU call measures only the *launch*, not
  the work. Call `torch.cuda.synchronize()` before you read the clock, or use CUDA events. The
  profilers below handle this for you.
- **Profile a short, realistic run.** Edit the entry point in `train.py`
  (`train(num_iters=1000, ...)`) down to e.g. `num_iters=50`, set `cfg.use_viewer = False`, and
  profile at the `num_envs` you actually train at — kernel-launch overhead vs. compute balance
  changes completely between 16 and 1000 envs.

### Tools to install

| Tool | Use it for | Install |
|------|-----------|---------|
| `py-spy` | Zero-code-change sampling profiler → flame graph of Python (and native) stacks | `pip install py-spy` |
| `torch.profiler` | Per-op CPU **and** CUDA time, incl. the custom kernel; Chrome/TensorBoard trace | built into PyTorch |
| `snakeviz` | Interactive viewer for `cProfile` output | `pip install snakeviz` |
| `line_profiler` | Line-by-line timing of one hot function | `pip install line_profiler` |
| **Nsight Systems** (`nsys`) | System-wide timeline: CPU↔GPU overlap, gaps, sync stalls, kernel launches | NVIDIA CUDA Toolkit / [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems) |
| **Nsight Compute** (`ncu`) | Deep per-kernel analysis (occupancy, memory throughput, warp stalls) | NVIDIA CUDA Toolkit / [developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute) |
| `nvtop` | Live GPU/mem utilization (htop-style) | `sudo apt install nvtop` |

### Step 1 — Triage: CPU-bound or GPU-bound?

Run training in one terminal and watch the GPU in another:

```bash
nvtop                       # or: watch -n 0.5 nvidia-smi
nvidia-smi dmon -s u        # utilization sampled over time (good for logging)
```

- **GPU util pinned near 100%** → you're GPU-bound; go to Steps 3–5 (kernels / GPU ops).
- **GPU util low and spiky** → you're CPU-bound or sync-bound (Python overhead, Isaac Gym CPU
  work, host↔device copies); Step 2 (py-spy) will show it fastest.

### Step 2 — Whole-pipeline profiling

**py-spy (start here — no code changes).** Sampling profiler; produces a flame graph.

```bash
# Profile a fresh run end-to-end:
py-spy record --native -o results/pyspy_train.svg -- python train.py
# --native also shows C/C++/CUDA-launch frames, not just Python.

# Or attach to an already-running training process (may need sudo for ptrace):
py-spy record --native -o results/pyspy_train.svg --pid <PID>

# Live, top-style view:
py-spy top -- python train.py
```

Open the `.svg` in a browser; the widest bars are where wall-clock time is spent.

**torch.profiler (per-op CPU + CUDA breakdown, incl. the SRBD kernel).** Wrap the iteration loop
in `train.py`. The `schedule` skips warmup automatically:

```python
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=5, warmup=5, active=10, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./results/torch_profiler"),
    record_shapes=True, with_stack=True,
) as prof:
    for it in pbar:                 # the existing outer training loop
        ...                         # one full iteration (rollout + loss + step)
        prof.step()                 # MUST be called once per iteration

# Print the top ops to the console:
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
```

View the timeline with `tensorboard --logdir results/torch_profiler`, or open the generated
`.json` trace in `chrome://tracing` / [ui.perfetto.dev](https://ui.perfetto.dev). The custom
kernels appear by name (`srbd_step_kernel`, etc.); high `cuda_time_total` for `aten::*` ops points
at the PyTorch SRBD path or Isaac Gym tensors instead.

**cProfile + snakeviz (Python-only; ignores async GPU time — use only for CPU hotspots):**

```bash
python -m cProfile -o results/train.prof train.py
snakeviz results/train.prof
```

**line_profiler (one suspicious function).** Add `@profile` above a hot function (e.g.
`RealQuadEnv.step`, `estimate_foot_forces`, or `_update_foot_targets_from_command`) and run:

```bash
kernprof -l -v train.py
```

### Step 3 — System timeline with Nsight Systems (`nsys`)

The best view of how CPU, Isaac Gym, PyTorch, and your kernels interleave — and where the GPU
sits idle waiting on the host.

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  --output=results/nsys_train \
  python train.py
```

Open `results/nsys_train.nsys-rep` in `nsys-ui`. Look for: gaps on the GPU rows (CPU-bound),
frequent `cudaStreamSynchronize`/`cudaMemcpy` (sync/copy stalls), and which kernels dominate.

**Annotate regions** so the timeline is readable — add NVTX ranges in `train.py`:

```python
import torch.cuda.nvtx as nvtx
nvtx.range_push("rollout");   ...inner step loop...   ; nvtx.range_pop()
nvtx.range_push("loss");      ...compute loss...       ; nvtx.range_pop()
nvtx.range_push("backward");  loss.backward();         ; nvtx.range_pop()
```

(Or wrap the whole loop in `with torch.autograd.profiler.emit_nvtx():` to auto-label every torch op.)

### Step 4 — Per-kernel deep dive with Nsight Compute (`ncu`)

Once `nsys`/`torch.profiler` names the worst kernel, analyze just that one. `ncu` *replays* each
kernel many times, so it's slow — always restrict the launches:

```bash
sudo ncu \
  --kernel-name "regex:srbd_step_kernel|srbd_step_backward_kernel|foot_positions_kernel" \
  --launch-skip 200 --launch-count 6 \
  --set full \
  --export results/ncu_srbd \
  python train.py
```

Open `results/ncu_srbd.ncu-rep` in `ncu-ui`. The report tells you directly whether the kernel is
**memory-bound** or **compute-bound**, plus achieved occupancy and warp-stall reasons — that's
your shopping list for `src/srbd_cuda.cu`.

> Note: reading GPU performance counters usually needs elevated privileges — run `ncu` with
> `sudo`, or have an admin set `NVreg_RestrictProfilingToAdminUsers=0` (the error message links to
> [this page](https://developer.nvidia.com/ERR_NVGPUCTRPERM) if you hit it).

### Step 5 — Does the CUDA kernel actually help?

Because the SRBD step has a PyTorch reference path, you can A/B test the kernel. Time a short run
each way (toggle `CUDA_KERNEL_SRBD` in `config.py`) at your real `num_envs`:

```python
import time, torch
torch.cuda.synchronize(); t0 = time.perf_counter()
# ... run N warmed-up iterations ...
torch.cuda.synchronize(); print(f"{N} iters: {time.perf_counter() - t0:.3f}s")
print("peak GPU mem (MB):", torch.cuda.max_memory_allocated() / 1e6)
```

If `True` (custom kernel) isn't meaningfully faster than `False` (PyTorch) at your `num_envs`, the
kernel isn't the bottleneck — re-check Step 2/3 before optimizing `src/srbd_cuda.cu`.

### Suggested loop

`nvtop` (triage) → `py-spy` + `torch.profiler` (find the hot region) → `nsys` (see why the GPU
waits) → `ncu` (fix the specific kernel) → re-measure with Step 5. Save each report under
`results/` (already git-ignored) so you can compare before/after.

## Debugging Features

### Initial Fall Debugging Prints
Set in `config.py`:
```python
DBG_INIT_FALL = True
DBG_INIT_FALL_STEPS = 250  # Only print first 250 steps
DBG_INIT_FALL_EVERY = 5    # Print every 5 steps
DBG_INIT_FALL_ENV = 0      # Only watch dog 0
```

### Gradient Flow Analysis
Training automatically prints gradient norms for each layer to diagnose vanishing/exploding gradients.

### Direction Checking
Prints body velocity direction every 20 iterations to verify velocity tracking is correct.

## FAQ

### Q: Robot keeps flipping forward during training?
A: Check the following:
1. `train_no_aerial = True` (disable aerial phase)
2. `alpha_align = 0.9` (sufficient alignment coefficient)
3. Step frequency not too high (recommend 1.5-1.8 Hz)
4. Check if `last_contact_z` is updated correctly

### Q: Isaac Gym import fails?
A: Ensure Isaac Gym is imported before PyTorch. This is already handled in the code, but if issues persist, check:
```python
# Correct order
from isaacgym import gymapi
import torch

# Wrong order
import torch
from isaacgym import gymapi  # Will error
```

### Q: Training is slow?
A:
1. Ensure `use_gpu_pipeline = True`
2. Increase `num_envs` (more parallel environments)
3. Turn off `use_viewer` (don't render during training)
4. Use a faster GPU

To find the *actual* bottleneck instead of guessing, see
[Profiling & Finding Bottlenecks](#profiling--finding-bottlenecks-linux).

### Q: How to deploy to real robot?
A: After training, use the TorchScript model:
```python
model = torch.jit.load("results/quad_diffsim_srbd_align_multi_robot.pt")
action = model(observation)  # (1, 36) -> (1, 12)
```
The height-mode export takes a (1, 223) observation, and the vision-mode export
(`..._vision.pt`) takes two inputs: `(obs (1, 36), depth (1, 1, 12, 16))`.

### Q: `CUDA_KERNEL_SRBD = True` but I see the fallback warning?
A: The extension hasn't been built (or not in the current Python environment). From the project root:
```bash
python setup.py build_ext --inplace
```
This produces `srbd_cuda_ext*.so` / `*.pyd` next to `srbd.py`. After that, set `CUDA_KERNEL_SRBD = True` in `config.py` and re-run. If the build itself fails, check that `nvcc --version` works and that the CUDA Toolkit matches your PyTorch build (`python -c "import torch; print(torch.version.cuda)"`).

### Q: Do I need to rebuild the extension after every code change?
A: Only after modifying `src/srbd_cuda.cu`, `src/srbd_ext.cpp`, or `setup.py`. Changes to `srbd.py` or any other `.py` file do **not** require a rebuild — just re-run `python train.py`.

## Tests

```bash
# CPU-only, no Isaac Gym required:
python tests/test_vectorization.py     # loop-vs-batched parity (foot forces, stance)
python tests/test_perception_grad.py   # differentiable terrain sampling values + gradients
python tests/test_vision_policy.py     # VisionPolicy shapes, encoder gradients, TorchScript

# GPU + built extension required:
python tests/test_srbd_kernel.py       # CUDA kernel vs PyTorch parity (fwd/bwd/autograd)
```

## Third-Party Code & Attribution

Parts of this repository are ported from or based on prior work:

- **legged_gym** (Rudin et al., *"Learning to Walk in Minutes Using Massively
  Parallel Deep Reinforcement Learning"*, CoRL 2021;
  https://github.com/leggedrobotics/legged_gym): the curriculum-grid terrain
  generation in `terrain.py` (`Terrain` class, `gap_terrain`, `pit_terrain`)
  is ported verbatim, and the terrain curriculum / robot-placement logic in
  `env.py` (`_assign_rudin_origins`, `_update_terrain_curriculum`) is a close
  port. Copyright (c) 2021 ETH Zurich, Nikita Rudin — BSD-3-Clause; that
  license continues to apply to the ported portions. The 187-point height-scan
  layout and the evaluation metrics also follow this project.
- **MGDP** (`warp_sensor`): the depth ray-cast kernel
  (`perception/warp_kernels/cam_kernel.py`) is a trimmed, self-contained copy
  of MGDP's depth kernel, and the camera wrapper / depth post-processing in
  `perception/` are adapted from the same project.
- **DiffPhysDrone** (Zhang et al.): the end-to-end depth-CNN training approach
  and the 12x16 depth input scale are inspired by this work (concepts only, no
  code copied).

## Citation

If you use this code, please cite the associated thesis (details to be added
upon publication).

## License

Not yet licensed — a license will be added at the end of the project. Until
then all rights are reserved for the original code; the third-party portions
listed above retain their respective licenses (BSD-3-Clause for the
legged_gym-derived code).

## Contact

Jakub Jura — kuba.jura3@gmail.com

---

**Note**: This project is based on Isaac Gym Preview 4 and is for research and educational purposes only.
