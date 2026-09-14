# Differentiable Simulation Quadruped Locomotion on a Terrain Curriculum

Bachelor thesis code. It is based on the implementation of Song, Kim and Scaramuzza, *Learning Quadruped Locomotion Using
Differentiable Simulation* (2024) paper, already existing on the Chair of Robotics, Artificial Intelligence and Real-time Systems at TUM. The implementation is rewritten for throughput and the method is pushed onto a
terrain curriculum in order to answer three research questions
about differentiable simulation as a way to learn locomotion.

> **RQ1.** What determines training throughput in a differentiable simulation locomotion pipeline and
is a hand-written fused CUDA kernel necessary to achieve it?
>
> **RQ2.** To what extent does propagating terrain gradients through a differentiable SRBD surrogate
improve hard-terrain locomotion, relative to a blind baseline and the identical terrain
signal supplied only as a policy observation?
>
> **RQ3.** Which structural properties of the reference tracking formulation limit the actionability of
terrain gradients and can targeted architectural interventions make them actionable?

For more details please check the paper itself, which is part of the repository.

The training loop optimises a first order objective through a differentiable Single Rigid Body
Dynamics surrogate whose state is aligned to ground truth PhysX physics at every step. Isaac Gym
supplies the values, the SRBD model supplies the gradient corridor, and the loss is backpropagated
through the whole rollout into the policy.

Every run folder behind the thesis is committed under `results/`, so the figures and tables can be
rebuilt on a laptop with no GPU and no Isaac Gym. This README is written so that one can
redo all of the experiments from scratch if needed.

---

## Quick start

A base run is one command. Flat ground, blind policy, 16 robots, 1000 iterations, roughly six
minutes on a GTX 1660 Ti. It is a reproduction of Song et al. work. Trains policy to walk forward on flat surface.

```bash
python train.py
```

Weights, a TorchScript export and the training curves land in `results/`. Then watch the policy
walk:

```bash
python play_many_dog.py
```

The two perception modes are one variable away:

```bash
MODE=height python train.py    # curriculum terrain, height scan in the observation and in the loss
MODE=depth  python train.py    # curriculum terrain, depth camera through a CNN encoder
```

Output files carry a suffix per mode, so the modes never overwrite each other:
`results/quad_diffsim_srbd_align_multi_robot<tag>.pth` for the weights, `...<tag>.pt` for the
TorchScript export, and `loss_*<tag>.png`, `vx_curve_*<tag>.png` and the rest for the curves. The
tag is empty for blind, `_height` for the height scan mode and `_vision` for the depth mode.

A few more playback options:

```bash
python play_many_dog.py --num_envs 16              # 16 robots at once
python play_many_dog.py --no_rand_cmd              # fixed command from cfg.cmd_fixed
python play_many_dog.py --gait_mode 1              # 0 stand, 1 trot, 2 pace, 3 bound, 4 gallop
python play_many_dog.py --weights your_model.pth
python play_many_dog.py --max_steps 1000
```

The perception stack can be inspected without Isaac Gym at all, which helps when the camera or the
height sampler is what you are debugging:

```bash
python -m perception.visualize_perception              # synthetic scene, picks a device
python -m perception.visualize_perception --device cpu
```

---

## What is in the repository

| file | what it does |
|---|---|
| `config.py` | `EnvCfg` plus every global switch and environment variable knob |
| `utils_math.py` | quaternions, rotation matrices, gravity projection |
| `terrain.py` | flat, rough and Rudin curriculum grid terrain, and the mapping from column to terrain family |
| `policy.py` | the blind MLP `Policy` and the depth CNN `VisionPolicy` |
| `gait.py` | `GaitPlanner`: phases, stance masks, Raibert footholds |
| `srbd.py` | `SRBDModel`: the differentiable dynamics step, PyTorch and CUDA backends |
| `env.py` | `RealQuadEnv`: Isaac Gym wrapper, force estimation, terrain curriculum, resets |
| `train.py` | training loop, the Eq. (5) loss, all run modes |
| `bench_log.py` | per run folder writer for `iters.csv`, `meta.json` and `summary.json` |
| `evaluate_rudin_comparison.py` | deterministic evaluation, and the saliency corruption harness |
| `play_many_dog.py` | viewer playback of a checkpoint |
| `collect_bench.py` | merges run folders into `runs.csv` and every thesis figure, needs no torch |
| `collect_step1c.py`, `collect_step3.py`, `collect_saliency.py` | readers for the three diagnostic campaigns |
| `setup.py` | builds the SRBD CUDA extension |
| `go2_description.urdf` | the robot |
| `perception/` | Warp depth camera, differentiable height sampler, preprocessing |
| `src/srbd_cuda.cu`, `src/srbd_ext.cpp` | the fused kernel and its pybind11 bindings |
| `run_campaign.sh`, `run_step1c.sh`, `run_step3.sh`, `run_saliency.sh` | run queues, described further down |
| `tests/` | parity and gradient tests, several of which run on CPU |
| `results/` | every run folder from the thesis campaign |

---

## How the system works

### One training iteration

An iteration rolls the policy out for `steps_per_iter` physics steps, computes the loss from the
SRBD states and takes one optimiser step. Inside the rollout:

1. `env.get_obs()` assembles the observation from the current Isaac Gym state. It carries no
   gradient, because observations are input leaves and the gradient path into the policy runs
   through the SRBD losses instead.
2. The policy is queried every `action_hold` steps, which is 5, so control runs at 100 Hz against
   physics at 500 Hz. Between queries the previous action is held and detached.
3. `env.step(delta_q)` turns the 12 policy outputs into PD joint targets with tanh squashing, per
   joint scaling, joint limits and rate limiting, advances PhysX once, refreshes the cached state
   and evaluates termination.
4. `env.estimate_foot_forces()` recovers the ground reaction forces with a damped pseudoinverse
   solve, and `SRBDModel._srbd_step` reintegrates them differentiably.
5. Alpha alignment blends the two. The value comes from Isaac Gym, the gradient comes from the SRBD
   model:

```python
env.srbd_p = env.base_pos + alpha * (env.srbd_p - env.srbd_p.detach())
env.srbd_v = env.base_lin + alpha * (env.srbd_v - env.srbd_v.detach())
```

The default `alpha_align` is 0.9. This is what keeps a differentiable surrogate usable over a long rollout without drifting away from the real
simulator.

The rollout window is `steps_per_iter = 24` physics steps, which is 48 ms of simulated time and
roughly five policy decisions. That window is the entire gradient horizon, and section 4 of the
reproduction below tests directly whether it is the reason the terrain results come out the way they
do.

### Observation and action

The blind observation is 36 numbers: 3 command values, 8 phase sine and cosine terms, 3 body frame
linear velocities, the 4 base quaternion components, 3 body frame angular velocities, 12 joint
deltas from the default posture, and the 3 component gravity projection. With `use_height_obs` a
height scan of 187 points is appended, which makes it 223. In depth mode the observation stays at 36
and the depth image is a separate CNN input.

The action is 12 joint angle offsets. In the foothold experiments the policy emits 4 or 8
extra outputs, which are per leg foothold corrections capped at `FOOT_RES_MAX`.


### The perception package

| file | what it does |
|---|---|
| `perception/config.py` | `PerceptionCfg`: camera intrinsics and mounting, height grid extent, noise, loss field blur |
| `perception/collector.py` | `PerceptionCollector`, one `collect()` entry point called once per step |
| `perception/warp_camera.py` | Warp ray casting depth camera, captured in a CUDA graph |
| `perception/warp_kernels/cam_kernel.py` | the depth kernel itself, vendored from MGDP |
| `perception/height_sampler.py` | differentiable height map lookup through `grid_sample` |
| `perception/terrain_mesh.py` | terrain adapter and Warp mesh construction |
| `perception/preprocessing.py` | depth clipping, resizing, normalisation and noise |
| `perception/visualize_perception.py` | offline renderer for sanity checks, no Isaac Gym needed |

Everything stays resident on the GPU and is captured in a CUDA graph, which is what keeps the depth
mode within a factor of 1.3 of the blind mode in wall time.

### The SRBD CUDA kernel

`SRBDModel._srbd_step` runs through one of two backends, chosen by `CUDA_KERNEL_SRBD`. The forward
pass of the fused kernel is a single launch in `src/srbd_cuda.cu` covering foot kinematics, force
and torque accumulation, the Newton and Euler equations and quaternion integration, with one thread
per robot. The backward pass is a handwritten analytic adjoint in the same file. It recomputes the
forward intermediates and applies the chain rule directly, returning gradients for all six tensor
inputs, so no PyTorch replay is involved.

`loss.backward()` behaves identically under both backends. Gradients propagate through the SRBD
state across the full rollout either way.

```bash
python setup.py build_ext --inplace                                # 3 to 8 min, sm_60 through sm_90 plus PTX
TORCH_CUDA_ARCH_LIST=native python setup.py build_ext --inplace    # about 30 s, this GPU only
```

On import you should see `[SRBD] Custom CUDA kernel active`. If the extension is missing the code
prints a warning and falls back to PyTorch, so leaving `CUDA_KERNEL_SRBD=1` on is always safe. The
fallback does not catch an extension built for the wrong architecture, which fails at the first
kernel launch instead. Rebuild only after touching `src/*.cu`, `src/*.cpp` or `setup.py`. Changes to
any `.py` file need no rebuild.

Keep `CUDA_KERNEL_SRBD=0` when you are debugging numerics. The PyTorch path is the reference
implementation.

---

## Installation

Isaac Gym has to be imported before torch. `train.py` already does this, and it is the reason for
the import order at the top of every entry point.

```bash
conda create -n diffsim python=3.8
conda activate diffsim
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy matplotlib tqdm

# Isaac Gym Preview 4, downloaded separately from NVIDIA
cd isaacgym/python && pip install -e .

# only for the perception and vision modes
pip install warp-lang==1.0.2
```

Building the CUDA kernel additionally needs a CUDA Toolkit on the path matching your PyTorch build,
so that `nvcc --version` works, and a C++ compiler compatible with that build.

### The machine everything was measured on

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 Ti, 6 GB, Turing, sm_75 |
| OS | Ubuntu LTS 20.04, Isaac Gym Preview 4 |
| Python and PyTorch | 3.8.20 and 1.13.1+cu117, CUDA 11.7 |
| Warp | 1.0.2, perception modes only |
| Robot | Unitree Go2, 500 Hz physics, 100 Hz control |

The GPU matters for the speed numbers and for nothing else. Everything else scales, and a larger
card mainly lets you raise `NUM_ENVS`. Each run folder records its own GPU, host, git commit and
library versions in `meta.json`, so nothing here has to be taken on trust.

**One thing to know before comparing any two runs.** Isaac Gym and PhysX do not reproduce results
bit for bit across processes. Two runs of identical source, identical seed and identical
configuration, launched 46 minutes apart, disagree by roughly 0.0005 relative on the loss of the
first row, and it is worse at small `NUM_ENVS`. Any gate written at 0.0001 or tighter will fire
falsely. Where a wiring claim has to be made, it is made with an identity inside a single process at
iteration 0 instead, which both the tilt barrier and the foothold stages below use.

### Sanity checks

```bash
python tests/test_vectorization.py     # parity between the loop and batched paths, CPU
python tests/test_perception_grad.py   # differentiable terrain sampling and gradients, CPU
python tests/test_vision_policy.py     # VisionPolicy shapes, gradients, TorchScript, CPU
python tests/test_srbd_kernel.py       # CUDA against PyTorch: forward, backward, autograd round trip
```

`test_srbd_kernel.py` needs the built extension and a GPU. It prints the largest absolute and
relative difference per tensor. Drift around 0.001 on `g_q_ref12` is the expected cost of
`--use_fast_math` on the chain of `sinf` and `cosf` calls per foot, and is not a regression.
Anything above the printed tolerances usually means the extension was not rebuilt after a change to
a `.cu` file.

---

## Configuring a run

Everything is an environment variable rather than a command line flag, because `config.py` is
imported at module scope and `srbd.py` reads `CUDA_KERNEL_SRBD` at import time, long before argument
parsing could run.

### Modes

`MODE` picks a preset in `train.py:MODE_CFG`. The `_fwd` variants exist because on Rudin terrain the
command distribution changes together with the terrain, which would otherwise confound "the terrain
is hard" with "the commands are hard".

| `MODE` | terrain | command | observation | how terrain reaches the policy |
|---|---|---|---|---|
| `blind` | flat | fixed 0.5 m/s forward | 36D | not at all |
| `blind_omni` | flat | omnidirectional | 36D | not at all |
| `blind_omni_heading` | flat | omnidirectional plus heading target | 36D | not at all |
| `blind_rudin` | rudin | omnidirectional | 36D | not at all |
| `hobs` | rudin | omnidirectional | 223D | observation |
| `hloss` | rudin | omnidirectional | 36D | loss |
| `height` | rudin | omnidirectional | 223D | both |
| `depth` | rudin | omnidirectional | 36D plus depth image | rendered camera |
| `blind_rudin_fwd` | rudin | fixed 0.5 m/s | 36D | not at all |
| `blind_rudin_rand` | rudin | vx in [0.4, 0.8] | 36D | not at all |
| `hobs_fwd` | rudin | fixed 0.5 m/s | 223D | observation |
| `fz_fwd` | rudin | fixed 0.5 m/s | 36D | swing target supervision |
| `fhold_fwd` | rudin | fixed 0.5 m/s | 223D | action space and observation |

`hobs` and `hloss` are the two ablation cells that split the terrain observation path from the
terrain gradient path. The same terrain signal reaches the policy two different ways.

On Rudin terrain the command switches to the omnidirectional ranges of Rudin et al., vx and vy in
[-1, 1] m/s and yaw in [-1, 1] rad/s, unless the mode fixes it. Flat ground uses a fixed forward
0.5 m/s. A policy trained on flat ground has therefore never seen a lateral or yaw command, which is
exactly what the `_fwd` modes exist to control for.

`PERCEPTION_TERRAIN=1|height|depth` still works as an older spelling of `MODE`.

### Variables

| variable | default | effect |
|---|---|---|
| `MODE` | `blind` | run mode, table above |
| `SEED` | 0 | RNG seed |
| `NUM_ENVS` | 16 | parallel robots |
| `ITERS` | 1000 | training iterations |
| `STEPS_PER_ITER` | 24 | rollout and gradient window in physics steps, 24 at 2 ms is 48 ms |
| `RUN_DIR` | unset | put every output of this run in one folder, and measure it |
| `CUDA_KERNEL_SRBD` | 1 | fused kernel on or off |
| `FORCE_DTYPE` | fp32 | precision of the foot force pseudoinverse solve |
| `DEBUG_TRAIN` | 0 | verbose diagnostics. Each one forces a sync from GPU to CPU, so leave it off for anything measured |
| `TILT_W`, `TILT_ON` | 0.0, 0.6 | soft tilt barrier weight and hinge onset in radians |
| `FOOT_Z_TERRAIN`, `FOOT_APEX_TERRAIN` | 0, 0 | swing target lands at, or clears, the true terrain |
| `FOOT_RES`, `FOOT_RES_Y` | 0, 0 | per leg foothold residual outputs, sagittal only or with lateral |
| `FOOT_Q_W`, `FOOT_RES_W` | 0.0, 0.0 | foothold quality and residual shrinkage weights |
| `FOOT_RES_MAX`, `FOOT_RES_DETACH` | 0.10, 1 | residual cap in metres, and whether to detach it inside the `loss_foot` target |

`TILT_W` and the `FOOT_*` flags are deliberately absent from `MODE_CFG`, because a mode dict is
applied after `EnvCfg()` and would silently override the variable.

One note on `NUM_ENVS` for Rudin terrain: use hundreds or thousands. The grid spreads robots one
column at a time, so with fewer robots than the 20 columns most columns stay empty. The scheme only
fills the landscape at the batch sizes it was designed for.

### What a run writes

With `RUN_DIR` set the mode suffix is dropped, since the folder already separates runs, and
`bench_log.py` writes:

| file | contents |
|---|---|
| `meta.json` | full configuration, every intervention flag, git commit, GPU, host, timestamps |
| `summary.json` | median, p10 and p90 iterations per second, peak allocator MB, total wall seconds, every `final_*` metric |
| `iters.csv` | one row per iteration: loss and all components, `vx`, `grad_norm`, `terrain_level`, falls by cause, `n_move_up` and `n_move_down`, base height, contacts, foot clearance |
| `final_state.json` and `.npz` | curriculum state at the end of training: the histogram over the 10 difficulty rows, mean distance walked, and `by_terrain_family` with mean and max level per terrain family |
| `*.png`, `*.npy` | the per run curves |

`by_terrain_family` is the important one. It is what turns "mean level 0.02" into "smooth slope
0.18, stairs up 0.00", and the second reading is the thesis result.

`t_iter_s` brackets each iteration with `torch.cuda.synchronize()`, so it times GPU work rather than
kernel launches. The first 20 iterations are left out of the summary statistics, because they carry
CUDA context creation, cuBLAS autotuning, the terrain build and allocator growth.

`bench_log.py` imports nothing from this repository beyond torch and the standard library, so it can
be copied into an older checkout to measure that checkout the same way.

---

## Reproducing the results

Roughly 34 GPU hours for everything, of which Experiment 2 alone is 28.7 h. The run queues are
sequential on purpose, because there is one GPU and two training processes would corrupt every
timing measurement. They are also resumable: a run whose `RUN_DIR` already contains `summary.json`
is skipped, so a queue that dies at run 9 of 13 picks up where it stopped. Delete a folder to force
a redo. Check .sh files to run the experiments automatically.

### 0. Rebuild the figures without running anything

The committed run folders are enough to regenerate every figure and table in the thesis.

```bash
python collect_bench.py --results-dir results --out results/figures
```

numpy and matplotlib only. No torch, no Isaac Gym, no GPU. This is the fastest way to confirm that
the numbers in the thesis come out of the data on disk.

### 1. Experiment 1, training throughput, RQ1, 15 min

Twelve runs, two SRBD backends across six batch sizes, flat ground, blind policy, 100 iterations
each.

```bash
./run_campaign.sh exp1
```

The third variant is the implementation as it stood before the rewrite, and it is not in this
working tree. It is the last upstream commit, `1731453`, since this repository is a fork of
`github.com/RonGenZ/RonGenZ`. To measure it, check that commit out into a separate directory, copy
`bench_log.py` across, and run its `train.py` at its default `num_envs=16`. Parity was checked
first: same fixed 0.5 m/s command, same trot, same `action_hold`, same settle steps, same
`steps_per_iter=24`, same fp32 force solve.

### 2. Experiment 2, the perception ablation across five conditions, RQ2, 28.7 h

Thirteen runs: five conditions, seeds 0, 1 and 2 for four of them, and seed 0 only for `depth`,
which is the slowest.

```bash
./run_campaign.sh smoke                                  # 5 runs of 30 iterations, sizes the batch
BSTAR=2048 ITERS=5000 SEEDS="0 1 2" ./run_campaign.sh exp2
```

### 4. The intervention campaign, RQ3, about 4 h

A different task from Experiment 2. Forward commands, 1024 robots and 850 iterations here, against
omnidirectional commands, 2048 robots and 5000 iterations there. The two never share a figure axis.

**Baseline, command style and height observation**, into `results/diag20` and `results/diag21`:

```bash
MODE=blind_rudin_fwd  SEED=0 ITERS=850 NUM_ENVS=1024 RUN_DIR=results/diag20/blind_rudin_fwd_s0 python train.py
MODE=blind_rudin_rand SEED=0 ITERS=850 NUM_ENVS=1024 RUN_DIR=results/diag20/blind_rudin_rand_s0 python train.py
MODE=hobs_fwd         SEED=0 ITERS=850 NUM_ENVS=1024 RUN_DIR=results/diag20/hobs_fwd_s0 python train.py

./run_step3.sh seeds     # baseline seeds 1 and 2, into results/diag21/rudin_fwd_w0_s{1,2}
```

The three baseline seeds are the noise band every treatment arm is read against. They end at mean
terrain level 0.0225, 0.0283 and 0.0146. `hobs_fwd` is the one separation in the campaign, where the
height scan in the observation delays the curriculum collapse roughly twofold before converging to
the same floor. It is one seed, so treat it as a remark rather than a result.

**Soft tilt barrier**, the second limitation the paper names for itself, implemented as a
differentiable stand in for a termination penalty. `run_step1c.sh` covers the wiring test, the flat
control pair and the calibration probe:

```bash
./run_step1c.sh all      # about 14 min
python collect_step1c.py
```

The two Rudin arms were run directly, at the weights the probe implies, which put the term at 5.1 %
and 15.8 % of the objective. Those weights come out 6 times smaller than a calibration on flat
ground would have given, which is why the probe runs on terrain in the first place:

```bash
MODE=blind_rudin_fwd SEED=0 ITERS=850 NUM_ENVS=1024 TILT_W=16 RUN_DIR=results/diag21/rudin_fwd_w16 python train.py
MODE=blind_rudin_fwd SEED=0 ITERS=850 NUM_ENVS=1024 TILT_W=48 RUN_DIR=results/diag21/rudin_fwd_w48 python train.py
```

**Perceptive foothold**, the first limitation the paper names, that the method cannot explore foot
placement through the velocity tracking loss alone. Phase 1 changes where the swing foot is aimed.
Phase 2 gives the policy explicit per leg foothold residual outputs.

```bash
./run_step3.sh phase1                     # fz_off, fz_z, fz_za, about 45 min
./run_step3.sh wiring                     # the same identity at iteration 0, on loss_fq
./run_step3.sh probe2                     # calibration in the regime the arms actually run in
./run_step3.sh xy                         # the two Phase 2 arms
python collect_step3.py
```

The `xy` stage calibrates its own weights from the probe run, which in the runs on disk gave
`FOOT_Q_W=13834`, worth 5 % of the objective, and `FOOT_RES_W=795`, worth 2 %:

```bash
MODE=fhold_fwd FOOT_RES=1 FOOT_Z_TERRAIN=1 FOOT_APEX_TERRAIN=1 SEED=0 ITERS=450 NUM_ENVS=1024 \
  FOOT_Q_W=13834 FOOT_RES_W=795 RUN_DIR=results/diag23/fhold_x_q13834 python train.py
# and the same with FOOT_RES_Y=1 into fhold_xy_q13834
```

**Gradient window.** The explanation offered for the three nulls above was that the 48 ms window is
too short to connect a foot placement to whether the robot is still upright a step later. That was
an argument rather than a measurement, so it got tested:

```bash
STEPS_PER_ITER=48 MODE=blind_rudin_fwd SEED=0 ITERS=425 NUM_ENVS=1024 \
  RUN_DIR=results/diag24/horizon48_matched python train.py     # matched simulated time
STEPS_PER_ITER=48 MODE=blind_rudin_fwd SEED=0 ITERS=850 NUM_ENVS=1024 \
  RUN_DIR=results/diag24/horizon48_long python train.py        # doubled simulated time
```

**The mechanism, if you want to check it directly.** Read `final_state.json` and the `n_move_up` and
`n_move_down` columns of `iters.csv` against the promotion and demotion rule in
`env.py:_update_terrain_curriculum`. A promotion needs 4.0 m of travel from the cell origin within
one episode. The measured mean distance walked is 1.11 to 1.21 m on the campaign task and 0.94 to
0.97 m on Experiment 2. Mean survival is 3.6 to 4.0 s against the roughly 12 s a promotion needs at
the achieved 0.33 m/s. Summed over a run, demotions outnumber promotions by between 200 to 1 and
780 to 1. Promotion is unreachable by construction, which is why none of the interventions above
could move it.

### 5. Perceptual saliency, 13 evaluations, no retraining

The already trained seed 0 policies from Experiment 2 are replayed on Rudin terrain with their
terrain channel corrupted. Invariance to that corruption is evidence that the policy learned to
disregard the signal.

```bash
./run_saliency.sh selftest    # does the corruption actually reach the policy?
./run_saliency.sh matrix      # 13 runs into results/saliency/
python collect_saliency.py
```

Each run is 1024 robots and 15000 steps, which is 30 s of simulated time, after 2000 warmup steps.
The arms per policy are `none` and `none2`, the same seed drawn twice to give the floor between
runs, then `zero`, and then `shuffle`, where robot *i* persistently sees the terrain of robot
`perm(i)`. `hloss` is absent by design, because its observation is 36D and there is no terrain
channel to corrupt.

## Watching a trained policy

`--obs-mode` must match the checkpoint, since it sets both the observation size and the policy
class, and `--terrain` must match how the policy was trained.

```bash
python play_many_dog.py --terrain rudin --obs-mode height --num_envs 64 \
    --weights results/train/height_s0/quad_diffsim_srbd_align_multi_robot.pth
```

Without `--terrain-level` the robots are spread randomly over rows 0 to 5, so playback shows the
policy on easy terrain no matter what the run achieved. Pin the cell to see it at the difficulty
training actually reached, which you can read off `final_state.json`:

```bash
python play_many_dog.py --terrain rudin --obs-mode height --num_envs 32 \
    --terrain-level 6 --terrain-col "stairs up" \
    --weights results/train/height_s0/quad_diffsim_srbd_align_multi_robot.pth
```

`--terrain-col` takes an index or a family name out of `smooth slope`, `rough slope`, `stairs down`,
`stairs up` and `discrete obstacles`. Setting `--terrain-level` also switches the promotion and
demotion curriculum off, so the robots stay where you put them.

---

## Parameters that matter

| | value | where |
|---|---|---|
| physics and control step | 0.002 s and `action_hold = 5`, so 500 Hz and 100 Hz | `config.py` |
| rollout and gradient window | `steps_per_iter = 24`, which is 48 ms | `train.py` |
| alignment coefficient | `alpha_align = 0.9` | `config.py` |
| target height | `h0 = 0.35` m | `config.py` |
| gait | trot, `gait_mode = 1`, `step_freq = 1.6` Hz, `swing_height = 0.12` m | `config.py`, `gait.py` |
| loss weights | `a1..a6 = 10, 1.0, 0.01, 0.01, 0.5, 5.0`, `a7 = 3.0`, `yaw_w = 0.1` | `train.py:343` |
| gradient clip | 0.3, with the norm logged before clipping | `train.py:925` |
| episode length | 20 s | `config.py:373` |
| terrain grid | 10 difficulty rows by 20 type columns, cells of 8 m by 8 m | `config.py`, `RudinTerrainCfg` |
| type proportions | `[0.1, 0.1, 0.35, 0.25, 0.2]` | `config.py` |
| column to family | 0 and 1 smooth slope, 2 and 3 rough slope, 4 to 10 stairs down, 11 to 15 stairs up, 16 to 19 discrete obstacles | `terrain.py` |
| initial placement | `max_init_terrain_level = 5`, so runs start near mean level 2.48 | `config.py` |
| promotion and demotion | more than 4.0 m from the cell origin, or less than `cmd_speed * 20 s * 0.5` | `env.py`, lines 847 to 859 |
| fall by tilt | `fall_tilt_thresh = 0.9` rad on roll or pitch | `config.py:348` |

Two properties of the terrain are inherited from legged_gym verbatim, and both change how the
results read. Stairs occupy 12 of the 20 columns, and no stepping stone terrain is generated at all,
because that branch sits above a cumulative proportion the defaults push to 1.0.

Two switches in `config.py` are worth knowing about. `PURE_PAPER_MODE = True` keeps the loop to what
the paper describes, with no engineering additions such as action smoothing.
`ONLY_ITERATE_NO_RESET = True` performs the full environment reset once, at the first iteration,
after which robots reset individually as they fall or time out.

---

## Attribution

- **legged_gym**, Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep
  Reinforcement Learning*, CoRL 2021, https://github.com/leggedrobotics/legged_gym. The curriculum
  grid terrain generation in `terrain.py` is ported verbatim, and the curriculum and robot placement
  logic in `env.py`, meaning `_assign_rudin_origins` and `_update_terrain_curriculum`, is a close
  port. The layout of the height scan of 187 points and the evaluation metrics follow the same
  project. Copyright (c) 2021 ETH Zurich, Nikita Rudin, released under the BSD 3 Clause license,
  which continues to apply to the ported portions.
- **MGDP**, `warp_sensor`. `perception/warp_kernels/cam_kernel.py` is a trimmed and self contained
  copy of the depth ray casting kernel from that project, and the camera wrapper and depth
  preprocessing under `perception/` are adapted from it.
- **DiffPhysDrone**, Zhang et al. The approach of training a depth CNN end to end, and the 12x16
  depth input scale, are inspired by this work. Concepts only, no code copied.
- The implementation this project forked from is `github.com/RonGenZ/RonGenZ` at commit `1731453`,
  which is the first variant in Experiment 1.

## License

MIT, see [LICENSE](LICENSE). Reuse, modification and redistribution are all permitted, which is the
point: everything needed to reproduce the thesis should be usable without asking. The third party
portions listed above keep the license they came with, and the LICENSE file names them.

Jakub Jura, jakub.jura@tum.de
