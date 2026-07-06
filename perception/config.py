"""Configuration for the terrain-perception module.

A single :class:`PerceptionCfg` dataclass holds every tunable of the depth
camera, the height-map sampler and the depth post-processing. It is intentionally
plain (stdlib ``dataclasses``) so it can be embedded as a field inside the
environment's ``EnvCfg`` without adding any dependency.

All lengths are in **metres**, all angles in **degrees** (converted internally).
Frame conventions follow the environment: positions are world-frame, the base
offset is expressed in the robot body frame, quaternions are ``xyzw``.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PerceptionCfg:
    """Tunables for :class:`perception.collector.PerceptionCollector`.

    The defaults describe a single forward-facing, slightly down-pitched depth
    camera roughly matching a RealSense mounted on the Go2 head, plus a
    Rudin-style 17x11 = 187-point local height scan.

    Camera geometry:
        render_h, render_w: Ray-cast resolution actually launched on the GPU.
            Rendered a little larger than the policy input, then down-sampled in
            post-processing for cheap anti-aliasing. Keep small for speed.
        out_h, out_w: Resolution the depth image is resized to before it leaves
            the module (the CNN input size). Default 12x16 matches the
            DiffPhysDrone depth CNN stem.
        fov_deg: Horizontal field of view of the pinhole camera, degrees.
        max_range: Far clip (metres). Rays that miss return this value; also the
            clip ceiling used in post-processing.
        min_range: Near clip (metres), used only for normalisation bookkeeping.

    Camera mounting (body frame -> attached to the base):
        cam_offset_xyz: Translation of the camera from the base origin, in the
            body frame (x forward, y left, z up).
        cam_pitch_deg: Downward pitch of the camera about the body y-axis
            (positive = looking down at the terrain ahead).

    Height map (local grid around the robot, yaw-aligned):
        hm_x_min/max, hm_y_min/max: Grid extent in the body frame (metres). x is
            forward, y is left. Defaults [-0.8, 0.8] x [-0.5, 0.5].
        hm_res: Grid spacing (metres). Default 0.1 -> 17 x 11 = 187 points.
        hm_subtract_base_z: If True the sampler returns terrain height *relative*
            to the base (base_z - terrain_z), matching Rudin's height scan; if
            False it returns absolute world terrain height.
        hm_loss_blur_cells: Gaussian blur sigma (in heightfield cells) of the
            *smoothed* heightfield used by ``sample_points(smooth=True)`` for
            loss terms. Bilinear grid_sample gradients are piecewise-constant
            and spike at stair risers; blurring turns steps into ramps so the
            terrain-slope gradient is informative. 0 disables (smooth field ==
            exact field). The exact field always serves the observation path.

    Depth post-processing:
        normalize_depth: Scale the clipped depth to [0, 1] (divide by max_range).
        noise_gaussian: Std of multiplicative Gaussian noise for the *noisy*
            depth output, or None to disable. The *clean* output is never noised.
        noise_dropout: Per-pixel dropout probability for the *noisy* output, or
            None to disable.

    Runtime:
        num_sensors: Cameras per environment (1 for a single head camera).
        device: CUDA device string; the module is designed to stay resident here.
    """

    # --- camera geometry ---
    render_h: int = 48
    render_w: int = 64
    out_h: int = 12
    out_w: int = 16
    fov_deg: float = 87.0
    max_range: float = 4.0
    min_range: float = 0.0

    # --- camera mounting (body frame) ---
    cam_offset_xyz: Tuple[float, float, float] = (0.28, 0.0, 0.06)
    cam_pitch_deg: float = 28.0

    # --- height map ---
    hm_x_min: float = -0.8
    hm_x_max: float = 0.8
    hm_y_min: float = -0.5
    hm_y_max: float = 0.5
    hm_res: float = 0.1
    hm_subtract_base_z: bool = True
    hm_loss_blur_cells: float = 2.0

    # --- depth post-processing ---
    normalize_depth: bool = True
    noise_gaussian: Optional[float] = None
    noise_dropout: Optional[float] = None

    # --- runtime ---
    num_sensors: int = 1
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validate a few invariants that would otherwise fail cryptically later."""
        assert self.render_h > 0 and self.render_w > 0, "render resolution must be positive"
        assert self.out_h > 0 and self.out_w > 0, "output resolution must be positive"
        assert self.max_range > self.min_range >= 0.0, "require max_range > min_range >= 0"
        assert self.hm_x_max > self.hm_x_min and self.hm_y_max > self.hm_y_min, "bad height-map extent"
        assert self.hm_res > 0.0, "height-map resolution must be positive"
        assert self.hm_loss_blur_cells >= 0.0, "hm_loss_blur_cells must be >= 0"
