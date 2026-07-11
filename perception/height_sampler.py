"""Differentiable local terrain height map via ``F.grid_sample``.

Samples the terrain heightfield on a robot-centred, yaw-aligned grid -- the same
idea as the height scan in Rudin's ``legged_gym`` (``_init_height_points`` /
``_get_heights``), but the lookup is done with bilinear ``F.grid_sample`` instead
of integer indexing, which makes the output **differentiable w.r.t. the robot's
planar position** (and yaw). This is the one perception signal that already
carries gradients, so it can be used directly inside the SRBD back-prop as a
privileged terrain observation.

The world->cell mapping matches ``env._terrain_height`` exactly
(``ix = (x - x_offset)/h_scale``, ``iy = (y - y_offset)/h_scale``, rows<->x,
cols<->y); the only difference is bilinear interpolation instead of rounding.
Everything is a single fused torch op, GPU-resident, no host transfer.
"""

import math

import torch
import torch.nn.functional as F

from .config import PerceptionCfg
from .terrain_mesh import TerrainField


def _gaussian_blur_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (1, 1, H, W) image, replicate-padded.

    Args:
        img: (1, 1, H, W) float tensor.
        sigma: Gaussian standard deviation in pixels (> 0).

    Returns:
        (1, 1, H, W) blurred tensor on the same device/dtype.
    """
    # Replicate padding must be smaller than the padded dim, so clamp the radius
    # to the field size (tiny fields, e.g. the 2x2 flat fallback, get a truncated
    # kernel or -- at radius 0 -- pass through unchanged, which is exact for them).
    radius = max(1, int(math.ceil(3.0 * sigma)))
    radius = min(radius, img.shape[-2] - 1, img.shape[-1] - 1)
    if radius < 1:
        return img
    x = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    k_row = kernel.view(1, 1, 1, -1)
    k_col = kernel.view(1, 1, -1, 1)
    out = F.pad(img, (radius, radius, 0, 0), mode="replicate")
    out = F.conv2d(out, k_row)
    out = F.pad(out, (0, 0, radius, radius), mode="replicate")
    out = F.conv2d(out, k_col)
    return out


class TerrainHeightSampler:
    """Sample a local height map around each robot from the terrain heightfield.

    Args:
        field: Normalised terrain description (heightfield + scale + offsets).
        cfg: Perception configuration (grid extent/resolution, subtract-base-z).
        device: Torch device the sampler and its buffers live on.
    """

    def __init__(self, field: TerrainField, cfg: PerceptionCfg, device: str = "cuda") -> None:
        """Load the heightfield onto ``device`` and build the sample grid.

        See the class docstring for argument descriptions.
        """
        self.cfg = cfg
        self.device = device
        self.horizontal_scale = float(field.horizontal_scale)
        self.x_offset = float(field.x_offset)
        self.y_offset = float(field.y_offset)

        # Heightfield as a (1, 1, rows, cols) image for grid_sample.
        hf = torch.as_tensor(field.heightfield_m, dtype=torch.float32, device=device)
        self.rows, self.cols = hf.shape
        self.heightfield = hf.view(1, 1, self.rows, self.cols)

        # Smoothed copy for the loss path (see PerceptionCfg.hm_loss_blur_cells):
        # blurring turns stair risers into ramps so grid_sample's bilinear
        # gradient carries a usable terrain slope instead of a one-cell spike.
        if cfg.hm_loss_blur_cells > 0.0:
            self.heightfield_smooth = _gaussian_blur_2d(self.heightfield, cfg.hm_loss_blur_cells)
        else:
            self.heightfield_smooth = self.heightfield

        self.grid_points = self._init_grid_points()  # (n_pts, 2) body-frame x,y
        self.num_points = self.grid_points.shape[0]

    def _init_grid_points(self) -> torch.Tensor:
        """Build the body-frame sample grid from the configured extent/resolution.

        Also records ``self.grid_shape = (nx, ny)`` so a flat (B, n_pts) height
        vector can be reshaped back into a 2-D map for visualisation.

        Returns:
            (n_pts, 2) tensor of (x, y) offsets in the robot body frame, x
            forward and y left, matching Rudin's height-point layout.
        """
        c = self.cfg
        # +res/2 so the inclusive upper bound is reached despite float stepping.
        xs = torch.arange(c.hm_x_min, c.hm_x_max + c.hm_res / 2.0, c.hm_res, device=self.device)
        ys = torch.arange(c.hm_y_min, c.hm_y_max + c.hm_res / 2.0, c.hm_res, device=self.device)
        self.grid_shape = (xs.shape[0], ys.shape[0])  # (nx, ny)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (n_pts, 2)

    def _sample_field(self, world_x: torch.Tensor, world_y: torch.Tensor, smooth: bool) -> torch.Tensor:
        """Bilinearly sample the heightfield at world-frame (x, y) points.

        Shared core of :meth:`sample` and :meth:`sample_points`: maps world
        coordinates to fractional heightfield indices (same convention as
        ``env._terrain_height``: rows along x, cols along y) and reads the field
        with ``F.grid_sample``. Differentiable in ``world_x`` / ``world_y``.

        Args:
            world_x: (B, N) world x coordinates, metres.
            world_y: (B, N) world y coordinates, metres.
            smooth: Sample the Gaussian-blurred loss-path field instead of the
                exact one.

        Returns:
            (B, N) terrain heights, metres.
        """
        B, N = world_x.shape
        i_idx = (world_x - self.x_offset) / self.horizontal_scale
        j_idx = (world_y - self.y_offset) / self.horizontal_scale

        # Fractional indices -> grid_sample coords in [-1, 1] (align_corners=True):
        # width axis maps to columns (j), height axis to rows (i).
        gx = 2.0 * j_idx / max(self.cols - 1, 1) - 1.0
        gy = 2.0 * i_idx / max(self.rows - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(B, N, 1, 2)

        field = self.heightfield_smooth if smooth else self.heightfield
        field = field.expand(B, -1, -1, -1)  # (B,1,rows,cols), no copy
        sampled = F.grid_sample(
            field, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled.view(B, N)

    def sample_points(self, world_xy: torch.Tensor, smooth: bool = True) -> torch.Tensor:
        """Sample terrain height at arbitrary world-frame points.

        Differentiable in ``world_xy`` -- this is the loss-path entry point: feed
        SRBD-predicted base/foot positions and the terrain slope flows back as a
        gradient. Defaults to the blurred field for well-behaved gradients on
        stepped terrain (the exact field's bilinear gradient is zero on flat
        treads and a spike at risers).

        Args:
            world_xy: (B, N, 2) world-frame (x, y) query points, metres.
            smooth: Use the Gaussian-blurred loss field (default) or the exact one.

        Returns:
            (B, N) absolute world terrain heights, metres.
        """
        return self._sample_field(world_xy[..., 0], world_xy[..., 1], smooth=smooth)

    def sample(self, base_pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Sample terrain height on the yaw-aligned grid around each robot.

        Differentiable in ``base_pos`` (and ``yaw``): gradients flow through the
        world->normalised-grid mapping into ``F.grid_sample``.

        Args:
            base_pos: (B, 3) world-frame base position (x, y, z), metres.
            yaw: (B,) base yaw in radians (grid is rotated by this so the map is
                expressed in the robot's heading frame).

        Returns:
            (B, n_pts) terrain heights, metres. If ``cfg.hm_subtract_base_z`` the
            value is ``base_z - terrain_z`` (height of the base above the ground,
            as in Rudin); otherwise it is the absolute world terrain height.
        """
        B = base_pos.shape[0]
        base_xy = base_pos[:, :2]                     # (B, 2)
        cos_y = torch.cos(yaw).view(B, 1)             # (B, 1)
        sin_y = torch.sin(yaw).view(B, 1)

        px = self.grid_points[:, 0].view(1, -1)       # (1, n_pts)
        py = self.grid_points[:, 1].view(1, -1)

        # Rotate body-frame grid into the world frame, then translate to base xy.
        world_x = base_xy[:, 0:1] + px * cos_y - py * sin_y   # (B, n_pts)
        world_y = base_xy[:, 1:2] + px * sin_y + py * cos_y

        terrain_z = self._sample_field(world_x, world_y, smooth=False)

        if self.cfg.hm_subtract_base_z:
            return base_pos[:, 2:3] - terrain_z
        return terrain_z
