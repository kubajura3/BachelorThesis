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

import torch
import torch.nn.functional as F

from .config import PerceptionCfg
from .terrain_mesh import TerrainField


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

        # World -> fractional heightfield indices (row index along x, col along y),
        # same convention as env._terrain_height.
        i_idx = (world_x - self.x_offset) / self.horizontal_scale
        j_idx = (world_y - self.y_offset) / self.horizontal_scale

        # Fractional indices -> grid_sample coords in [-1, 1] (align_corners=True):
        # width axis maps to columns (j), height axis to rows (i).
        gx = 2.0 * j_idx / max(self.cols - 1, 1) - 1.0
        gy = 2.0 * i_idx / max(self.rows - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(B, self.num_points, 1, 2)

        heightfield = self.heightfield.expand(B, -1, -1, -1)  # (B,1,rows,cols), no copy
        sampled = F.grid_sample(
            heightfield, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        terrain_z = sampled.view(B, self.num_points)

        if self.cfg.hm_subtract_base_z:
            return base_pos[:, 2:3] - terrain_z
        return terrain_z
