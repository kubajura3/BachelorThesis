"""
Tests for the differentiable terrain sampling used by the terrain-aware losses.

Covers perception.height_sampler.TerrainHeightSampler on a synthetic ramp+step
heightfield:

  1. sample_points (exact field) reproduces the heightfield values at cell
     centres (same convention as env._terrain_height, minus the rounding)
  2. autograd gradient of sample_points w.r.t. the query xy matches the known
     analytic ramp slope (and a finite-difference probe)
  3. the Gaussian-blurred loss field gives a finite, nonzero gradient on the
     flat tread next to a step riser, where the exact field's gradient is zero
     (the gradient signal the swing-foot clearance loss relies on)
  4. refactor regression: sample() (yaw-aligned body grid) agrees with
     sample_points on the same world points

Self-contained: runs on CPU, no Isaac Gym / Warp / GPU required:

    python tests/test_perception_grad.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.config import PerceptionCfg
from perception.height_sampler import TerrainHeightSampler
from perception.terrain_mesh import TerrainField

torch.manual_seed(0)

H_SCALE = 0.1        # m per cell
RAMP_SLOPE = 0.5     # dz/dx on the ramp section
STEP_HEIGHT = 0.15   # m riser
ROWS, COLS = 80, 60  # 8 m x 6 m
X_OFF, Y_OFF = -4.0, -3.0


def _make_field() -> TerrainField:
    """Ramp along +x for x in [0, 2] m, then a 0.15 m step up at x = 3 m."""
    hf = np.zeros((ROWS, COLS), dtype=np.float32)
    xs = np.arange(ROWS, dtype=np.float32) * H_SCALE + X_OFF  # world x per row
    ramp = np.clip(xs, 0.0, 2.0) * RAMP_SLOPE                 # ramp: 0 -> 1 m
    ramp[xs > 3.0] += STEP_HEIGHT                             # step riser at x=3
    hf[:] = ramp[:, None]
    return TerrainField(
        heightfield_m=hf, horizontal_scale=H_SCALE, x_offset=X_OFF, y_offset=Y_OFF
    )


def _sampler(blur_cells: float) -> TerrainHeightSampler:
    cfg = PerceptionCfg(device="cpu", hm_loss_blur_cells=blur_cells)
    return TerrainHeightSampler(_make_field(), cfg, device="cpu")


def test_values_match_heightfield_at_cell_centres():
    s = _sampler(blur_cells=0.0)
    field = _make_field()
    # A handful of exact cell centres (bilinear at grid nodes is exact).
    ij = torch.tensor([[5, 7], [20, 30], [45, 12], [70, 50]], dtype=torch.float32)
    xy = torch.stack([ij[:, 0] * H_SCALE + X_OFF, ij[:, 1] * H_SCALE + Y_OFF], dim=-1)
    got = s.sample_points(xy.view(1, -1, 2), smooth=False).view(-1)
    want = torch.tensor(
        [field.heightfield_m[int(i), int(j)] for i, j in ij], dtype=torch.float32
    )
    assert torch.allclose(got, want, atol=1e-5), f"{got} vs {want}"
    print("[1] exact values at cell centres: OK")


def test_gradient_matches_ramp_slope():
    s = _sampler(blur_cells=0.0)
    # Points well inside the ramp section (x in (0.3, 1.7)), off cell centres.
    xy = torch.tensor([[[0.53, 0.21], [1.17, -0.42], [0.88, 1.03]]], requires_grad=True)
    z = s.sample_points(xy, smooth=False)
    z.sum().backward()
    grad = xy.grad  # (1, 3, 2)
    slope_x = grad[0, :, 0]
    slope_y = grad[0, :, 1]
    assert torch.allclose(slope_x, torch.full_like(slope_x, RAMP_SLOPE), atol=1e-4), slope_x
    assert torch.allclose(slope_y, torch.zeros_like(slope_y), atol=1e-4), slope_y

    # Finite-difference cross-check at one point.
    eps = 1e-3
    p = torch.tensor([[[0.53, 0.21]]])
    p_dx = p.clone(); p_dx[0, 0, 0] += eps
    fd = (s.sample_points(p_dx, smooth=False) - s.sample_points(p, smooth=False)) / eps
    assert abs(fd.item() - RAMP_SLOPE) < 1e-2, fd.item()
    print("[2] analytic grad == ramp slope (and matches finite differences): OK")


def test_blurred_field_carries_gradient_across_step():
    s = _sampler(blur_cells=2.0)
    # Flat tread just before the riser at x = 3.0 (riser is ~2 cells away).
    xy = torch.tensor([[[2.80, 0.05]]], requires_grad=True)

    z_exact = s.sample_points(xy, smooth=False)
    (g_exact,) = torch.autograd.grad(z_exact.sum(), xy)
    z_smooth = s.sample_points(xy, smooth=True)
    xy.grad = None
    (g_smooth,) = torch.autograd.grad(z_smooth.sum(), xy)

    assert abs(g_exact[0, 0, 0].item()) < 1e-6, "tread should be flat in the exact field"
    assert g_smooth[0, 0, 0].item() > 1e-3, "blurred field must signal the upcoming riser"
    assert torch.isfinite(g_smooth).all()

    # Blur must not invent terrain far from features: far tread heights agree.
    far = torch.tensor([[[-3.5, 0.0]]])
    d = (s.sample_points(far, smooth=True) - s.sample_points(far, smooth=False)).abs()
    assert d.item() < 1e-4, d.item()
    print("[3] blurred field: nonzero riser gradient on the tread, exact far away: OK")


def test_sample_matches_sample_points():
    s = _sampler(blur_cells=0.0)
    base = torch.tensor([[0.7, 0.2, 0.5], [2.4, -0.8, 0.6]])
    yaw = torch.tensor([0.3, -1.1])
    hm = s.sample(base, yaw)  # (2, n_pts), base_z - terrain_z by default

    # Rebuild the same world points and query them through sample_points.
    cos_y, sin_y = torch.cos(yaw).view(2, 1), torch.sin(yaw).view(2, 1)
    px = s.grid_points[:, 0].view(1, -1)
    py = s.grid_points[:, 1].view(1, -1)
    wx = base[:, 0:1] + px * cos_y - py * sin_y
    wy = base[:, 1:2] + px * sin_y + py * cos_y
    tz = s.sample_points(torch.stack([wx, wy], dim=-1), smooth=False)
    want = base[:, 2:3] - tz
    assert torch.allclose(hm, want, atol=1e-6)
    print("[4] sample() vs sample_points() refactor parity: OK")


if __name__ == "__main__":
    test_values_match_heightfield_at_cell_centres()
    test_gradient_matches_ramp_slope()
    test_blurred_field_carries_gradient_across_step()
    test_sample_matches_sample_points()
    print("\nAll perception-gradient tests passed.")
