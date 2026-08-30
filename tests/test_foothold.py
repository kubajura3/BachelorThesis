"""Perceptive foothold (Step 3): the swing-target geometry and the foothold-quality cost.

Needs torch but not isaacgym, so it runs anywhere the training stack is installed -- the same
split that makes tests/test_tilt_barrier.py and tests/test_perception_grad.py laptop-runnable.
What it protects, in order of severity:

  * **Inertness.** With the Step 3 flags off, the apex formula must reduce *bit-exactly* to the
    pre-Step-3 `0.5*(p0_z + p1_z) + h`. Every claim in STEP3_FOOTHOLD.md that "a run setting no
    FOOT_* variable reproduces the numbers on disk" rests on this, and nothing else in the repo
    would catch a silent drift.
  * **The direction of the quality cost.** `foothold_quality` is the only term that shapes the
    residual toward good ground. If its gradient pointed *into* the edge instead of away from
    it, training would still complete and still log a plausible loss while optimising for
    exactly the wrong thing -- the Step 1(c) sign trap (CAMPAIGN_FINDINGS.md 22) one layer down.
  * **The apex actually clearing a riser**, which is the whole point of sampling the chord
    rather than just its endpoints.

Run:  python tests/test_foothold.py
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.config import PerceptionCfg
from perception.height_sampler import TerrainHeightSampler
from perception.terrain_mesh import TerrainField
from utils_math import foothold_quality, ring_points, swing_apex_z, swing_chord_points

# Deliberately NO module-level torch.manual_seed and no global-RNG draws anywhere in this
# file. Every other suite here seeds at *import* scope, so a manual_seed() call inside a test
# body shifts the stream those modules draw from once pytest starts running -- which silently
# broke tests/test_vectorization.py when this file was added. Local generators keep it neutral.
def _gen(seed):
    """A private RNG, so nothing here perturbs another suite's random inputs."""
    return torch.Generator().manual_seed(seed)


# Same synthetic world as tests/test_perception_grad.py, so the two suites agree on geometry.
H_SCALE = 0.1        # m per cell
STEP_X = 3.0         # riser at world x = 3 m
STEP_HEIGHT = 0.15   # m
ROWS, COLS = 80, 60
X_OFF, Y_OFF = -4.0, -3.0


def _make_field() -> TerrainField:
    """Flat ground with a single 0.15 m step up at x = 3 m (no ramp: isolates the edge)."""
    hf = np.zeros((ROWS, COLS), dtype=np.float32)
    xs = np.arange(ROWS, dtype=np.float32) * H_SCALE + X_OFF
    hf[xs > STEP_X, :] = STEP_HEIGHT
    return TerrainField(heightfield_m=hf, horizontal_scale=H_SCALE,
                        x_offset=X_OFF, y_offset=Y_OFF)


def _sampler(blur_cells: float = 0.0) -> TerrainHeightSampler:
    cfg = PerceptionCfg(device="cpu", hm_loss_blur_cells=blur_cells)
    return TerrainHeightSampler(_make_field(), cfg, device="cpu")


def _sample_legs(sampler, xy, smooth=False):
    """env.terrain_height_legs without an env: (B,4,...,2) -> (B,4,...)."""
    shape = xy.shape[:-1]
    return sampler.sample_points(xy.reshape(shape[0], -1, 2), smooth=smooth).view(*shape)


# ---------------------------------------------------------------------------
# 1. Chord sampling
# ---------------------------------------------------------------------------
def test_chord_endpoints_and_spacing():
    p0 = torch.tensor([[[0.0, 0.0]] * 4])          # (1,4,2)
    p1 = torch.tensor([[[1.0, 2.0]] * 4])
    pts = swing_chord_points(p0, p1, 5)
    assert pts.shape == (1, 4, 5, 2)
    assert torch.allclose(pts[..., 0, :], p0), "first sample must be the liftoff point"
    assert torch.allclose(pts[..., -1, :], p1), "last sample must be the touchdown point"
    # Evenly spaced: consecutive gaps identical.
    gaps = (pts[..., 1:, :] - pts[..., :-1, :]).norm(dim=-1)
    assert torch.allclose(gaps, gaps[..., :1].expand_as(gaps), atol=1e-6)


def test_chord_n2_is_endpoints_only():
    g = _gen(11)
    p0 = torch.randn(2, 4, 2, generator=g)
    p1 = torch.randn(2, 4, 2, generator=g)
    pts = swing_chord_points(p0, p1, 2)
    assert torch.equal(pts[..., 0, :], p0) and torch.equal(pts[..., 1, :], p1)


# ---------------------------------------------------------------------------
# 2. Apex -- inertness first, then the behaviour it exists for
# ---------------------------------------------------------------------------
def test_apex_is_bit_exact_on_flat_ground():
    """THE inertness claim: flat terrain level with the endpoints -> the old formula, exactly.

    Not `allclose`. `torch.equal`. If this ever needs a tolerance, the flag is no longer
    inert when off and every run already on disk stops being a valid baseline.
    """
    p0_z = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    p1_z = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    h = torch.tensor([[0.12, 0.12, 0.12, 0.12]])
    chord_z = torch.zeros(1, 4, 7)                       # flat, level with both endpoints
    old = 0.5 * (p0_z + p1_z) + h
    assert torch.equal(swing_apex_z(chord_z, p0_z, p1_z, h), old)


def test_apex_never_drops_below_the_old_arc():
    """A dip under the chord must not lower the apex -- the max() floor is what guarantees it."""
    p0_z = torch.tensor([[0.4]])
    p1_z = torch.tensor([[0.6]])
    h = torch.tensor([[0.12]])
    chord_z = torch.tensor([[[-5.0, -5.0, -5.0]]])       # a pit under the swing
    old = 0.5 * (p0_z + p1_z) + h
    assert torch.equal(swing_apex_z(chord_z, p0_z, p1_z, h), old)


def test_apex_clears_a_riser_between_the_endpoints():
    """The case the endpoints cannot see: ground between them higher than both."""
    p0_z = torch.tensor([[0.0]])
    p1_z = torch.tensor([[0.0]])
    h = torch.tensor([[0.12]])
    chord_z = torch.tensor([[[0.0, STEP_HEIGHT, 0.0]]])  # riser mid-swing
    apex = swing_apex_z(chord_z, p0_z, p1_z, h)
    assert torch.allclose(apex, torch.tensor([[STEP_HEIGHT + 0.12]]))
    assert float(apex) > float(0.5 * (p0_z + p1_z) + h), "must be above the old apex"


def test_apex_over_a_real_step_beats_the_endpoint_formula():
    """End-to-end against the sampled heightfield, stepping up over the x = 3 m riser."""
    s = _sampler()
    p0_xy = torch.tensor([[[2.75, 0.0]]])               # before the step
    p1_xy = torch.tensor([[[3.25, 0.0]]])               # after it
    p0_z = torch.zeros(1, 1)
    p1_z = torch.full((1, 1), STEP_HEIGHT)
    h = torch.full((1, 1), 0.12)
    chord_z = _sample_legs(s, swing_chord_points(p0_xy, p1_xy, 5))
    apex = swing_apex_z(chord_z, p0_z, p1_z, h)
    old = 0.5 * (p0_z + p1_z) + h                       # 0.075 + 0.12 = 0.195
    # The old apex sits only 0.045 m above the tread it has to land on; the new one clears it.
    assert float(apex) >= STEP_HEIGHT + 0.12 - 1e-6
    assert float(apex) > float(old)


# ---------------------------------------------------------------------------
# 3. Foothold quality -- value, and the direction of its gradient
# ---------------------------------------------------------------------------
def test_ring_points_geometry():
    xy = torch.tensor([[[1.0, 2.0]]])
    pts = ring_points(xy, radius=0.06, k=8)
    assert pts.shape == (1, 1, 8, 2)
    r = (pts - xy.unsqueeze(-2)).norm(dim=-1)
    assert torch.allclose(r, torch.full_like(r, 0.06), atol=1e-6)


def test_quality_is_zero_on_flat_ground_and_large_at_an_edge():
    s = _sampler()
    flat = torch.tensor([[[1.0, 0.0]]])                 # far from the riser
    edge = torch.tensor([[[STEP_X, 0.0]]])              # straddling it
    out = []
    for xy in (flat, edge):
        ring = ring_points(xy, 0.06, 8)
        out.append(float(foothold_quality(_sample_legs(s, ring), _sample_legs(s, xy))))
    q_flat, q_edge = out
    assert q_flat < 1e-8, "flat tread must be free"
    assert q_edge > 1e-3, "an edge must be expensive"
    assert q_edge > 1000 * max(q_flat, 1e-12)


def test_quality_gradient_pushes_the_foothold_away_from_the_edge():
    """The sign test. A foothold just short of the riser must be pushed further back (-x).

    Gradient descent moves xy by -grad, so d(cost)/dx > 0 means "step back from the edge".
    Checked against a finite difference so a sign flip cannot hide behind autograd.
    """
    s = _sampler(blur_cells=1.0)   # a little blur so the exact field's flat plateaus carry slope
    xy = torch.tensor([[[STEP_X - 0.04, 0.0]]], requires_grad=True)
    ring = ring_points(xy, 0.06, 8)
    cost = foothold_quality(_sample_legs(s, ring), _sample_legs(s, xy)).sum()
    cost.backward()
    gx = float(xy.grad[0, 0, 0])
    assert gx > 0.0, "gradient must push the foothold back from the riser, not into it"

    with torch.no_grad():
        def c(dx):
            q = torch.tensor([[[STEP_X - 0.04 + dx, 0.0]]])
            return float(foothold_quality(_sample_legs(s, ring_points(q, 0.06, 8)),
                                          _sample_legs(s, q)))
        fd = (c(1e-3) - c(-1e-3)) / 2e-3
    assert fd > 0.0 and np.sign(fd) == np.sign(gx)
    assert abs(gx - fd) <= 0.25 * max(abs(fd), 1e-6), "autograd and finite difference disagree"


def test_exact_field_gradient_is_mostly_dead_but_the_blurred_one_is_not():
    """Why cfg.foot_q_smooth defaults to True -- the trap that the cost's *value* hides.

    The ring spread detects an edge on the exact heightfield perfectly well. Its **gradient**
    does not: bilinear grid_sample is piecewise-constant, so unless a ring point lands inside
    a riser cell the derivative is exactly zero and the residual gets no signal to move. The
    blurred field turns the step into a ramp and the gradient exists everywhere.

    If this ever flips, the foothold residual silently stops training over most of the state
    space while every logged loss still looks healthy.
    """
    xs = [STEP_X - 0.30 + 0.03 * i for i in range(21)]
    # One sampler, both fields: `smooth` is exactly what cfg.foot_q_smooth switches, and the
    # blurred copy only exists when the sampler was built with hm_loss_blur_cells > 0.
    s = _sampler(blur_cells=2.0)

    def nonzero_count(smooth):
        n = 0
        for x in xs:
            xy = torch.tensor([[[x, 0.0]]], requires_grad=True)
            foothold_quality(_sample_legs(s, ring_points(xy, 0.06, 8), smooth=smooth),
                             _sample_legs(s, xy, smooth=smooth)).sum().backward()
            n += abs(float(xy.grad[0, 0, 0])) > 1e-12
        return n

    n_exact = nonzero_count(smooth=False)
    n_blur = nonzero_count(smooth=True)
    assert n_blur == len(xs), "the blurred field must be differentiable everywhere (%d/%d)" % (
        n_blur, len(xs))
    assert n_exact < len(xs) // 2, "exact-field gradient was expected to be mostly dead"
    assert n_blur > n_exact


def test_quality_gradient_is_finite_and_nonzero_near_the_edge():
    s = _sampler(blur_cells=1.0)
    xy = torch.tensor([[[STEP_X - 0.04, 0.0]]], requires_grad=True)
    foothold_quality(_sample_legs(s, ring_points(xy, 0.06, 8)),
                     _sample_legs(s, xy)).sum().backward()
    g = xy.grad
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0.0


# ---------------------------------------------------------------------------
# 4. Residual squashing -- the bound the joints can actually deliver
# ---------------------------------------------------------------------------
def _residual(raw, max_m=0.10):
    """Exactly train.py's squashing, kept in one line so the bound is testable."""
    return torch.tanh(raw) * max_m


def test_residual_is_bounded_and_zero_at_zero():
    assert torch.equal(_residual(torch.zeros(3, 4)), torch.zeros(3, 4))
    big = _residual(torch.tensor([[-1e4, 1e4, -3.0, 3.0]]))
    # 1e-6, not 1e-9: float32 0.10 widens to 0.10000000149..., which already exceeds a 1e-9 band.
    assert float(big.abs().max()) <= 0.10 + 1e-6
    assert float(big[0, 0]) < 0 and float(big[0, 1]) > 0, "tanh must preserve sign"


def test_residual_padding_keeps_y_at_zero_when_disabled():
    """train.py pads the x-only residual to (B,4,2); the y column must be exactly zero."""
    res_x = _residual(torch.randn(5, 4, generator=_gen(12)))
    padded = torch.stack([res_x, torch.zeros_like(res_x)], dim=-1)
    assert padded.shape == (5, 4, 2)
    assert torch.equal(padded[..., 0], res_x)
    assert torch.count_nonzero(padded[..., 1]) == 0


# ---------------------------------------------------------------------------
# 5. Detach semantics -- the exact guard against loss_foot's degenerate minimum
# ---------------------------------------------------------------------------
def test_detached_residual_has_no_gradient_path_but_the_plan_does():
    """`res.detach()` in the target and `res` in the plan: one path dead, the other live.

    This is the guard from STEP3_FOOTHOLD.md 4.3. If the detach were dropped, loss_foot could
    be minimised by dragging the target onto the foot rather than moving the foot.
    """
    res = torch.zeros(2, 4, 2, requires_grad=True)
    raibert = torch.randn(2, 4, 2, generator=_gen(13))

    # The detached target carries no graph at all -- calling backward() on it would raise,
    # which is a stronger statement than "the gradient happened to be zero".
    target = raibert + res.detach()
    assert not target.requires_grad, "loss_foot's target must not reach the residual"

    plan = raibert + res
    assert plan.requires_grad, "the quality term's plan must reach the residual"
    plan.sum().backward()
    assert res.grad is not None and float(res.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 6. Full inertness of the target-building path
# ---------------------------------------------------------------------------
def test_target_construction_matches_the_pre_step3_formula_when_flags_are_off():
    """Reproduces gait.py's p1/pm construction both ways and demands bit equality.

    The rewrite swapped `torch.zeros_like` + in-place slice writes for `torch.cat` (needed
    because p_land carries a graph once the residual is on). Values must not have moved.
    """
    g = _gen(1)
    p0 = torch.randn(3, 4, 3, generator=g)
    p_land_xy = torch.randn(3, 4, 2, generator=g)
    last_contact_z = torch.randn(3, 4, generator=g)
    h_leg = torch.full((3, 4), 0.12)

    # -- old --
    p1_old = torch.zeros_like(p0)
    p1_old[..., 0:2] = p_land_xy
    p1_old[..., 2] = last_contact_z
    pm_old = 0.5 * (p0 + p1_old)
    pm_old[..., 2] = 0.5 * (p0[..., 2] + p1_old[..., 2]) + h_leg

    # -- new, flags off --
    p1_new = torch.cat([p_land_xy, last_contact_z.unsqueeze(-1)], dim=-1)
    pm_z = 0.5 * (p0[..., 2] + p1_new[..., 2]) + h_leg
    pm_new = torch.cat([0.5 * (p0[..., 0:2] + p1_new[..., 0:2]), pm_z.unsqueeze(-1)], dim=-1)

    assert torch.equal(p1_old, p1_new)
    assert torch.equal(pm_old, pm_new)


def test_swing_parabola_endpoints_and_x_gradient_profile():
    """Without the no_grad decorator the parabola must still interpolate the three points,
    and d p_swing_x / d p1_x must be s^2 -- the profile STEP3_FOOTHOLD.md 4.2 relies on."""
    def parabola(p0, pm, p1, s):
        c = p0
        b = 4 * (pm - (p0 + p1) / 2.0)
        a = p1 - p0 - b
        return a * (s ** 2) + b * s + c

    p0 = torch.zeros(1, 1, 3)
    p1v = torch.tensor([[[0.3, 0.0, 0.05]]])

    def build_pm(p1_, detach_xy):
        """gait.py builds pm's xy from the ATTACHED p1. detach_xy reproduces the trap below."""
        xy = p1_[..., 0:2].detach() if detach_xy else p1_[..., 0:2]
        return torch.cat([0.5 * (p0[..., 0:2] + xy), torch.full((1, 1, 1), 0.145)], dim=-1)

    pm = build_pm(p1v, detach_xy=False)
    assert torch.allclose(parabola(p0, pm, p1v, torch.zeros(1, 1, 1)), p0)
    assert torch.allclose(parabola(p0, pm, p1v, torch.ones(1, 1, 1)), p1v)

    for s_val in (0.0, 0.25, 0.5, 1.0):
        p1g = p1v.clone().requires_grad_(True)
        parabola(p0, build_pm(p1g, detach_xy=False), p1g,
                 torch.full((1, 1, 1), s_val))[..., 0].sum().backward()
        assert abs(float(p1g.grad[0, 0, 0]) - s_val ** 2) < 1e-6,             "d p_swing_x / d p1_x must be s^2"


def test_detaching_the_apex_xy_would_invert_the_residual_gradient():
    """Why gait.py must build pm's xy from the attached p1 -- a real, silent trap.

    pm's xy is the chord midpoint 0.5*(p0 + p1), so b_x = 4*(pm_x - (p0_x+p1_x)/2) is
    identically zero and the x gradient is a clean s^2. Detach that midpoint and b_x picks up
    a -2 dependence on p1_x, giving 3s^2 - 2s -- **negative for s < 2/3**, i.e. the foothold
    residual would be pushed the wrong way for most of the swing. Nothing would crash and the
    loss would look fine. This test pins the correct branch by showing the broken one differs.
    """
    def parabola(p0, pm, p1, s):
        c = p0
        b = 4 * (pm - (p0 + p1) / 2.0)
        a = p1 - p0 - b
        return a * (s ** 2) + b * s + c

    p0 = torch.zeros(1, 1, 3)
    p1v = torch.tensor([[[0.3, 0.0, 0.05]]])
    s_val = 0.25                                     # inside the s < 2/3 danger zone

    grads = {}
    for tag, detach_xy in (("attached", False), ("detached", True)):
        p1g = p1v.clone().requires_grad_(True)
        xy = p1g[..., 0:2].detach() if detach_xy else p1g[..., 0:2]
        pm = torch.cat([0.5 * (p0[..., 0:2] + xy), torch.full((1, 1, 1), 0.145)], dim=-1)
        parabola(p0, pm, p1g, torch.full((1, 1, 1), s_val))[..., 0].sum().backward()
        grads[tag] = float(p1g.grad[0, 0, 0])

    assert abs(grads["attached"] - s_val ** 2) < 1e-6
    assert abs(grads["detached"] - (3 * s_val ** 2 - 2 * s_val)) < 1e-6
    assert grads["attached"] > 0 > grads["detached"], "the trap flips the gradient's sign"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print("PASS %s" % name)
        except Exception as exc:  # noqa: BLE001 - a standalone runner wants every failure
            failed += 1
            print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
