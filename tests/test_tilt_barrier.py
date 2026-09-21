"""Soft tilt barrier: the geometry, the hinge, and the gradient.

Needs torch but not isaacgym, so it runs anywhere the training stack is installed and takes
about a second. What it protects, in order of severity:

  * The **sign**. `utils_math.tilt_barrier` reads the tilt off the z component of the
    gravity projection. Flip that sign and the term rewards falling over -- a training run
    would still complete, still log a plausible-looking loss, and quietly optimise for the
    opposite of what the term is meant to do. Nothing else in the repo would catch it.
  * The **relationship to loss_gproj**, via the identity |g_xy/g|^2 + cos_tilt^2 == 1. The
    two terms share `gproj_seq`; this pins them together.
  * The **hinge**, including that the term is exactly inert -- value *and* gradient --
    below it. That is what makes TILT_W=0 a byte-for-byte reproduction of the objective
    without the barrier, and what keeps the term silent at the ~16 deg lean the policy
    walks with.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_math import quat_to_rot, tilt_barrier

G = 9.81                      # EnvCfg.g
TILT_ON = 0.6                 # config.TILT_ON default [rad]
COS_ON = math.cos(TILT_ON)    # 0.8253356
FALL_TILT = 0.9               # EnvCfg.fall_tilt_thresh [rad]


def quat_wxyz(roll, pitch, yaw):
    """(roll, pitch, yaw) -> quaternion in wxyz order, ZYX composition.

    Duplicates utils_math.quat_from_rpy's algebra rather than calling it, because that
    helper returns a ``gymapi.Quat`` and gymapi is optional on the machines this test is
    meant to run on.
    """
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def gravity_projection(rpy_list):
    """(B,3) body-frame gravity, built exactly the way train.py's rollout builds it.

    Mirrors train.py's ``q_b = env.srbd_q; R_b = quat_to_rot(...); einsum('bji,j->bi', ...)``
    so this test exercises the real expression rather than a convenient rewrite of it.
    """
    q = torch.tensor([quat_wxyz(*rpy) for rpy in rpy_list], dtype=torch.float32)
    R = quat_to_rot(q[:, 0], q[:, 1], q[:, 2], q[:, 3], "cpu")      # (B,3,3)
    g_w = torch.tensor([0.0, 0.0, -G], dtype=torch.float32)
    return torch.einsum('bji,j->bi', R, g_w)                        # (B,3)


# ---------------------------------------------------------------------------
# 1. Sign and convention
# ---------------------------------------------------------------------------

RPY_CASES = [
    (0.0, 0.0, 0.0),
    (0.9, 0.0, 0.0),
    (0.0, 0.9, 0.0),
    (0.3, -0.7, 1.1),
    (0.45, 0.45, -2.0),
    (-0.8, 0.2, 3.0),
]


@pytest.mark.parametrize("roll,pitch,yaw", RPY_CASES)
def test_cos_tilt_is_cos_roll_times_cos_pitch(roll, pitch, yaw):
    """-g_body_z / g is R[2,2] = cos(roll)*cos(pitch), and carries no yaw.

    This is the identity the whole term rests on: it is what makes the barrier a
    relaxation of env.step's ``|roll| > thresh or |pitch| > thresh`` rule rather than some
    other quantity that merely correlates with it.
    """
    g_body = gravity_projection([(roll, pitch, yaw)])
    cos_tilt = float(-g_body[0, 2] / G)
    assert cos_tilt == pytest.approx(math.cos(roll) * math.cos(pitch), abs=1e-6)


def test_cos_tilt_is_yaw_invariant():
    """Spinning on the spot is not tilting; the term must not see yaw at all."""
    g_body = gravity_projection([(0.4, -0.25, y) for y in (-3.0, -1.0, 0.0, 1.0, 3.0)])
    cos_tilt = -g_body[:, 2] / G
    assert float(cos_tilt.max() - cos_tilt.min()) == pytest.approx(0.0, abs=1e-6)


def test_upright_projects_to_minus_g_on_z():
    """The convention anchor: a level body sees gravity as (0, 0, -g)."""
    g_body = gravity_projection([(0.0, 0.0, 0.0)])
    assert g_body[0].tolist() == pytest.approx([0.0, 0.0, -G], abs=1e-5)


# ---------------------------------------------------------------------------
# 2. The invariant tying this term to loss_gproj
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("roll,pitch,yaw", RPY_CASES)
def test_gproj_and_cos_tilt_are_the_two_halves_of_a_unit_vector(roll, pitch, yaw):
    """loss_gproj's per-sample value + cos_tilt^2 == 1.

    R's third row is a unit vector for a unit quaternion, which is what
    use_strict_alpha_align guarantees by re-normalising env.srbd_q every step. So the
    existing attitude term measures sin^2(tilt) and this one hinges on cos(tilt) -- the
    same angle. A change to either that broke the pairing shows up here.
    """
    g_body = gravity_projection([(roll, pitch, yaw)])
    sin_sq = float(((g_body[0, :2] / G) ** 2).sum())
    cos_tilt = float(-g_body[0, 2] / G)
    assert sin_sq + cos_tilt ** 2 == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Inert below the hinge -- value and gradient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("roll", [0.0, 0.1, 0.3, 0.5, 0.59])
def test_below_the_hinge_the_term_is_exactly_zero(roll):
    """Including 0.3 rad (17 deg), which is where the trained policy actually walks."""
    loss, frac = tilt_barrier(gravity_projection([(roll, 0.0, 0.0)]), G, COS_ON)
    assert float(loss) == 0.0
    assert float(frac) == 0.0


def test_below_the_hinge_the_gradient_is_exactly_zero():
    """Not merely small.

    A term that leaked gradient at the operating point would be a re-weighting of
    loss_gproj rather than a barrier, and TILT_W=0 would stop being inert.
    """
    g_body = gravity_projection([(0.3, 0.0, 0.0)]).requires_grad_(True)
    loss, _ = tilt_barrier(g_body, G, COS_ON)
    loss.backward()
    assert float(g_body.grad.abs().max()) == 0.0


def test_at_the_hinge_the_term_is_still_zero():
    """relu boundary: tilt_on itself costs nothing.

    Only the value is asserted, not frac_active. Exactly at the hinge the sign of
    ``cos_on - cos_tilt`` is decided by float32 rounding in ``quat_to_rot`` (r22 is
    ``1 - 2*(qx^2 + qy^2)``, cos_on comes from math.cos in double), so ``> 0`` is a coin
    flip there and asserting it would be a flaky test rather than a property. The
    surrounding cases at 0.59 and 0.7 pin the behaviour either side of it.
    """
    loss, _ = tilt_barrier(gravity_projection([(TILT_ON, 0.0, 0.0)]), G, COS_ON)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 4. Exact past the hinge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("roll", [0.7, FALL_TILT, 1.2])
def test_past_the_hinge_matches_the_closed_form(roll):
    loss, frac = tilt_barrier(gravity_projection([(roll, 0.0, 0.0)]), G, COS_ON)
    assert float(loss) == pytest.approx((COS_ON - math.cos(roll)) ** 2, abs=1e-6)
    assert float(frac) == 1.0


def test_value_at_the_termination_threshold():
    """The value at the termination threshold, pinned so the hinge cannot drift."""
    loss, _ = tilt_barrier(gravity_projection([(FALL_TILT, 0.0, 0.0)]), G, COS_ON)
    assert float(loss) == pytest.approx(0.0415041, abs=1e-6)


def test_the_barrier_is_monotone_in_tilt():
    """More tilt must never cost less."""
    vals = [float(tilt_barrier(gravity_projection([(r, 0.0, 0.0)]), G, COS_ON)[0])
            for r in (0.0, 0.4, 0.6, 0.75, 0.9, 1.1, 1.3)]
    assert vals == sorted(vals)


# ---------------------------------------------------------------------------
# 5. Conservative against the per-axis termination rule
# ---------------------------------------------------------------------------

def test_combined_tilt_fires_where_neither_axis_would_terminate():
    """roll = pitch = 0.45 rad terminates neither axis (both < 0.9) but does trip the
    barrier, because cos_tilt is the combined tilt cos(roll)*cos(pitch) = 0.8108.

    The barrier is therefore conservative with respect to the fall rule: it speaks
    slightly early, never late, which is the safe direction for a term whose whole job is
    to act before the robot is committed to falling.
    """
    loss, frac = tilt_barrier(gravity_projection([(0.45, 0.45, 0.0)]), G, COS_ON)
    assert 0.45 < FALL_TILT                      # neither axis terminates
    assert float(frac) == 1.0                    # but the barrier is active
    assert float(loss) == pytest.approx(2.1114e-4, rel=1e-3)


# ---------------------------------------------------------------------------
# 6. The gradient points back towards upright
# ---------------------------------------------------------------------------

def test_gradient_pushes_the_body_back_upright():
    """d(loss)/d(g_body_z) must be positive past the hinge.

    cos_tilt = -g_body_z/g, so descending that gradient drives g_body_z down (more
    negative), which is cos_tilt going up -- the body righting itself. A sign error here
    is the failure mode this whole file exists for: the run would train normally and
    optimise for tipping over.
    """
    g_body = gravity_projection([(1.0, 0.0, 0.0)]).requires_grad_(True)
    loss, _ = tilt_barrier(g_body, G, COS_ON)
    loss.backward()
    assert float(g_body.grad[0, 2]) > 0.0


# ---------------------------------------------------------------------------
# 7. Batch reduction and the activation counter
# ---------------------------------------------------------------------------

def test_loss_is_the_mean_over_the_batch_and_frac_counts_the_active_share():
    """Three of four samples are past the hinge; the fourth must dilute the mean.

    tilt_frac_active is the column that separates "the term did nothing" from "the term
    never fired" when a training arm comes back flat, so its arithmetic is pinned here.
    """
    rolls = [0.0, 0.9, 1.0, 1.1]
    loss, frac = tilt_barrier(gravity_projection([(r, 0.0, 0.0) for r in rolls]), G, COS_ON)
    expected = sum(max(0.0, COS_ON - math.cos(r)) ** 2 for r in rolls) / len(rolls)
    assert float(loss) == pytest.approx(expected, abs=1e-6)
    assert float(frac) == pytest.approx(0.75, abs=1e-6)


def test_it_accepts_the_stacked_rollout_shape():
    """train.py hands it gproj_seq, which is (T, B, 3), not (B, 3)."""
    g_body = gravity_projection([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]).view(2, 1, 3)
    loss, frac = tilt_barrier(g_body, G, COS_ON)
    assert loss.dim() == 0 and frac.dim() == 0
    assert float(frac) == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. The knobs behave
# ---------------------------------------------------------------------------

def test_a_later_hinge_is_a_weaker_term():
    """cos is decreasing in tilt, so a smaller cos_on means the barrier engages later.

    Guards the direction of the TILT_ON knob: the obvious response to a null result is
    "lower TILT_ON so it fires more often", and that only holds if this is true.
    """
    g_body = gravity_projection([(0.8, 0.0, 0.0)])
    early = float(tilt_barrier(g_body, G, math.cos(0.45))[0])
    default = float(tilt_barrier(g_body, G, COS_ON)[0])
    late = float(tilt_barrier(g_body, G, math.cos(FALL_TILT))[0])
    assert early > default > late == 0.0


def test_config_defaults_are_the_inert_ones():
    """A plain `python train.py` must not have picked up a tilt barrier by accident."""
    from config import EnvCfg
    cfg = EnvCfg()
    assert cfg.tilt_w == 0.0
    assert cfg.tilt_on == pytest.approx(0.6)
    assert cfg.tilt_on < cfg.fall_tilt_thresh     # hinge before the kill line
