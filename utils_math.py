"""Math and reproducibility helpers shared across the project.

Quaternion utilities (wxyz convention unless stated otherwise), gravity
projection, seeding and simple signal smoothing. All tensor helpers support
both single inputs and batches of shape (B, ...).
"""

import math
import random

import numpy as np

# NOTE: Isaac Gym must be imported before torch (hard requirement of isaacgym).
try:
    from isaacgym import gymapi
except Exception:
    gymapi = None

import torch


def set_seed(seed: int = 0):
    """Seed python, numpy and torch (CPU + CUDA) and force deterministic cuDNN.

    Args:
        seed: The seed applied to every RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def moving_average(x: np.ndarray, k: int = 25) -> np.ndarray:
    """Box-filter smoothing used for the training curves.

    Args:
        x: 1-D array of samples.
        k: Window length; ``k <= 1`` returns ``x`` unchanged.

    Returns:
        Array of the same length (``np.convolve`` with ``mode="same"``).
    """
    if k <= 1:
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")


def quat_to_rot(qw, qx, qy, qz, device):
    """Quaternion (wxyz components) -> rotation matrix.

    Args:
        qw, qx, qy, qz: Quaternion components, either scalars (0-dim tensors)
            or batched tensors of shape (B,).
        device: Device the resulting matrix is placed on.

    Returns:
        (3, 3) rotation matrix for scalar input, (B, 3, 3) for batched input.
    """
    r00 = 1 - 2 * (qy*qy + qz*qz)
    r01 = 2 * (qx*qy - qz*qw)
    r02 = 2 * (qx*qz + qy*qw)
    r10 = 2 * (qx*qy + qz*qw)
    r11 = 1 - 2 * (qx*qx + qz*qz)
    r12 = 2 * (qy*qz - qx*qw)
    r20 = 2 * (qx*qz - qy*qw)
    r21 = 2 * (qy*qz + qx*qw)
    r22 = 1 - 2 * (qx*qx + qy*qy)

    is_batched = qw.dim() > 0
    stack_dim = 1 if is_batched else 0

    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=stack_dim).to(device)


def project_gravity_to_body(q_wxyz, g, device):
    """Project the world gravity vector into the body frame.

    Used for the gravity-projection observation and the corresponding loss
    term (a level body sees gravity as (0, 0, -g)).

    Args:
        q_wxyz: Base orientation quaternion components (w, x, y, z).
        g: Gravity magnitude in m/s^2.
        device: Torch device.

    Returns:
        (3,) gravity vector expressed in the body frame.
    """
    qw, qx, qy, qz = q_wxyz
    R = quat_to_rot(qw, qx, qy, qz, device)
    g_w = torch.tensor([0.0, 0.0, -g], dtype=torch.float32, device=device)
    return R.t().matmul(g_w)  # (3,)


def tilt_barrier(g_body, g, cos_on):
    """Soft one-sided barrier on body tilt, read off the gravity projection (Step 1c).

    The training objective has no term on the tilt *angle*: ``loss_omega`` penalises
    roll/pitch angular *velocity*, and ``term_penalty`` only ever reaches the plotted
    episodic reward, never the backward pass -- while every termination on the Rudin
    curriculum is a tilt fall (CAMPAIGN_FINDINGS.md 18.4, 20.8). This is that missing
    term, hinged so it stays silent during normal walking.

    Geometry, so the sign cannot drift: the gravity projection is
    ``g_body = R(q)^T @ (0, 0, -g)``, whose z component is ``-g * R[2,2]``. R's third row
    is the world z-axis written in body coordinates, so

        cos_tilt := -g_body_z / g = R[2,2] = 1 - 2*(qx^2 + qy^2) = cos(roll) * cos(pitch)

    -- the cosine of the angle between the body z-axis and world up, independent of yaw.
    That is the same quantity ``env.step``'s termination rule thresholds, reached without
    trig and without an ``asin`` singularity. Two consequences worth knowing:

    * It pairs exactly with ``loss_gproj``: for a unit quaternion R's third row is a unit
      vector, so ``|g_xy / g|^2 + cos_tilt^2 == 1`` per sample. (Unit-length is what
      ``use_strict_alpha_align`` guarantees by re-normalising ``env.srbd_q`` every step.)
    * The termination rule is per-axis (``|roll| > thresh`` *or* ``|pitch| > thresh``)
      while ``cos_tilt`` is the combined tilt, so the barrier is conservative: it fires on
      roll = pitch = 0.45 rad, which terminates neither axis. That is the safe direction.

    Args:
        g_body: Gravity projected into the body frame, shape (..., 3) -- e.g. the (T, B, 3)
            stack ``train.py`` already builds for ``loss_gproj``. Differentiable in, and
            only in, whatever produced it; detach it to get the value without the graph.
        g: Gravity magnitude in m/s^2, matching the one used to build ``g_body``.
        cos_on: ``cos(tilt_on)`` -- the cosine of the tilt angle at which the barrier
            starts pushing. Larger means it engages earlier (cos is decreasing in tilt).

    Returns:
        ``(loss, frac_active)``: the mean squared hinge violation, and the fraction of
        samples past the hinge. ``frac_active`` is detached and is not optional in the
        logs -- without it a null result cannot be told apart from a term that never fired.
    """
    cos_tilt = -g_body[..., 2] / g
    gap = torch.relu(cos_on - cos_tilt)
    return (gap ** 2).mean(), (gap.detach() > 0).to(gap.dtype).mean()


def quat_from_rpy(roll: float, pitch: float, yaw: float):
    """Euler angles (roll, pitch, yaw in radians) -> ``gymapi.Quat`` (x, y, z, w).

    Standard ZYX (yaw-pitch-roll) composition, returned in Isaac Gym's xyzw
    quaternion order for use in actor poses.
    """
    cr, sr = math.cos(roll*0.5),  math.sin(roll*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cy, sy = math.cos(yaw*0.5),   math.sin(yaw*0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return gymapi.Quat(x, y, z, w)


def quat_rotate_inverse_wxyz(q_wxyz, v, device):
    """Rotate a vector from the world frame into the body frame.

    Computes ``v_body = R(q)^T @ v_world`` for a single pose or a batch.

    Args:
        q_wxyz: Quaternion in wxyz order, shape (4,) or (B, 4).
        v: World-frame vector, shape (3,) or (B, 3).
        device: Torch device.

    Returns:
        Body-frame vector with the same leading shape as the input.
    """
    is_batched = q_wxyz.dim() == 2

    # Promote single inputs to a batch of one so the einsum below is uniform.
    if not is_batched:
        q_wxyz = q_wxyz.unsqueeze(0)  # (4,) -> (1, 4)
        v = v.unsqueeze(0)            # (3,) -> (1, 3)

    qw, qx, qy, qz = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    R = quat_to_rot(qw, qx, qy, qz, device)  # (B, 3, 3)

    # Summing over i (the rows of R) multiplies by the transpose: R^T @ v.
    v_body = torch.einsum('bij,bi->bj', R, v)  # (B, 3)

    if not is_batched:
        return v_body.squeeze(0)

    return v_body
