"""Differentiable Single Rigid Body Dynamics (SRBD) model.

The SRBD model is the differentiable physics surrogate that trains the policy:
Isaac Gym steps the full-fidelity (non-differentiable) robot, while this module
re-integrates the same contact forces analytically so ``loss.backward()`` can
reach the policy through the dynamics. Two numerically equivalent backends are
provided: a pure PyTorch implementation (default) and an optional fused CUDA
kernel (``CUDA_KERNEL_SRBD`` in config.py) wrapped in a
``torch.autograd.Function`` with a hand-written analytic adjoint.
"""

# NOTE: Isaac Gym must be imported before torch (hard requirement of isaacgym).
try:
    from isaacgym import gymapi
except Exception:
    pass

import torch

from config import CUDA_KERNEL_SRBD
from utils_math import quat_rotate_inverse_wxyz, quat_to_rot

# ---------------------------------------------------------------------------
# CUDA kernel dispatch setup
#
# Controlled by CUDA_KERNEL_SRBD in config.py.
# Set to True to use the custom CUDA kernel for _srbd_step.
# Requires building the extension first: python setup.py build_ext --inplace
# Falls back to PyTorch automatically if the extension is not available.
# ---------------------------------------------------------------------------
_USE_CUDA_KERNEL = CUDA_KERNEL_SRBD
_srbd_cuda_ext = None

if _USE_CUDA_KERNEL:
    try:
        import srbd_cuda_ext as _srbd_cuda_ext
        print("[SRBD] Custom CUDA kernel active (CUDA_KERNEL_SRBD=True in config.py).")
    except ImportError as e:
        print(f"[SRBD] WARNING: CUDA_KERNEL_SRBD=True but extension not found ({e}).")
        print("[SRBD]          Falling back to PyTorch implementation.")
        print("[SRBD]          Run: python setup.py build_ext --inplace")
        _USE_CUDA_KERNEL = False


class SRBDStepFunction(torch.autograd.Function):
    """Autograd wrapper around the fused CUDA SRBD kernel.

    Forward calls ``srbd_step_forward`` (one thread per environment); backward
    calls ``srbd_step_backward``, a hand-written analytic adjoint (no PyTorch
    replay). All six tensor inputs (p, v, q, w, f_world, q_ref12) can carry
    gradients. Within a training iteration the SRBD state keeps its grad_fn
    chain back into earlier steps' policy outputs (the alpha-alignment in
    train.py preserves the chain scaled by alpha); detachment happens at the
    iteration boundary, not per step.
    """

    @staticmethod
    def forward(ctx, p, v, q, w, f_world, q_ref12, m, g, Ixx, Iyy, Izz, dt):
        """Run one fused SRBD step on the CUDA kernel; returns (p, v, q, w) at t+dt."""
        # Saved tensors must be contiguous because the backward kernel
        # re-reads them via raw float pointers.
        p = p.contiguous()
        v = v.contiguous()
        q = q.contiguous()
        w = w.contiguous()
        f_world = f_world.contiguous()
        q_ref12 = q_ref12.contiguous()
        ctx.save_for_backward(p, v, q, w, f_world, q_ref12)
        ctx.physics = (m, g, Ixx, Iyy, Izz, dt)

        # _ext returns a std::vector → Python list; autograd.Function expects
        # a tuple of tensors, so unpack and repack explicitly.
        p_new, v_new, q_new, w_new = _srbd_cuda_ext.srbd_step_forward(
            p, v, q, w, f_world, q_ref12,
            m, g, Ixx, Iyy, Izz, dt
        )
        return p_new, v_new, q_new, w_new

    @staticmethod
    def backward(ctx, grad_p_new, grad_v_new, grad_q_new, grad_w_new):
        """Analytic adjoint: gradients for the six tensor inputs (scalars get None)."""
        p, v, q, w, f_world, q_ref12 = ctx.saved_tensors
        m, g, Ixx, Iyy, Izz, dt = ctx.physics

        # ctx.needs_input_grad: slots 0..5 are the tensor inputs;
        # slots 6..11 are the scalars (always None grad).
        needs = ctx.needs_input_grad[:6]
        if not any(needs):
            return (None,) * 12

        g_p, g_v, g_q, g_w, g_f, g_qr = _srbd_cuda_ext.srbd_step_backward(
            p, v, q, w, f_world, q_ref12,
            grad_p_new.contiguous(),
            grad_v_new.contiguous(),
            grad_q_new.contiguous(),
            grad_w_new.contiguous(),
            m, g, Ixx, Iyy, Izz, dt,
        )

        return (
            g_p  if needs[0] else None,
            g_v  if needs[1] else None,
            g_q  if needs[2] else None,
            g_w  if needs[3] else None,
            g_f  if needs[4] else None,
            g_qr if needs[5] else None,
            None, None, None, None, None, None,
        )


class SRBDModel:
    """Differentiable Single Rigid Body Dynamics proxy used during training.

    Holds no state of its own: it proxies attribute access onto the wrapped
    ``env`` via ``__getattr__``/``__setattr__``, so ``self.srbd_p``,
    ``self.cfg``, ``self.device`` etc. all resolve to the corresponding env
    attributes. The class is effectively a mixin split into its own file that
    shares ``RealQuadEnv``'s state namespace.

    The hot path is :meth:`_srbd_step`, which dispatches to either the custom
    CUDA kernel (``CUDA_KERNEL_SRBD=True`` and extension built), wrapped by
    :class:`SRBDStepFunction` for autograd, or the pure PyTorch implementation
    (default). Both paths produce numerically equivalent forward outputs and
    matching gradients.
    """

    def __init__(self, env):
        """Wrap the environment whose state this model reads and writes."""
        object.__setattr__(self, 'env', env)

    def __getattr__(self, name):
        """Resolve unknown attributes on the wrapped env (shared state namespace)."""
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        """Write attributes onto the wrapped env (except the ``env`` reference itself)."""
        if name == 'env':
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def _srbd_init_from_isaac(self):
        """Initialise the SRBD state (p, v, q, w) from the current Isaac state.

        Position/velocity are world-frame, the quaternion is wxyz, and the
        angular velocity is rotated into the body frame (the frame the Euler
        equation in :meth:`_srbd_step` integrates in).
        """
        dev = self.device
        self.srbd_p = self.base_pos.clone()            # (B,3)
        self.srbd_v = self.base_lin_world.clone()      # (B,3)
        self.srbd_q = self.base_quat.clone()           # (B,4)

        self.srbd_w = quat_rotate_inverse_wxyz(self.srbd_q, self.base_ang_world, dev) # (B,3)

    def _quat_norm(self, q):
        """Normalise quaternion(s) to unit length; supports (4,) and (B, 4)."""
        if q.dim() == 1:
            return q / (q.norm() + 1e-9)
        else:
            return q / (q.norm(dim=-1, keepdim=True) + 1e-9)


    def _srbd_step(self, f_world, q_ref12, dt):
        """
        One Euler integration step of the centroidal SRBD dynamics (batch).

        Reads  self.srbd_p, self.srbd_v, self.srbd_q (wxyz), self.srbd_w (body frame)
        Writes self.srbd_p, self.srbd_v, self.srbd_q, self.srbd_w  (in place via the
               SRBDModel proxy, i.e. on the underlying env)

        Args:
            f_world : (B, 4, 3) ground-reaction forces per foot, world frame
            q_ref12 : (B, 12)   reference joint angles (hip, thigh, calf) × 4 legs
            dt      : float     integration step (e.g. cfg.dt = 0.002)

        Dispatch:
            CUDA_KERNEL_SRBD=True  -> SRBDStepFunction.apply (fused CUDA kernel +
                                       autograd wrapper with full gradient flow).
            CUDA_KERNEL_SRBD=False -> the pure PyTorch implementation below
                                       (default; autograd flows natively).
        Both paths produce equivalent results to float32 precision.
        """
        if _USE_CUDA_KERNEL:
            # ---- CUDA kernel path ----
            p_new, v_new, q_new, w_new = SRBDStepFunction.apply(
                self.srbd_p.contiguous(),
                self.srbd_v.contiguous(),
                self.srbd_q.contiguous(),
                self.srbd_w.contiguous(),
                f_world,
                q_ref12,
                float(self.cfg.m),
                float(self.cfg.g),
                float(self.cfg.Ixx),
                float(self.cfg.Iyy),
                float(self.cfg.Izz),
                float(dt),
            )
            self.srbd_p = p_new
            self.srbd_v = v_new
            self.srbd_q = q_new
            self.srbd_w = w_new
        else:
            # ---- Original PyTorch path (unchanged) ----
            dev = self.device
            m, g = self.cfg.m, self.cfg.g

            # Snapshot of current SRBD state (don't modify self.srbd_* midway)
            p = self.srbd_p          # (B,3)
            v = self.srbd_v          # (B,3)
            q = self.srbd_q          # (B,4)
            w = self.srbd_w          # (B,3)

            # ---- Translational part ----
            Fsum = f_world.sum(dim=1) + torch.tensor(
                [0.0, 0.0, -m * g], device=dev
            ).view(1, 3)                               # (B,3)
            a = Fsum / m                                # (B,3)

            # Foot positions from the pre-step state (torque arms use the old pose).
            p_foot = self.foot_positions_srbd(q_ref12)

            # Force arm vectors r: from COM to feet for the whole batch
            r = p_foot - p.unsqueeze(1)                 # (B,4,3) -> automatic broadcasting

            # Batched cross product (world-frame torque)
            tau_world = torch.cross(r, f_world, dim=-1).sum(dim=1)  # (B,3)

            # Generate rotation matrices R for the whole batch at once
            R = quat_to_rot(q[:, 0], q[:, 1], q[:, 2], q[:, 3], dev) # (B,3,3)

            # world -> body: equivalent to R.t() @ tau_world for each batch element
            tau_body = torch.einsum('bji,bj->bi', R, tau_world) # (B,3)

            # Since I is diagonal, treat it as a (3,) vector and use fast broadcasting
            I_diag = torch.tensor([self.cfg.Ixx, self.cfg.Iyy, self.cfg.Izz], device=dev) # (3,)

            # Euler equation: I * wdot = tau - w × (I w) without the loop
            Iw = w * I_diag.unsqueeze(0)                # (B,3) - fast elementwise multiplication
            w_cross_Iw = torch.cross(w, Iw, dim=-1) # (B,3)
            wdot = (tau_body - w_cross_Iw) / I_diag.unsqueeze(0) # (B,3)

            w_new = w + wdot * dt                      # (B,3)

            # Quaternion update using batch operations
            wx, wy, wz = w_new.unbind(dim=-1)           # each has shape (B,)
            z = torch.zeros_like(wx)                    # (B,)

            # Construct batch Omega tensor (B, 4, 4)
            Omega = torch.stack([
                torch.stack([z,  -wx, -wy, -wz], dim=-1),
                torch.stack([wx,  z,   wz, -wy], dim=-1),
                torch.stack([wy, -wz,  z,   wx], dim=-1),
                torch.stack([wz,  wy, -wx,  z ], dim=-1),
            ], dim=1)                                   # (B,4,4)

            # qdot = 0.5 * (Omega @ q) for the whole batch
            qdot = 0.5 * torch.einsum('bij,bj->bi', Omega, q) # (B,4)
            q_new = self._quat_norm(q + qdot * dt)      # (B,4)

            # ---- State update ----
            self.srbd_p = p + v * dt                    # (B,3)
            self.srbd_v = v + a * dt                    # (B,3)
            self.srbd_q = q_new                         # (B,4)
            self.srbd_w = w_new                         # (B,3)



    # ---------------- SRBD-based foot positions ----------------

    def foot_positions_srbd(self, q_ref12):
        """World-frame foot positions predicted from the SRBD state (planar leg FK).

        Uses the SRBD base pose (``srbd_p``, ``srbd_q``) plus a sagittal-plane
        two-link forward kinematics of each leg (thigh/calf angles only; the hip
        abduction joint is neglected). Differentiable in the SRBD state and in
        ``q_ref12`` -- this is the quantity terrain-clearance losses sample at.

        Args:
            q_ref12: (B, 12) reference joint angles (hip, thigh, calf) x 4 legs.

        Returns:
            (B, 4, 3) foot positions in the world frame (FL, FR, RL, RR).
        """
        dev = self.device
        B = self.B
        p_base = self.srbd_p          # (B,3)
        q = self.srbd_q               # (B,4)

        hx, hy = float(self.cfg.hip_offset_x), float(self.cfg.hip_offset_y)
        hip_offsets = torch.tensor([
            [ +hx, +hy, 0.0 ],        # FL, FR, RL, RR
            [ +hx, -hy, 0.0 ],
            [ -hx, +hy, 0.0 ],
            [ -hx, -hy, 0.0 ],
        ], device=dev)                # (4,3)

        L1, L2 = float(self.cfg.leg_l1), float(self.cfg.leg_l2)

        # Get rotation matrices for the whole batch
        R = quat_to_rot(q[:, 0], q[:, 1], q[:, 2], q[:, 3], dev) # (B,3,3)

        # Project hip_offsets from body to world using einsum
        # (B,3,3) @ (4,3) -> (B,4,3)
        hip_world = p_base.unsqueeze(1) + torch.einsum('bij,nj->bni', R, hip_offsets) # (B,4,3)

        # Leg kinematics for the whole batch
        q_leg = q_ref12.view(B, 4, 3)               # (B,4,3)
        q2, q3 = q_leg[:, :, 1], q_leg[:, :, 2]     # (B,4)

        x = L1 * torch.sin(q2) + L2 * torch.sin(q2 + q3) # (B,4)
        z = -L1 * torch.cos(q2) - L2 * torch.cos(q2 + q3) # (B,4)
        y = torch.zeros_like(x)                     # (B,4)

        off_body = torch.stack([x, y, z], dim=-1)  # (B,4,3)

        # Rotate foot positions from body to world for the whole batch
        off_world = torch.einsum('bij,bnj->bni', R, off_body) # (B,4,3)

        return hip_world + off_world                # (B,4,3)
