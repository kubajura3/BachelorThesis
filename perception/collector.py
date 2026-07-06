"""Unified terrain-perception entry point.

:class:`PerceptionCollector` wires the depth camera and the height sampler
together behind a single ``collect(base_pos, base_quat_xyzw)`` call that returns
a dict of tensors. This is the one interface every downstream option uses -- they
differ only in what they do *after* the call (end-to-end back-prop, offline
dataset collection for a pretrained encoder, or a hybrid), never in how data is
gathered.

It consumes the environment's own ``terrain.TerrainData`` (or ``None`` for flat
ground) via the :class:`~perception.terrain_mesh.TerrainField` adapter, so there
is no second terrain type and no import of the env's ``terrain`` module.

GPU-residency contract: all buffers are allocated once on ``cfg.device`` and
reused in place; ``collect`` performs no host<->device copy when the inputs are
already on that device (the env's default ``use_gpu_pipeline=True`` case). If the
inputs arrive on a different device it falls back to a one-time upload and warns
once, so it still works under the CPU physics pipeline.
"""

import warp as wp
import torch

from .config import PerceptionCfg
from .terrain_mesh import TerrainField, build_warp_mesh
from .warp_camera import WarpDepthCamera
from .height_sampler import TerrainHeightSampler
from .preprocessing import DepthPreprocessor

_WARP_INITIALIZED = False


def _ensure_warp_init() -> None:
    """Initialise Warp exactly once per process (idempotent guard)."""
    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True


def quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions (both ``xyzw``), batched with broadcasting.

    Args:
        q: (..., 4) quaternions in xyzw order.
        v: (..., 3) vectors (broadcastable against ``q``).

    Returns:
        (..., 3) rotated vectors, ``q * v * q^-1``.
    """
    qvec = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * torch.cross(qvec, v.expand_as(qvec), dim=-1)
    return v + qw * t + torch.cross(qvec, t, dim=-1)


def quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two ``xyzw`` quaternions, batched with broadcasting.

    Args:
        q1: (..., 4) left quaternion (xyzw).
        q2: (..., 4) right quaternion (xyzw).

    Returns:
        (..., 4) product ``q1 (x) q2`` in xyzw order.
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def yaw_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Extract the yaw (rotation about world z) from ``xyzw`` quaternions.

    Args:
        q: (B, 4) quaternions in xyzw order.

    Returns:
        (B,) yaw angles in radians.
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


class PerceptionCollector:
    """Gather depth + height-map terrain perception for a batch of robots.

    Args:
        terrain_data: The environment's ``terrain.TerrainData`` (duck-typed), or
            ``None`` for flat ground (a flat field is synthesised).
        cfg: Perception configuration.
        num_envs: Batch size B (number of parallel robots / poses).
        device: Torch/Warp device; overrides ``cfg.device`` if given.
    """

    def __init__(
        self,
        terrain_data,
        cfg: PerceptionCfg,
        num_envs: int,
        device: str = None,
    ) -> None:
        """Build the mesh, GPU buffers, camera, height sampler and preprocessor.

        See the class docstring for argument descriptions.
        """
        assert cfg.num_sensors == 1, "multi-sensor is a documented future extension"
        _ensure_warp_init()

        self.cfg = cfg
        self.device = device or cfg.device
        self.num_envs = num_envs
        self.num_sensors = cfg.num_sensors
        self._warned_transfer = False

        # Normalise the env terrain (or flat) into a single field description.
        # An already-built TerrainField (visualiser / tests) passes straight through.
        if terrain_data is None:
            self.field = TerrainField.flat()
        elif isinstance(terrain_data, TerrainField):
            self.field = terrain_data
        else:
            self.field = TerrainField.from_terrain_data(terrain_data)

        # Terrain mesh for the camera (keep references alive).
        self._mesh, self._mesh_ids = build_warp_mesh(self.field, device=self.device)

        # Persistent, reused-in-place pose + pixel buffers (GPU-resident).
        self.cam_pos = torch.zeros(num_envs, self.num_sensors, 3, device=self.device, dtype=torch.float32)
        self.cam_quat = torch.zeros(num_envs, self.num_sensors, 4, device=self.device, dtype=torch.float32)
        self.depth_pixels = torch.zeros(
            num_envs, self.num_sensors, cfg.render_h, cfg.render_w, device=self.device, dtype=torch.float32
        )

        self.camera = WarpDepthCamera(
            num_envs=num_envs,
            num_sensors=self.num_sensors,
            height=cfg.render_h,
            width=cfg.render_w,
            fov_deg=cfg.fov_deg,
            max_range=cfg.max_range,
            mesh_ids_array=self._mesh_ids,
            calculate_depth=True,
            device=self.device,
        )
        self.camera.set_image_tensors(self.depth_pixels)
        self.camera.set_pose_tensor(self.cam_pos, self.cam_quat)

        self.height_sampler = TerrainHeightSampler(self.field, cfg, device=self.device)
        self.preprocessor = DepthPreprocessor(cfg)

        self._init_camera_offset()

    def _init_camera_offset(self) -> None:
        """Precompute the body-frame camera translation and orientation offset.

        The orientation offset is a downward pitch about the body y-axis composed
        with the ``[-0.5, 0.5, -0.5, 0.5]`` body->optical-axis conversion (so the
        rendered image axes match the Isaac Gym convention). Stored as ``xyzw``.
        """
        c = self.cfg
        self.offset_p = torch.tensor(
            c.cam_offset_xyz, dtype=torch.float32, device=self.device
        ).view(1, 3)

        half = torch.deg2rad(torch.tensor(c.cam_pitch_deg, device=self.device)) * 0.5
        # Pitch about +y (forward tilts down for positive angle), xyzw.
        q_pitch = torch.tensor(
            [0.0, torch.sin(half).item(), 0.0, torch.cos(half).item()],
            dtype=torch.float32, device=self.device,
        ).view(1, 4)
        axis_quat = torch.tensor(
            [-0.5, 0.5, -0.5, 0.5], dtype=torch.float32, device=self.device
        ).view(1, 4)
        self.offset_q = quat_mul_xyzw(q_pitch, axis_quat)  # (1, 4)

    def _maybe_to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move an input to the collector device, warning once if a copy is needed.

        Zero-copy when already on-device (the default, since the env runs
        ``use_gpu_pipeline=True``). A mismatch means the CPU physics pipeline; we
        upload and nudge toward ``use_gpu_pipeline=True``.
        """
        if t.device.type != torch.device(self.device).type:
            if not self._warned_transfer:
                print(
                    "[perception] inputs are not on", self.device,
                    "- doing a one-time host->device upload per step. For a fully "
                    "zero-copy path, run the env with use_gpu_pipeline=True.",
                )
                self._warned_transfer = True
            return t.to(self.device)
        return t

    @torch.no_grad()
    def _update_camera_pose(self, base_pos: torch.Tensor, base_quat: torch.Tensor) -> None:
        """Write the world-frame camera pose into the persistent Warp buffers.

        Args:
            base_pos: (B, 3) world base position, on ``self.device``.
            base_quat: (B, 4) world base orientation (xyzw), on ``self.device``.

        The camera has no gradient path, so this runs under ``no_grad`` and writes
        in place (``copy_``) so the captured CUDA graph replays with fresh poses.
        """
        cam_pos = quat_rotate_xyzw(base_quat, self.offset_p) + base_pos          # (B, 3)
        cam_quat = quat_mul_xyzw(base_quat, self.offset_q.expand_as(base_quat))  # (B, 4)
        self.cam_pos[:, 0, :].copy_(cam_pos)
        self.cam_quat[:, 0, :].copy_(cam_quat)

    def collect(self, base_pos: torch.Tensor, base_quat_xyzw: torch.Tensor) -> dict:
        """Gather perception for the current robot poses.

        Args:
            base_pos: (B, 3) world-frame base position (x, y, z), metres. Use
                ``env.root_state[:, 0:3]``.
            base_quat_xyzw: (B, 4) world-frame base orientation in **xyzw** order.
                Use ``env.root_state[:, 3:7]`` directly (not the cached wxyz).

        Returns:
            dict with:
                ``depth_clean`` / ``depth_noisy``: (B, num_sensors, out_h, out_w)
                    processed depth in [0, 1] (or metres if not normalised).
                ``height_map``: (B, n_pts) local terrain height, differentiable in
                    ``base_pos``.
                ``cam_pos`` / ``cam_quat``: (B, num_sensors, 3/4) world camera pose
                    used for the render (handy for visualisation/debug).
        """
        base_pos = self._maybe_to_device(base_pos)
        base_quat = self._maybe_to_device(base_quat_xyzw)

        self._update_camera_pose(base_pos, base_quat)
        depth_raw = self.camera.capture()                    # (B, S, render_h, render_w)
        depth_clean, depth_noisy = self.preprocessor(depth_raw)

        yaw = yaw_from_quat_xyzw(base_quat)
        height_map = self.height_sampler.sample(base_pos, yaw)  # differentiable

        return {
            "depth_clean": depth_clean,
            "depth_noisy": depth_noisy,
            "height_map": height_map,
            "cam_pos": self.cam_pos,
            "cam_quat": self.cam_quat,
        }
