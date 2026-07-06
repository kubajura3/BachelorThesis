"""Slim depth camera wrapper around the vendored Warp ray-cast kernel.

Adapted (self-contained, no external import) from the MGDP ``warp_sensor``
``WarpCam`` class, reduced to the depth-range path only. It owns the pinhole
intrinsics, the persistent Warp views of the pose/pixel buffers and an optional
CUDA-graph capture of the kernel launch.

Performance notes:
    * **Zero-copy.** ``set_pose_tensor`` / ``set_image_tensors`` wrap existing
      CUDA torch tensors as Warp arrays via ``wp.from_torch`` (a view). As long
      as the caller writes new poses *in place* into the same torch tensors, no
      host<->device copy ever happens.
    * **CUDA-graph capture.** On CUDA the kernel launch is captured once and
      replayed with ``wp.capture_launch``, removing per-step launch overhead.
      On CPU (used only by the offline visualiser) graph capture is unsupported,
      so the kernel is launched directly.

.. note::
    Differentiability seam -- :meth:`capture` currently returns a plain,
    grad-free depth tensor, which is all any intended downstream use needs. To
    make depth differentiable later, wrap the launch in a
    ``torch.autograd.Function`` backed by ``wp.Tape()``; the public
    ``set_pose_tensor`` / ``capture`` interface stays identical, so nothing
    upstream of this file changes.
"""

import math

import warp as wp

from .warp_kernels.cam_kernel import draw_depth_range


class WarpDepthCamera:
    """Forward-only pinhole depth camera that ray-casts against a Warp mesh.

    Args:
        num_envs: Number of parallel environments (batch dimension).
        num_sensors: Cameras per environment.
        height: Ray-cast image height in pixels.
        width: Ray-cast image width in pixels.
        fov_deg: Horizontal field of view in degrees.
        max_range: Far clip distance in metres.
        mesh_ids_array: Warp ``uint64`` array of terrain mesh id(s); ``[0]`` is
            used as the single shared terrain.
        calculate_depth: Return planar depth (True) or radial range (False).
        device: Warp/torch device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        num_envs: int,
        num_sensors: int,
        height: int,
        width: int,
        fov_deg: float,
        max_range: float,
        mesh_ids_array: wp.array,
        calculate_depth: bool = True,
        device: str = "cuda",
    ) -> None:
        """Store camera settings and precompute the pinhole intrinsics.

        See the class docstring for argument descriptions.
        """
        self.num_envs = num_envs
        self.num_sensors = num_sensors
        self.height = height
        self.width = width
        self.fov = math.radians(fov_deg)
        self.far_plane = float(max_range)
        self.calculate_depth = bool(calculate_depth)
        self.mesh_ids_array = mesh_ids_array
        self.device = device
        self.use_graph = str(device).startswith("cuda")

        self.camera_position_array = None
        self.camera_orientation_array = None
        self.pixels = None
        self.graph = None

        self._init_intrinsics()

    def _init_intrinsics(self) -> None:
        """Build the pinhole intrinsics ``K`` and its inverse from the FOV.

        Uses a standard pinhole model with the principal point at the image
        centre; the focal length follows from the horizontal FOV and the
        vertical FOV is derived from the aspect ratio (square pixels).
        """
        W, H = self.width, self.height
        u_0, v_0 = W / 2.0, H / 2.0
        f = (W / 2.0) / math.tan(self.fov / 2.0)
        vertical_fov = 2.0 * math.atan(H / (2.0 * f))
        alpha_u = u_0 / math.tan(self.fov / 2.0)
        alpha_v = v_0 / math.tan(vertical_fov / 2.0)

        self.K = wp.mat44(
            alpha_u, 0.0, u_0, 0.0,
            0.0, alpha_v, v_0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        self.K_inv = wp.inverse(self.K)
        self.c_x = int(u_0)
        self.c_y = int(v_0)

    def set_image_tensors(self, pixels) -> None:
        """Bind the output depth buffer (zero-copy view of a CUDA torch tensor).

        Args:
            pixels: torch tensor of shape (num_envs, num_sensors, height, width),
                float32, on ``self.device``. Written in place by every capture.
        """
        self.pixels = wp.from_torch(pixels, dtype=wp.float32)

    def set_pose_tensor(self, positions, orientations) -> None:
        """Bind camera pose buffers (zero-copy views of CUDA torch tensors).

        Args:
            positions: (num_envs, num_sensors, 3) world-frame camera origins.
            orientations: (num_envs, num_sensors, 4) world-frame quaternions,
                ``xyzw``.

        The caller must keep writing into these *same* tensors in place so the
        captured CUDA graph replays with fresh data.
        """
        self.camera_position_array = wp.from_torch(positions, dtype=wp.vec3)
        self.camera_orientation_array = wp.from_torch(orientations, dtype=wp.quat)

    def _launch(self) -> None:
        """Launch the depth kernel once over all (env, sensor, x, y) rays."""
        wp.launch(
            kernel=draw_depth_range,
            dim=(self.num_envs, self.num_sensors, self.width, self.height),
            inputs=[
                self.mesh_ids_array,
                self.camera_position_array,
                self.camera_orientation_array,
                self.K_inv,
                self.far_plane,
                self.pixels,
                self.c_x,
                self.c_y,
                self.calculate_depth,
            ],
            device=self.device,
        )

    def capture(self):
        """Render the depth image for the current poses.

        On first call (CUDA) the launch is captured into a replayable graph; on
        subsequent calls the graph is replayed. On CPU the kernel is launched
        directly every call.

        Returns:
            torch tensor (num_envs, num_sensors, height, width) of depth in
            metres, a zero-copy view of the bound pixel buffer.
        """
        assert self.pixels is not None, "call set_image_tensors() first"
        assert self.camera_position_array is not None, "call set_pose_tensor() first"

        if self.use_graph:
            if self.graph is None:
                wp.capture_begin(device=self.device)
                try:
                    self._launch()
                finally:
                    self.graph = wp.capture_end(device=self.device)
            wp.capture_launch(self.graph)
        else:
            self._launch()

        return wp.to_torch(self.pixels)
