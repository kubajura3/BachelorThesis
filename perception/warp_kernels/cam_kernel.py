"""Depth-range ray-casting kernel (NVIDIA Warp).

This kernel is a trimmed, self-contained copy of the depth-range kernel from the
MGDP ``warp_sensor`` project (paper: *Multi-Modal Guided Data Perception* -- see
``PERCEPTION_IMPLEMENTATION.md`` references). It is vendored here on purpose: the
thesis repository must run without any dependency on the MGDP code. Only the
single depth-range variant is kept; the lidar / point-cloud / segmentation /
normal kernels are dropped.

Why this exact formulation:
    * **Pinhole model via K_inv.** Pixel coordinates are back-projected through
      the inverse intrinsics to get a per-pixel ray direction, then rotated into
      the world frame by the camera quaternion. This matches a real depth camera.
    * **True planar depth, not radial range.** The raw ray hit distance ``t`` is
      the *range* along the ray. Multiplying by ``dot(rd, rd_principal)``
      projects it onto the principal (optical) axis, giving the *depth* value a
      RealSense-style sensor reports. The far plane is divided by the same factor
      so off-axis rays still reach the configured maximum depth.
    * **Out-parameter query signature.** ``wp.mesh_query_ray(mesh, ro, rd, max,
      t, u, v, sign, n, f)`` is used (rather than the newer return-struct form)
      because it is the signature proven against the Warp build shipped with the
      reference project.
"""

import warp as wp

#: Value written for rays that hit nothing. Large on purpose so misses survive
#: until the downstream clip to ``max_range`` (see :mod:`perception.preprocessing`).
NO_HIT_RAY_VAL = wp.constant(1000.0)


@wp.kernel
def draw_depth_range(
    mesh_ids: wp.array(dtype=wp.uint64),
    cam_pos: wp.array(dtype=wp.vec3, ndim=2),
    cam_quat: wp.array(dtype=wp.quat, ndim=2),
    K_inv: wp.mat44,
    far_plane: float,
    pixels: wp.array(dtype=wp.float32, ndim=4),
    c_x: int,
    c_y: int,
    calculate_depth: bool,
):
    """Cast one ray per pixel against a shared terrain mesh and write depth.

    Launch dims are ``(num_envs, num_sensors, width, height)`` so that
    ``wp.tid()`` unpacks to ``(env_id, cam_id, x, y)`` with ``x`` indexing the
    image width and ``y`` the image height.

    Args:
        mesh_ids: Warp mesh id(s), uint64. A single shared terrain mesh is
            assumed, so ``mesh_ids[0]`` is used for every environment.
        cam_pos: (num_envs, num_sensors) world-frame camera origins (vec3, m).
        cam_quat: (num_envs, num_sensors) world-frame camera orientations (quat,
            xyzw).
        K_inv: Inverse pinhole intrinsics (mat44).
        far_plane: Maximum sensing distance in metres.
        pixels: (num_envs, num_sensors, height, width) output buffer, written in
            place with depth in metres (or ``NO_HIT_RAY_VAL`` on a miss).
        c_x, c_y: Integer principal-point pixel coordinates (image centre).
        calculate_depth: If True return planar depth (projected onto the optical
            axis); if False return radial range along the ray.
    """
    env_id, cam_id, x, y = wp.tid()

    mesh = mesh_ids[0]
    ro = cam_pos[env_id, cam_id]
    quat = cam_quat[env_id, cam_id]

    # Back-project this pixel and the principal point through K_inv to get ray
    # directions, then rotate them into the world frame.
    cam_coords = wp.vec3(float(x), float(y), 1.0)
    cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)
    uv = wp.transform_vector(K_inv, cam_coords)
    uv_principal = wp.transform_vector(K_inv, cam_coords_principal)
    rd = wp.normalize(wp.quat_rotate(quat, uv))
    rd_principal = wp.normalize(wp.quat_rotate(quat, uv_principal))

    # range->depth projection factor (1.0 when returning raw range)
    multiplier = float(1.0)
    if calculate_depth:
        multiplier = wp.dot(rd, rd_principal)

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face = int(0)

    dist = NO_HIT_RAY_VAL
    if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, normal, face):
        dist = multiplier * t

    pixels[env_id, cam_id, y, x] = dist
