# ---------------- Isaac Gym ----------------
from dataclasses import dataclass

try:
    from isaacgym import gymapi, terrain_utils
    ISAAC_AVAILABLE = True
except Exception:
    ISAAC_AVAILABLE = False
    gymapi = None
    terrain_utils = None


@dataclass
class TerrainData:
    """Everything needed to look up the shared terrain's surface height at a world (x, y).

    height_field_raw : (num_rows, num_cols) int16 heightfield (rows ~ world x, cols ~ world y)
    horizontal_scale : meters per heightfield cell
    vertical_scale   : meters per height unit (height_m = raw * vertical_scale)
    x_offset/y_offset: world coords of cell (0, 0) = the mesh transform origin
    """
    height_field_raw: object
    horizontal_scale: float
    vertical_scale: float
    x_offset: float
    y_offset: float


def _setup_physx_stable(sim_params, use_gpu=True):
    if hasattr(sim_params, "substeps"):
        sim_params.substeps = 3
    if not hasattr(sim_params, "physx"):
        return
    ph = sim_params.physx
    if hasattr(ph, "num_position_iterations"):
        ph.num_position_iterations = 12
    if hasattr(ph, "num_velocity_iterations"):
        ph.num_velocity_iterations = 2
    if hasattr(ph, "solver_type"):
        if hasattr(gymapi, "SOLVER_TGS"):
            ph.solver_type = gymapi.SOLVER_TGS
        else:
            try:
                ph.solver_type = 1
            except Exception:
                pass
    if hasattr(ph, "use_gpu"):
        ph.use_gpu = bool(use_gpu)
    if hasattr(ph, "rest_offset"):
        ph.rest_offset = 0.0
    if hasattr(ph, "contact_offset"):
        ph.contact_offset = 0.01
    if hasattr(ph, "bounce_threshold_velocity"):
        ph.bounce_threshold_velocity = 0.2
    if hasattr(ph, "max_depenetration_velocity"):
        ph.max_depenetration_velocity = 1.0
    if hasattr(ph, "default_buffer_size_multiplier"):
        ph.default_buffer_size_multiplier = 2.0
    if hasattr(ph, "enable_stabilization"):
        ph.enable_stabilization = True
    if hasattr(ph, "enable_ccd"):
        ph.enable_ccd = True


# ================== Terrain Creation Tools ==================
def create_ground_plane(gym, sim):
    """Wrap the original flat ground plane into a small function."""
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    print("DEBUG 3: before add_ground", flush=True)
    gym.add_ground(sim, plane_params)
    print("DEBUG 4: after add_ground", flush=True)
    # Flat plane: no heightfield to sample -> spawn height is just h0.
    return None


def create_random_rough_terrain(gym, sim):
    """
    Use isaacgym.terrain_utils to create a large random rough terrain,
    then convert to triangle mesh and add to PhysX.

    For “approximately infinite”, we create an 80m x 80m large terrain,
    centered at (0,0), with robots spawning near the center.
    """
    # Scale parameters
    horizontal_scale = 0.25   # Each heightfield cell is 0.25m
    vertical_scale = 0.005    # Each height unit is 0.005m

    terrain_size = 200        # 80m x 80m
    num_rows = int(terrain_size / horizontal_scale)
    num_cols = int(terrain_size / horizontal_scale)

    # Create a sub-terrain: all random undulations
    sub = terrain_utils.SubTerrain(
        terrain_name="random_uniform",
        width=num_rows,
        length=num_cols,
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale,
    )

    # Random height range (unit: meters)
    # Don't make range too large to avoid spawning with buried feet / too high to jump down
    min_h = -0.02
    max_h = 0.02

    terrain_utils.random_uniform_terrain(
        sub,
        min_height=min_h,
        max_height=max_h,
        step=0.02,           # Step height granularity ~3cm
        downsampled_scale=0.25,
    )

    heightfield = sub.height_field_raw   # (num_rows, num_cols) int16

    # Heightfield -> triangle mesh
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        slope_threshold=1.5,
    )

    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]

    # Center terrain at world origin
    tm_params.transform.p.x = -terrain_size * 0.5
    tm_params.transform.p.y = -terrain_size * 0.5
    tm_params.transform.p.z = 0.0

    # Friction
    tm_params.static_friction = 1.0
    tm_params.dynamic_friction = 1.0
    tm_params.restitution = 0.0

    print("DEBUG 3: before add_triangle_mesh", flush=True)
    gym.add_triangle_mesh(
        sim,
        vertices.flatten(order="C"),
        triangles.flatten(order="C"),
        tm_params,
    )
    print("DEBUG 4: after add_triangle_mesh", flush=True)

    # Retain the heightfield + scales/offsets so the env can sample surface height at any (x, y).
    return TerrainData(
        height_field_raw=heightfield,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        x_offset=tm_params.transform.p.x,
        y_offset=tm_params.transform.p.y,
    )
