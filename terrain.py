# ---------------- Isaac Gym ----------------
from dataclasses import dataclass

import numpy as np

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
    # Optional (num_rows, num_cols, 3) grid of per-cell spawn origins. Only populated
    # for the Rudin curriculum-grid terrain; None for flat / single-heightfield terrain.
    env_origins: object = None


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


# ================== Rudin curriculum-grid terrain ==================
# Ported verbatim from legged_gym (Rudin et al., "Learning to Walk in Minutes ..."):
# legged_gym/legged_gym/utils/terrain.py. Builds one big heightfield arranged as a grid
# where rows = increasing difficulty and columns = terrain-type variety, plus the per-cell
# spawn origins. Copyright (c) 2021 ETH Zurich, Nikita Rudin (BSD-3-Clause).
class Terrain:
    def __init__(self, cfg, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def create_rudin_terrain(gym, sim, rudin_cfg, num_robots):
    """Build Rudin et al.'s curriculum-grid landscape and add it to PhysX.

    Mirrors legged_gym: rows = increasing difficulty, columns = terrain-type variety.
    The mesh is shifted by -border_size in (x, y) so world (0, 0) sits at the grid corner
    (matching legged_robot._create_trimesh). Returns a TerrainData carrying the shared
    heightfield (for spawn-height lookup) plus the (num_rows, num_cols, 3) grid of per-cell
    spawn origins the env uses for Rudin-style robot placement.
    """
    terrain = Terrain(rudin_cfg, num_robots)

    if rudin_cfg.mesh_type != "trimesh":
        raise ValueError(
            f"create_rudin_terrain: unsupported mesh_type={rudin_cfg.mesh_type!r} (use 'trimesh')"
        )

    border = float(rudin_cfg.border_size)

    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = terrain.vertices.shape[0]
    tm_params.nb_triangles = terrain.triangles.shape[0]

    # Shift mesh so world (0,0) is the grid corner (same convention as legged_gym).
    tm_params.transform.p.x = -border
    tm_params.transform.p.y = -border
    tm_params.transform.p.z = 0.0

    # Friction
    tm_params.static_friction = 1.0
    tm_params.dynamic_friction = 1.0
    tm_params.restitution = 0.0

    print("DEBUG 3: before add_triangle_mesh (rudin)", flush=True)
    gym.add_triangle_mesh(
        sim,
        terrain.vertices.flatten(order="C"),
        terrain.triangles.flatten(order="C"),
        tm_params,
    )
    print("DEBUG 4: after add_triangle_mesh (rudin)", flush=True)

    return TerrainData(
        height_field_raw=terrain.height_field_raw,
        horizontal_scale=rudin_cfg.horizontal_scale,
        vertical_scale=rudin_cfg.vertical_scale,
        x_offset=-border,
        y_offset=-border,
        env_origins=terrain.env_origins,
    )
