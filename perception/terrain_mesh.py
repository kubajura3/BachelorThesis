"""Terrain adapter for perception, plus Warp-mesh construction.

The perception module does **not** define its own terrain type. The environment
already owns ``terrain.TerrainData`` (heightfield + scales + world offsets, and
-- after the small extension for this module -- the world-frame mesh vertices /
triangles). :class:`TerrainField` is a thin adapter that normalises whatever the
env hands us (a ``TerrainData``, or ``None`` for flat ground) into the exact
fields the camera and the height sampler need, without importing ``terrain.py``
(pure duck-typing -> perception stays self-contained).

World<->heightfield convention (identical to ``env._terrain_height``):

    height_m(r, c) = height_field_raw[r, c] * vertical_scale
    x = r * horizontal_scale + x_offset       (rows  <-> world x)
    y = c * horizontal_scale + y_offset       (cols  <-> world y)

This file imports **no** Warp at module load (only ``build_warp_mesh`` needs it,
lazily), so ``TerrainField`` / ``heightfield_to_trimesh`` are importable without
Warp installed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TerrainField:
    """Normalised terrain description consumed by the perception components.

    Attributes:
        heightfield_m: (rows, cols) float32 terrain height in metres.
        horizontal_scale: Metres between adjacent heightfield cells.
        x_offset: World x of heightfield row 0 (= mesh transform origin x).
        y_offset: World y of heightfield col 0 (= mesh transform origin y).
        vertices: Optional (N, 3) float32 world-frame mesh vertices. When present
            the camera ray-casts this exact PhysX mesh (correct vertical stair
            faces); otherwise a mesh is generated from the heightfield.
        triangles: Optional (M, 3) int32 triangle indices into ``vertices``.
    """

    heightfield_m: np.ndarray
    horizontal_scale: float
    x_offset: float
    y_offset: float
    vertices: Optional[np.ndarray] = None
    triangles: Optional[np.ndarray] = None

    @classmethod
    def from_terrain_data(cls, td) -> "TerrainField":
        """Adapt the environment's ``terrain.TerrainData`` (duck-typed).

        Args:
            td: An object exposing ``height_field_raw``, ``vertical_scale``,
                ``horizontal_scale``, ``x_offset``, ``y_offset`` and optionally
                ``vertices`` / ``triangles`` (the env's ``terrain.TerrainData``).

        Returns:
            A :class:`TerrainField` with the heightfield converted to metres and
            the world-frame mesh passed through when available.
        """
        hf_m = np.asarray(td.height_field_raw, dtype=np.float32) * float(td.vertical_scale)
        return cls(
            heightfield_m=hf_m,
            horizontal_scale=float(td.horizontal_scale),
            x_offset=float(td.x_offset),
            y_offset=float(td.y_offset),
            vertices=getattr(td, "vertices", None),
            triangles=getattr(td, "triangles", None),
        )

    @classmethod
    def flat(cls, size: float = 200.0) -> "TerrainField":
        """Flat ground centred on the origin (used when the env terrain is None).

        Args:
            size: Side length of the square ground quad in metres.

        Returns:
            A :class:`TerrainField` with a 2x2 zero heightfield (height map is
            uniformly zero) and one large quad for the camera to hit.
        """
        half = size / 2.0
        verts = np.array(
            [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
            dtype=np.float32,
        )
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        hf = np.zeros((2, 2), dtype=np.float32)
        # 2 cells span the whole quad, so horizontal_scale == size.
        return cls(hf, horizontal_scale=size, x_offset=-half, y_offset=-half, vertices=verts, triangles=tris)

    @classmethod
    def from_heightfield(
        cls,
        heightfield_m: np.ndarray,
        horizontal_scale: float,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
    ) -> "TerrainField":
        """Build a field (and a fresh trimesh) from a metric heightfield.

        Used by the offline visualiser / tests where there is no env terrain.

        Args:
            heightfield_m: (rows, cols) heights in metres.
            horizontal_scale: Cell spacing in metres.
            x_offset, y_offset: World offset of cell (0, 0).

        Returns:
            A populated :class:`TerrainField`.
        """
        verts, tris = heightfield_to_trimesh(heightfield_m, horizontal_scale, (x_offset, y_offset))
        return cls(
            heightfield_m=heightfield_m.astype(np.float32),
            horizontal_scale=float(horizontal_scale),
            x_offset=float(x_offset),
            y_offset=float(y_offset),
            vertices=verts,
            triangles=tris,
        )


def heightfield_to_trimesh(
    heightfield_m: np.ndarray,
    horizontal_scale: float,
    transform_xy: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a metric heightfield into a world-frame triangle mesh (pure numpy).

    Mirrors the vertex layout of ``isaacgym.terrain_utils.convert_heightfield_to
    _trimesh`` (row index -> world x, col index -> world y, value -> world z) but
    without any Isaac Gym dependency, so it runs anywhere. Note this simple
    converter produces *sloped* connections between cells (no vertical faces); the
    env passes the real PhysX mesh for rough/rudin terrain, so this fallback is
    only used for flat ground and synthetic visualiser scenes.

    Args:
        heightfield_m: (rows, cols) heights in metres.
        horizontal_scale: Cell spacing in metres.
        transform_xy: World offset (tx, ty) added to the grid origin.

    Returns:
        (vertices (rows*cols, 3) float32, triangles (2*(rows-1)*(cols-1), 3) int32).
    """
    rows, cols = heightfield_m.shape
    tx, ty = transform_xy
    xs = np.arange(rows, dtype=np.float32) * horizontal_scale + tx
    ys = np.arange(cols, dtype=np.float32) * horizontal_scale + ty
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack([xx, yy, heightfield_m.astype(np.float32)], axis=-1).reshape(-1, 3)

    # Two triangles per grid cell.
    i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    tl = (i * cols + j).reshape(-1)          # top-left vertex index of each cell
    tr = tl + 1
    bl = tl + cols
    br = bl + 1
    tri_a = np.stack([tl, tr, bl], axis=-1)
    tri_b = np.stack([tr, br, bl], axis=-1)
    triangles = np.concatenate([tri_a, tri_b], axis=0).astype(np.int32)
    return vertices.astype(np.float32), triangles


def build_warp_mesh(field: TerrainField, device: str = "cuda"):
    """Create a :class:`warp.Mesh` and its id array from a terrain field.

    Uses the field's world-frame ``vertices`` / ``triangles`` when present (the
    exact PhysX mesh), otherwise generates a mesh from the heightfield.

    Args:
        field: The normalised terrain description.
        device: Warp device string.

    Returns:
        (mesh, mesh_ids) where ``mesh`` is a :class:`warp.Mesh` (keep a reference
        alive so it is not garbage-collected) and ``mesh_ids`` is a
        ``wp.array([mesh.id], dtype=wp.uint64)`` ready to pass to the kernel.
    """
    import warp as wp  # lazy: only this helper needs Warp

    if field.vertices is not None and field.triangles is not None:
        verts = np.asarray(field.vertices, dtype=np.float32).reshape(-1, 3)
        tris = np.asarray(field.triangles, dtype=np.int32).reshape(-1)
    else:
        verts, tris = heightfield_to_trimesh(
            field.heightfield_m, field.horizontal_scale, (field.x_offset, field.y_offset)
        )
        verts = verts.reshape(-1, 3)
        tris = tris.reshape(-1)

    points = wp.array(verts, dtype=wp.vec3, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)
    mesh = wp.Mesh(points=points, indices=indices)
    mesh_ids = wp.array([mesh.id], dtype=wp.uint64, device=device)
    return mesh, mesh_ids
