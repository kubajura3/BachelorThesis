"""Warp compute kernels for the perception module.

Currently exposes a single depth-range ray-casting kernel
(:func:`perception.warp_kernels.cam_kernel.draw_depth_range`). Kept in its own
subpackage so additional kernels (e.g. a lidar/point-cloud variant) can be added
later without cluttering the camera wrapper.
"""

from .cam_kernel import draw_depth_range

__all__ = ["draw_depth_range"]
