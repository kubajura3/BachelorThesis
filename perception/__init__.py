"""Terrain-perception module for the differentiable-SRBD Go2 environment.

This package gathers terrain information from the Isaac Gym simulation so the
locomotion policy can be extended from a *blind* observation to a
*terrain-aware* one. It produces two complementary signals every control step:

* a **forward depth image** (fast, forward-only ray-cast against the terrain
  mesh with NVIDIA Warp) -- the "vision" path used by camera-based approaches
  (Zhang et al. / DiffPhysDrone, or the MGDP depth encoder);
* a **local height map** sampled from the terrain heightfield with
  ``F.grid_sample`` -- natively differentiable w.r.t. the robot's planar
  position, usable as a privileged signal or a Rudin-style height scan.

Design principles (see ``PERCEPTION_IMPLEMENTATION.md`` for the full rationale):

* **Forward-only.** None of the intended downstream uses needs a differentiable
  renderer *now* -- gradients flow through the CNN and the SRBD dynamics, not
  through scene geometry. The depth path is therefore a plain forward pass; a
  seam is left to make it differentiable later without touching callers.
* **GPU-resident, zero-copy.** All buffers live on CUDA and are reused in place;
  torch<->Warp interop is a view, not a copy.
* **Self-contained.** Everything needed lives inside this repository; there is
  no import of, or runtime dependency on, any external sensor package.

Public API::

    from perception import PerceptionCollector, PerceptionCfg

The collector is the single entry point; ``collect(base_pos, base_quat_xyzw)``
returns a dict of tensors and is used identically by every downstream option.
It consumes the environment's own ``terrain.TerrainData`` (or ``None`` for flat
ground) -- perception does not define a second terrain type.
"""

from .config import PerceptionCfg  # lightweight (stdlib only)

__all__ = ["PerceptionCfg", "PerceptionCollector"]


def __getattr__(name):
    """Lazily expose the heavy collector (PEP 562).

    Importing ``perception`` (or ``perception.config``) must stay cheap so the
    environment's ``config.py`` can embed a ``PerceptionCfg`` without dragging in
    torch/warp on the blind-training path. ``PerceptionCollector`` -- which does
    need them -- is imported only when actually accessed.
    """
    if name == "PerceptionCollector":
        from .collector import PerceptionCollector

        return PerceptionCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
