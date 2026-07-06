"""Depth-image post-processing (clip / resize / normalise / noise).

A small, pure-torch stage modelled on MGDP's ``make_image_processor`` (section
4.1.2 of the paper): clip the raw ray-cast depth to the sensor range, resize to
the policy input resolution, optionally normalise to [0, 1], and produce a second
*noisy* copy with sim-to-real style perturbations. The *clean* copy is never
noised, so both a privileged and a realistic view are available at once.

The ops (clamp / interpolate) are differentiable, so this stage does not close
off the future differentiable-depth seam; noise is applied only to the noisy
branch and only when configured.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from .config import PerceptionCfg


class DepthPreprocessor:
    """Turn raw ray-cast depth into (clean, noisy) policy-ready depth tensors.

    Args:
        cfg: Perception configuration (range, output size, normalisation, noise).
    """

    def __init__(self, cfg: PerceptionCfg) -> None:
        """Store the perception config (range, output size, normalisation, noise)."""
        self.cfg = cfg

    def __call__(self, depth_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of raw depth images.

        Args:
            depth_raw: (B, num_sensors, H, W) depth in metres, as returned by the
                Warp camera (``num_sensors`` is treated as the channel axis).

        Returns:
            (clean, noisy), each (B, num_sensors, out_h, out_w). ``clean`` is
            clipped/resized/normalised; ``noisy`` is ``clean`` plus multiplicative
            Gaussian noise and per-pixel dropout when those are enabled in the
            config (otherwise it is identical to ``clean``).
        """
        cfg = self.cfg

        x = depth_raw.clamp(0.0, cfg.max_range)
        x = F.interpolate(x, size=(cfg.out_h, cfg.out_w), mode="bilinear", align_corners=False)
        if cfg.normalize_depth:
            x = x / cfg.max_range

        clean = x
        noisy = clean
        if cfg.noise_gaussian is not None or cfg.noise_dropout is not None:
            noisy = clean.clone()
            if cfg.noise_gaussian is not None:
                # Multiplicative noise: larger absolute error at larger depths.
                noisy = noisy + torch.randn_like(noisy) * (cfg.noise_gaussian * noisy)
            if cfg.noise_dropout is not None:
                mask = (torch.rand_like(noisy) > cfg.noise_dropout).to(noisy.dtype)
                noisy = noisy * mask

        return clean, noisy
