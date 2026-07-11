"""Neural network policies.

``Policy`` is the blind MLP (flat observation -> joint-angle offsets) used for
flat-ground and privileged height-scan training. ``VisionPolicy`` adds a small
``DepthEncoder`` CNN so the policy can consume the depth camera image next to
the proprioceptive observation; it is trained end-to-end through the SRBD
rollout (see ``train.py``).
"""

# NOTE: Isaac Gym must be imported before torch (hard requirement of isaacgym).
try:
    from isaacgym import gymapi
except Exception:
    pass

import torch
import torch.nn as nn


class Policy(nn.Module):
    """MLP policy: flat observation -> 256 x 256 trunk -> 12 joint-angle offsets.

    Args:
        dim_obs: Observation size (36 blind, 36+187 with the height scan).
        dim_action: Number of controlled joints (12 for the Go2).

    The last layer is initialised near zero (std 1e-2, zero bias) so training
    starts from approximately the default standing posture.
    """

    def __init__(self, dim_obs=36, dim_action=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_obs, 256), nn.LeakyReLU(0.05),
            nn.Linear(256, 256), nn.LeakyReLU(0.05),
            nn.Linear(256, dim_action),
        )
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, std=1e-2)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, h=None):
        """Map observations (B, dim_obs) to actions (B, dim_action).

        The second argument/return mirrors a recurrent interface (hidden state)
        but is unused by this feed-forward policy; ``None`` is always returned.
        """
        return self.net(s), None

    def reset(self):
        """Reset internal state between episodes (no-op for a feed-forward net)."""
        pass


class DepthEncoder(nn.Module):
    """Tiny CNN: (B, 1, in_h, in_w) depth image -> (B, latent_dim) feature vector.

    Deliberately DiffPhysDrone-scale (~30k params): the 12x16 input carries little
    detail, and a small encoder keeps the end-to-end SRBD backprop cheap.
    """

    def __init__(self, latent_dim=64, in_h=12, in_w=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.05),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.LeakyReLU(0.05),
        )
        # Two stride-2 convs with padding 1: each halves the size, rounding up.
        flat = 32 * ((in_h + 3) // 4) * ((in_w + 3) // 4)
        self.fc = nn.Sequential(nn.Linear(flat, latent_dim), nn.LeakyReLU(0.05))

    def forward(self, depth):
        """Encode depth images (B, 1, in_h, in_w) into latents (B, latent_dim)."""
        z = self.conv(depth)
        return self.fc(z.flatten(1))


class VisionPolicy(nn.Module):
    """Vision policy: proprioception + depth image -> 12-dim joint angle offsets.

    A DepthEncoder latent is concatenated to the proprio observation and fed to
    the same 256x256 trunk as Policy. Trained end-to-end through the SRBD
    rollout: gradients reach the encoder via the action -> SRBD -> loss chain
    (the depth image itself is a grad-free input from the Warp camera).
    """

    def __init__(self, dim_obs=36, dim_action=12, latent_dim=64, in_h=12, in_w=16):
        super().__init__()
        self.encoder = DepthEncoder(latent_dim=latent_dim, in_h=in_h, in_w=in_w)
        self.net = nn.Sequential(
            nn.Linear(dim_obs + latent_dim, 256), nn.LeakyReLU(0.05),
            nn.Linear(256, 256), nn.LeakyReLU(0.05),
            nn.Linear(256, dim_action),
        )
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, std=1e-2)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, depth, h=None):
        """Map proprio obs (B, dim_obs) + depth (B, 1, in_h, in_w) to actions.

        The hidden-state argument/return mirrors ``Policy.forward`` and is
        unused; ``None`` is always returned as the second element.
        """
        z = self.encoder(depth)                     # (B, latent_dim)
        return self.net(torch.cat([s, z], dim=-1)), None

    def reset(self):
        """Reset internal state between episodes (no-op for a feed-forward net)."""
        pass
