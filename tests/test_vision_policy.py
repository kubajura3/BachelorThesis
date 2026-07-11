"""
Tests for the stage-2 vision policy (policy.DepthEncoder / policy.VisionPolicy).

Covers the properties the end-to-end depth training depends on:

  1. shapes: (B,36) proprio + (B,1,12,16) depth -> (B,12) action
  2. end-to-end gradient: a scalar loss on the action back-propagates nonzero
     gradients into the depth encoder's first conv layer (the whole point of
     training the CNN through the SRBD rollout)
  3. depth sensitivity: perturbing the depth input changes the action at init
     (the encoder is neither dead nor ignored by the trunk)
  4. TorchScript: the act-only wrapper traces and reproduces the eager output
     (mirrors the export in train.py)

Self-contained: runs on CPU, no Isaac Gym / Warp / GPU required:

    python tests/test_vision_policy.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy import DepthEncoder, VisionPolicy

torch.manual_seed(0)

B, DIM_OBS, DIM_ACT = 5, 36, 12
IN_H, IN_W = 12, 16


def _inputs(requires_grad_depth: bool = False):
    s = torch.randn(B, DIM_OBS)
    depth = torch.rand(B, 1, IN_H, IN_W, requires_grad=requires_grad_depth)
    return s, depth


def test_shapes():
    model = VisionPolicy(dim_obs=DIM_OBS, dim_action=DIM_ACT)
    s, depth = _inputs()
    a, h = model(s, depth)
    assert a.shape == (B, DIM_ACT), a.shape
    assert h is None
    z = model.encoder(depth)
    assert z.shape == (B, 64), z.shape
    # Non-default resolution keeps the flatten arithmetic honest.
    enc = DepthEncoder(latent_dim=32, in_h=24, in_w=32)
    assert enc(torch.rand(B, 1, 24, 32)).shape == (B, 32)
    print("[1] shapes: OK")


def test_encoder_receives_gradients():
    model = VisionPolicy(dim_obs=DIM_OBS, dim_action=DIM_ACT)
    s, depth = _inputs()
    a, _ = model(s, depth)
    a.sum().backward()
    g = model.encoder.conv[0].weight.grad
    assert g is not None and g.abs().sum().item() > 0.0, "first conv got no gradient"
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
    print("[2] gradients reach the depth encoder through the action: OK")


def test_action_depends_on_depth():
    model = VisionPolicy(dim_obs=DIM_OBS, dim_action=DIM_ACT)
    s, depth = _inputs()
    a1, _ = model(s, depth)
    a2, _ = model(s, depth + 0.5)
    diff = (a1 - a2).abs().max().item()
    assert diff > 1e-9, f"action ignores the depth input (max diff {diff})"
    print(f"[3] action responds to depth (max diff {diff:.2e}): OK")


def test_torchscript_trace():
    class VisionActOnly(torch.nn.Module):
        """Mirrors the export wrapper in train.py."""
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, d):
            a, _ = self.m(x, d)
            return a

    model = VisionPolicy(dim_obs=DIM_OBS, dim_action=DIM_ACT).eval()
    wrapper = VisionActOnly(model)
    example = (torch.zeros(1, DIM_OBS), torch.zeros(1, 1, IN_H, IN_W))
    traced = torch.jit.trace(wrapper, example)
    s, depth = _inputs()
    with torch.no_grad():
        eager = wrapper(s, depth)
        scripted = traced(s, depth)
    assert torch.allclose(eager, scripted, atol=1e-6)
    print("[4] TorchScript trace reproduces eager output: OK")


if __name__ == "__main__":
    test_shapes()
    test_encoder_receives_gradients()
    test_action_depends_on_depth()
    test_torchscript_trace()
    print("\nAll vision-policy tests passed.")
