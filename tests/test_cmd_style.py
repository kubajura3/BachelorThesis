"""Command-style resolution: which velocity-command distribution a config selects.

Runs without torch or isaacgym -- `resolve_cmd_style` / `heading_command_active` live in
config.py precisely so this is importable on a dev machine with no GPU stack.

Why this file exists: `terrain_type` used to select the command ranges as a side effect.
CAMPAIGN_FINDINGS.md 19.2 decoupled them behind `cmd_style` and verified the cross-product by
hand, leaving nothing behind; 19.9 then found that train.py had gone on resolving the same
question independently, so `blind_omni` drew Rudin yaw commands while the loss still measured
yaw against a fixed reset heading. The legacy table below is the regression guard that makes
that class of drift a test failure instead of a campaign post-mortem.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import heading_command_active, resolve_cmd_style


class FakeCfg:
    """Only the four fields the resolvers read, so this needs no EnvCfg (and no torch)."""

    def __init__(self, terrain_type="flat", rand_cmd=False, cmd_style=None, heading_command=False):
        self.terrain_type = terrain_type
        self.rand_cmd = rand_cmd
        self.cmd_style = cmd_style
        self.heading_command = heading_command


# (terrain_type, rand_cmd) -> style, for cmd_style=None. This is the ladder _sample_command
# contained inline before 19.2; every run on disk predating that commit was produced by it, so
# these six rows are what "reproduces bit-for-bit" means.
LEGACY = [
    ("flat",  False, "fixed"),
    ("flat",  True,  "rand"),
    ("rough", False, "fixed"),
    ("rough", True,  "rand"),
    ("rudin", False, "rudin"),
    ("rudin", True,  "rudin"),   # terrain wins over rand_cmd
]


@pytest.mark.parametrize("terrain_type,rand_cmd,expected", LEGACY)
def test_default_reproduces_legacy_ladder(terrain_type, rand_cmd, expected):
    cfg = FakeCfg(terrain_type=terrain_type, rand_cmd=rand_cmd)
    assert resolve_cmd_style(cfg) == expected


@pytest.mark.parametrize("style", ["rudin", "rand", "fixed"])
@pytest.mark.parametrize("terrain_type", ["flat", "rough", "rudin"])
@pytest.mark.parametrize("rand_cmd", [False, True])
def test_explicit_style_overrides_terrain(style, terrain_type, rand_cmd):
    """An explicit cmd_style pins the distribution whatever the terrain says.

    This is what makes the 19 isolation arms possible: flat ground with Rudin commands.
    """
    cfg = FakeCfg(terrain_type=terrain_type, rand_cmd=rand_cmd, cmd_style=style)
    assert resolve_cmd_style(cfg) == style


@pytest.mark.parametrize("terrain_type,cmd_style,flag,expected", [
    ("flat",  "rudin", True,  True),    # the blind_omni_heading arm
    ("rudin", None,    True,  True),    # heading command on the curriculum
    ("flat",  None,    True,  False),   # fixed forward -- no yaw command to redefine
    ("flat",  "rand",  True,  False),   # rand style uses yaw_min/yaw_max, not the heading
    ("flat",  "rudin", False, False),   # flag off -> held yaw rate, the pre-port behaviour
    ("rudin", None,    False, False),   # every campaign run on disk
])
def test_heading_command_needs_both_flag_and_rudin_style(terrain_type, cmd_style, flag, expected):
    cfg = FakeCfg(terrain_type=terrain_type, cmd_style=cmd_style, heading_command=flag)
    assert heading_command_active(cfg) is expected


def test_heading_command_defaults_off_for_every_legacy_config():
    """No run predating the heading port may silently acquire it."""
    for terrain_type, rand_cmd, _ in LEGACY:
        cfg = FakeCfg(terrain_type=terrain_type, rand_cmd=rand_cmd)
        assert heading_command_active(cfg) is False


def test_resolvers_tolerate_a_config_missing_the_new_fields():
    """An older pickled/duck-typed cfg must not crash the resolvers.

    Both read the new fields through getattr with a default, which is what lets train.py call
    them while loading a run folder written before the fields existed.
    """
    class Old:
        terrain_type = "rudin"
        rand_cmd = False

    assert resolve_cmd_style(Old()) == "rudin"
    assert heading_command_active(Old()) is False


def test_real_env_cfg_defaults_match_the_legacy_flat_run():
    """Guards the actual dataclass defaults, not just the FakeCfg stand-in."""
    from config import EnvCfg

    cfg = EnvCfg()
    assert cfg.cmd_style is None
    assert cfg.heading_command is False
    assert resolve_cmd_style(cfg) == "fixed"
    assert heading_command_active(cfg) is False
