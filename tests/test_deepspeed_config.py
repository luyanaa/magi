"""DeepSpeed precision selection (fp16 vs bf16).

The two precisions differ in exactly one field, so they share a config and are
chosen by a parameter.  These tests pin that behaviour and, more importantly,
pin the refusal to use DeepSpeed's *native* fp16/bf16 blocks: those cast module
weights to half and disable autocast inside ``engine.forward()``, which breaks
this model's fp32 ``create_graph`` physics path.
"""

import json
from pathlib import Path

import pytest

from brain_moe_pinn.training.training_loop import (
    _PRECISION_DTYPES,
    apply_precision,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SHIPPED = [
    "ds_config_zero2.json",
    "ds_config_zero2_multinode.json",
    "ds_config_zero3_multinode.json",
]


def _load(name):
    return json.loads((CONFIG_DIR / name).read_text())


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_configs_use_torch_autocast(name):
    """Every shipped config must use torch_autocast, not a native block.

    Native fp16 casts weights to half and disables autocast inside
    engine.forward(); the resulting fp32-vs-half mismatch is what made the
    earlier DeepSpeed runs fail outright.
    """
    cfg = _load(name)
    assert cfg["torch_autocast"]["enabled"] is True
    assert cfg["torch_autocast"]["dtype"] in _PRECISION_DTYPES.values()
    assert not cfg.get("fp16", {}).get("enabled", False)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("precision,expected", sorted(_PRECISION_DTYPES.items()))
def test_precision_selects_dtype(name, precision, expected):
    out = apply_precision(_load(name), precision)
    assert out["torch_autocast"]["dtype"] == expected
    assert out["torch_autocast"]["enabled"] is True


def test_none_precision_preserves_config():
    """Omitting --precision must not change the config's own choice."""
    cfg = _load("ds_config_zero2.json")
    assert apply_precision(cfg, None) == cfg


def test_precision_does_not_mutate_input():
    cfg = _load("ds_config_zero2.json")
    before = json.dumps(cfg, sort_keys=True)
    apply_precision(cfg, "bf16")
    assert json.dumps(cfg, sort_keys=True) == before


def test_precision_drops_native_blocks():
    """A native block left beside torch_autocast would still cast weights."""
    cfg = dict(_load("ds_config_zero2.json"),
               fp16={"enabled": True}, bf16={"enabled": False})
    out = apply_precision(cfg, "bf16")
    assert "fp16" not in out
    assert "bf16" not in out


def test_native_only_config_is_rejected():
    """A config without torch_autocast must fail loudly, not silently."""
    with pytest.raises(ValueError, match="torch_autocast"):
        apply_precision({"fp16": {"enabled": True}}, "bf16")


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="unknown precision"):
        apply_precision(_load("ds_config_zero2.json"), "int8")


def test_precision_is_a_choice_of_exactly_two():
    assert sorted(_PRECISION_DTYPES) == ["bf16", "fp16"]
