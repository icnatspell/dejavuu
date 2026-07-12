"""The pydantic config models: bounds, known-method, and backend/device validation."""

import pytest
from pydantic import ValidationError

from dejavuu.config import GenerationConfig, ModelConfig


def test_generation_defaults_are_valid():
    cfg = GenerationConfig()
    assert cfg.method == "token_recycling"
    assert cfg.temperature == 0.0
    assert cfg.top_p == 1.0


@pytest.mark.parametrize(
    "kw",
    [
        {"max_new": 0},  # must be >= 1
        {"budget": -1},  # must be >= 0
        {"temperature": -0.5},  # must be >= 0
        {"top_p": 0.0},  # must be > 0
        {"top_p": 1.5},  # must be <= 1
        {"width": 0},  # must be >= 1
        {"method": "nope"},  # must be a registered method
        {"unexpected": 1},  # extra="forbid"
    ],
)
def test_generation_rejects_bad_values(kw):
    with pytest.raises(ValidationError):
        GenerationConfig(**kw)


def test_model_config_hf_requires_device():
    ModelConfig(backend="hf", device="cpu")  # ok
    with pytest.raises(ValidationError, match="device"):
        ModelConfig(backend="hf")  # missing device


def test_model_config_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        ModelConfig(backend="tensorrt")


def test_configs_are_frozen():
    cfg = GenerationConfig()
    with pytest.raises(ValidationError):
        cfg.budget = 4  # frozen
