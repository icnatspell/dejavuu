"""Validated configuration for the public API and CLI.

These pydantic models live at the boundary: the `DejaVu` API and the CLI construct one
per call, which validates and documents every knob (bounds, known method, valid
backend), then unpack the plain values into the engine. The engine and drafters take
ordinary arguments -- no pydantic in the per-token hot loop.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dejavuu.drafters import require_method


class GenerationConfig(BaseModel):
    """Decode parameters. `temperature=0` is greedy; a positive temperature draws from
    the target distribution (still lossless -- position-seeded speculative sampling)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = "token_recycling"
    max_new: int = Field(64, ge=1)
    budget: int = Field(8, ge=0, description="max draft guesses per step; 0 = plain autoregressive")
    temperature: float = Field(0.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    seed: int = 0
    tree: bool = False
    width: int = Field(2, ge=1, description="max children per node under tree verification")

    @field_validator("method")
    @classmethod
    def _known_method(cls, v: str) -> str:
        require_method(v)  # clear error listing valid names
        return v


class ModelConfig(BaseModel):
    """Backend + model-loading options. `variant`/`provider` apply to the ORT backend;
    `device`/`dtype`/`attn_implementation` to the HF backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["ort", "hf"] = "ort"
    variant: str = "q4"
    provider: str = "cpu"
    device: str | None = None
    dtype: str | None = None
    attn_implementation: str = "eager"

    @model_validator(mode="after")
    def _hf_needs_device(self) -> ModelConfig:
        if self.backend == "hf" and self.device is None:
            raise ValueError(
                "the hf backend needs an explicit device=, e.g. device='cuda' or 'cpu'"
            )
        return self
