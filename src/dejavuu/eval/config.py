"""Validated, immutable specification for one comparable benchmark run."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dejavuu.drafters import require_method


class DatasetSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["specbench", "speedbench", "mmspec"] = "specbench"
    split: str = "test"
    revision: str | None = None
    protocol: Literal["conversation", "first-turn-workload", "official"] = "conversation"
    workload: str = "all"
    n: int = Field(20, ge=1)
    per_category: int = Field(0, ge=0)
    image_size: int = Field(0, ge=0)


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str | None = None
    kind: Literal["auto", "text_onnx", "smolvlm_onnx"] = "auto"
    variant: str = "q4"
    provider: Literal["cpu", "cuda"] = "cpu"
    threads: int = Field(0, ge=0)
    allow_provider_fallback: bool = False
    allow_unverified_artifact: bool = False


class DecodeSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_new: int = Field(128, ge=1)
    budget: int = Field(8, ge=0)
    temperature: float = Field(0.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    seed: int = 0
    tree: bool = False
    width: int = Field(2, ge=1)
    # Loose (lossy) top-k acceptance: 1 = exact lossless (default); >1 accepts a drafted
    # token in the target's top-k, trading token identity for speed. Quality cost is
    # measured against the greedy baseline via the response scorers.
    accept_top_k: int = Field(1, ge=1)


class MeasurementSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    warmups: int = Field(1, ge=0)
    repetitions: int = Field(1, ge=1)
    order_seed: int = 0
    cache_scope: Literal["request", "run"] = "run"
    model_memory: Literal["cold", "warm"] = "warm"

    @model_validator(mode="after")
    def _cold_has_no_warmups(self) -> MeasurementSpec:
        if self.model_memory == "cold" and self.warmups:
            raise ValueError("cold model measurements require warmups=0")
        return self


class OutputSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = "results/run"
    responses: bool = True


class RunSpec(BaseModel):
    """One immutable comparison unit, validated before model or dataset loading."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: DatasetSpec = DatasetSpec()
    model: ModelSpec = ModelSpec()
    methods: tuple[str, ...] = ("baseline", "pld", "token_recycling")
    decode: DecodeSpec = DecodeSpec()
    measurement: MeasurementSpec = MeasurementSpec()
    output: OutputSpec = OutputSpec()

    @field_validator("methods")
    @classmethod
    def _validated_methods(cls, methods: tuple[str, ...]) -> tuple[str, ...]:
        ordered: list[str] = []
        for method in methods:
            require_method(method)
            if method not in ordered and method != "baseline":
                ordered.append(method)
        return ("baseline", *ordered)
