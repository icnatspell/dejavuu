"""Model-agnostic speculative-decoding core: the Verifier contract + the engine
(generate loop, tree verification, sampling)."""

from dejavuu.core.engine import GenResult, generate, generate_seeded
from dejavuu.core.sampling import Sampler, pick
from dejavuu.core.verifier import KVCache, Verifier, trim_kv

__all__ = [
    "GenResult",
    "KVCache",
    "Sampler",
    "Verifier",
    "generate",
    "generate_seeded",
    "pick",
    "trim_kv",
]
