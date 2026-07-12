"""Sampling verification: generalise the greedy accept rule to a sampler.

The accept loop's only model decision is "what token does the target predict at this
node?" -- `argmax` for greedy. Swap that for a draw from the (temperature/top-p)
target distribution and, because these drafters are *deterministic* (a retrieved /
recycled token, no draft probabilities), the result is exactly Leviathan speculative
sampling with the draft as a point mass: accept the guess with prob p(guess), else the
fresh draw is the correction. So the emitted token at every position is distributed as
the target -- the draft only changes *how many* land per pass, never the output law.

The draw is seeded by **absolute sequence position** (`pick`): position P is sampled
with the same randomness whether it shows up as a rejected draft node in one pass or
the committed anchor in the next, and whether under spec-decode or the plain baseline.
That coupling is what makes "sampling spec-decode == sampling baseline, token for
token" an exact, testable identity (tests/test_sampling.py) instead of a distribution
match needing millions of trials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _nucleus(p: np.ndarray, top_p: float) -> np.ndarray:
    """Zero all but the smallest set of tokens whose cumulative prob reaches top_p
    (the crossing token is kept), then renormalise."""
    order = np.argsort(p)[::-1]
    keep = (np.cumsum(p[order]) - p[order]) < top_p  # exclusive cumsum below cutoff
    out = np.zeros_like(p)
    out[order[keep]] = p[order[keep]]
    return out / out.sum()


@dataclass(frozen=True)
class Sampler:
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0

    def token(self, logits: np.ndarray, position: int) -> int:
        if self.temperature <= 0:  # temp 0 == greedy
            return int(logits.argmax())
        p = _softmax(logits / self.temperature)
        if self.top_p < 1.0:
            p = _nucleus(p, self.top_p)
        # ponytail: fresh RNG per draw, seeded by (seed, position) so the same absolute
        # position always draws the same uniform -- the coupling. Cheap vs a forward pass.
        u = np.random.default_rng((self.seed, position)).random()
        return int(min(np.searchsorted(np.cumsum(p), u, side="right"), len(p) - 1))

    def gumbel_topk(self, logits: np.ndarray, position: int, k: int) -> list[int]:
        """Position-seeded Gumbel-Top-K without replacement for draft branches.

        This ranks candidate tokens only; the verifier's `token` draw still decides
        which token is emitted, preserving the target sampling distribution.
        """
        if k < 1:
            return []
        scaled = logits / max(self.temperature, 1e-8)
        rng = np.random.default_rng((self.seed, position, 1))
        u = np.clip(rng.random(len(scaled)), 1e-12, 1 - 1e-12)
        score = scaled - np.log(-np.log(u))
        k = min(k, len(score))
        part = np.argpartition(-score, k - 1)[:k]
        return [int(t) for t in part[np.argsort(-score[part])]]


def pick(logits: np.ndarray, position: int, sampler: Sampler | None) -> int:
    """The one model decision in the accept loop: greedy argmax, or a seeded draw."""
    return int(logits.argmax()) if sampler is None else sampler.token(logits, position)
