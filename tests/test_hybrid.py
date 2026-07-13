"""Hybrid (retrieval + verifier-logit fallback) drafter behavior.

Registry-wide validity and bit-exactness (chain + tree) are covered for suffix_recycle
and pld_recycle by tests/test_conformance.py -- the verifier still owns acceptance, so a
fallback can never break correctness. These tests pin the composition logic: fall back
only when the base is empty, and feed every hook to both sub-drafters. Refs #19.
"""

from dejavuu.drafters.base import Drafter, DraftTree
from dejavuu.drafters.hybrid import Hybrid


class _Empty(Drafter):
    """Never finds a match -> proposes only the root (0 guesses)."""

    def propose(self, ctx, past_len, budget):
        return DraftTree.chain([ctx[-1]])


class _Fixed(Drafter):
    """Always proposes one fixed continuation token, and records every hook call."""

    def __init__(self, token):
        self.token = token
        self.calls: list[str] = []

    def propose(self, ctx, past_len, budget):
        return DraftTree.chain([ctx[-1], self.token])

    def reset(self, prompt_ids):
        self.calls.append("reset")

    def update(self, accepted):
        self.calls.append("update")

    def observe(self, input_tokens, logits):
        self.calls.append("observe")


def test_falls_back_to_logit_drafter_only_when_base_is_empty():
    # Base empty -> use the fallback's draft.
    h = Hybrid(_Empty(), _Fixed(42))
    assert h.propose([1, 2, 3], 0, 4).token_ids == [3, 42]

    # Base fires -> use the base's draft, fallback ignored.
    base = _Fixed(7)
    h2 = Hybrid(base, _Fixed(42))
    assert h2.propose([1, 2, 3], 0, 4).token_ids == [3, 7]


def test_every_hook_feeds_both_sub_drafters():
    base, fallback = _Fixed(7), _Fixed(42)
    h = Hybrid(base, fallback)
    h.reset([1, 2])
    h.update([3])
    h.observe([1], object())
    assert base.calls == ["reset", "update", "observe"]
    assert fallback.calls == ["reset", "update", "observe"]  # fallback stays warm too
