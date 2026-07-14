"""Hybrid (retrieval + verifier-logit fallback) drafter behavior.

Registry-wide validity and bit-exactness (chain + tree) are covered for suffix_recycle
and pld_recycle by tests/test_conformance.py -- the verifier still owns acceptance, so a
fallback can never break correctness. These tests pin the composition logic: fall back
only when the base is empty, and feed every hook to both sub-drafters. Refs #19.
"""

from dejavuu.drafters.base import Drafter, DraftTree
from dejavuu.drafters.hybrid import Hybrid, _graft


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


def test_graft_adds_fallback_branch_on_leftover_budget():
    # base chain: root 3 -> 7 (1 guess). Fallback chain: root 3 -> 9 -> 8. With budget 4,
    # graft hangs 9,8 off the root as a second branch (root now has children 7 and 9).
    base = DraftTree.chain([3, 7])
    fb = DraftTree.chain([3, 9, 8])
    merged = _graft(base, fb, budget=4)
    root_children = {
        merged.token_ids[c] for c in range(1, len(merged.token_ids)) if merged.parent[c] == 0
    }
    assert root_children == {7, 9}
    assert len(merged.token_ids) - 1 <= 4  # respects budget


def test_graft_skips_a_duplicate_first_branch():
    # Fallback's first token (7) already branches off the root -> don't duplicate it.
    base = DraftTree.chain([3, 7])
    fb = DraftTree.chain([3, 7, 8])
    merged = _graft(base, fb, budget=4)
    assert merged.token_ids == [3, 7]  # unchanged


def test_tail_mode_extends_the_deepest_path_not_a_sibling():
    # base copies root 3 -> 7 (1 guess, budget left). Fallback continues from the tail
    # token 7 with 8, 9. Tail mode appends them to the SAME path (a chain), not a branch.
    class _FromTail(Drafter):
        def propose(self, ctx, past_len, budget):
            return DraftTree.chain([ctx[-1], 8, 9]) if ctx[-1] == 7 else DraftTree.chain([ctx[-1]])

    class _CopyOne(Drafter):
        def propose(self, ctx, past_len, budget):
            return DraftTree.chain([ctx[-1], 7])

    h = Hybrid(_CopyOne(), _FromTail(), mode="tail")
    t = h.propose([1, 2, 3], 0, 4)
    assert t.token_ids == [3, 7, 8, 9]  # one path, extended
    assert t.parent == [-1, 0, 1, 2]  # a chain, no branching


def test_every_hook_feeds_both_sub_drafters():
    base, fallback = _Fixed(7), _Fixed(42)
    h = Hybrid(base, fallback)
    h.reset([1, 2])
    h.update([3])
    h.observe([1], object())
    assert base.calls == ["reset", "update", "observe"]
    assert fallback.calls == ["reset", "update", "observe"]  # fallback stays warm too
