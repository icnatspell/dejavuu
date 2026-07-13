"""Loose (lossy) top-k acceptance -- opt-in, measured, never the default.

`accept_top_k == 1` is the exact lossless greedy path (guarded bit-exact for every
drafter by tests/test_conformance.py). These tests cover the added behavior: a drafted
token in the target's top-k is accepted even when it is not the argmax, which trades
token identity for a longer accepted length. Refs #15.
"""

import numpy as np

from dejavuu.core.engine import generate
from dejavuu.core.tree import DraftTree, pick_child
from dejavuu.core.verifier import Verifier
from dejavuu.drafters import make_drafter


def test_pick_child_greedy_corrects_a_non_argmax_draft():
    # Drafted child is token 3; the model's argmax is token 5. Lossless (k=1) rejects
    # the draft and emits the argmax correction, descending into no child.
    tree = DraftTree.chain([0, 3])
    logits = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 3.0])  # argmax=5, token 3 is second
    assert pick_child(tree, 0, logits, 0, None, 1) == (5, None)


def test_pick_child_loose_accepts_a_top_k_draft():
    # Same setup: token 3 is the model's second choice. At k=2 it is in the top-k, so
    # the loose rule accepts the drafted token and descends into that child.
    tree = DraftTree.chain([0, 3])
    logits = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 3.0])
    assert pick_child(tree, 0, logits, 0, None, 2) == (3, 1)


def test_pick_child_loose_prefers_the_most_probable_qualifying_child():
    # Two drafted branches (tokens 3 and 5) both in the top-3; the more probable (5) wins.
    tree = DraftTree.branches(0, [[3], [5]], budget=4)
    logits = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 3.0])  # 5 > 3
    tok, child = pick_child(tree, 0, logits, 0, None, 3)
    assert tok == 5
    assert tree.token_ids[child] == 5


class _Cycle(Verifier):
    """next = (t + 1) % V, with (t + 2) % V a strong runner-up -- so a drafted
    runner-up token is rejected greedily but accepted under loose top-2."""

    V = 6

    def empty_kv(self):
        return []

    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        logits = np.full((len(token_ids), self.V), -9.0, np.float32)
        for i, t in enumerate(token_ids):
            logits[i, (t + 1) % self.V] = 9.0
            logits[i, (t + 2) % self.V] = 5.0
        return logits, [], None


def test_loose_acceptance_never_accepts_fewer_than_lossless():
    prompt = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]  # repetitive -> pld fires
    drafter = lambda: make_drafter("pld")  # noqa: E731
    lossless = generate(_Cycle(), prompt, 20, drafter(), budget=4, accept_top_k=1)
    loose = generate(_Cycle(), prompt, 20, drafter(), budget=4, accept_top_k=3)
    assert loose.accepted >= lossless.accepted
