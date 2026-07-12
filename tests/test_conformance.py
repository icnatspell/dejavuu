"""Registry-wide conformance: every drafter in `DRAFTERS` must (a) emit a
structurally valid `DraftTree` from both entry points and (b) stay bit-exact with
the plain autoregressive baseline under *both* chain and tree verification.

This is the standardization guard for drop-in drafters: a new entry in
`dejavuu.drafters.DRAFTERS` is covered here for free, so "usable in chain- and
tree-based verification" is enforced, not assumed. Model-free -- the toy verifier
below predicts next = (t+1) % V, a deterministic repetitive cycle that both makes
n-gram drafts fire and keeps the expected output trivially checkable.
"""

import numpy as np
import pytest

from dejavuu.core.engine import generate
from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.drafters import DRAFTERS, make_drafter
from dejavuu.drafters.base import DraftTree

PROMPT = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]  # repetitive -> live-context drafters fire


class _Toy(Verifier):
    """next = (t + 1) % V. Each row's logits depend only on that row's token, so the
    same graph is correct for chain and for tree verification (a node is predicted
    from its own token, never its siblings) -- hence supports_tree=True."""

    V = 5

    @property
    def supports_tree(self) -> bool:
        return True

    def empty_kv(self) -> KVCache:
        return []

    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        logits = np.full((len(token_ids), self.V), -9.0, np.float32)
        for i, t in enumerate(token_ids):
            logits[i, (t + 1) % self.V] = 9.0
        return logits, [], None


def assert_valid_tree(t: DraftTree, budget: int) -> None:
    assert len(t.token_ids) == len(t.parent)
    assert t.parent[0] == -1  # root is the last accepted real token
    assert all(0 <= p < i for i, p in enumerate(t.parent) if i)  # topological, single root
    assert len(t.token_ids) - 1 <= budget  # guesses respect the budget


NAMES = list(DRAFTERS)  # every registered drafter (baseline is not one)


@pytest.fixture(scope="module")
def baseline() -> list[int]:
    return generate(_Toy(), PROMPT, 30).tokens


@pytest.mark.parametrize("name", NAMES)
def test_drafter_emits_valid_tree(name: str):
    d = make_drafter(name)
    assert d is not None
    d.reset(PROMPT)
    assert_valid_tree(d.propose(list(PROMPT), len(PROMPT), budget=8), budget=8)
    assert_valid_tree(d.propose_tree(list(PROMPT), len(PROMPT), budget=8, width=2), budget=8)


@pytest.mark.parametrize("use_tree", [False, True], ids=["chain", "tree"])
@pytest.mark.parametrize("name", NAMES)
def test_drafter_is_lossless(name: str, use_tree: bool, baseline: list[int]):
    """Every drafter must reproduce the baseline exactly under chain AND tree
    verification. Datastore-backed drafters (rest, sam, asam) get seeded with the
    baseline continuation so their drafts actually fire."""
    m = _Toy()
    store = [PROMPT + baseline]
    d = make_drafter(name, datastore=store)
    out = generate(m, PROMPT, 30, drafter=d, budget=8, tree=use_tree).tokens
    assert out == baseline
