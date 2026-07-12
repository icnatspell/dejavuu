"""Model-free correctness for the tree primitives.

The property that matters: a single tree forward must give each node the same logits
as running its root->node path as an ordinary causal chain. We pin that down against
a tiny reference attention layer (positions + additive mask, no real model), so the
tree masking/positions/gather are proven independent of any ONNX re-export.
"""

import numpy as np
import pytest

from dejavuu.core import generate
from dejavuu.core import tree as T
from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.drafters import (
    REST,
    DraftTree,
    SAMDecoding,
    SuffixDecoding,
    TokenRecycling,
)

rng = np.random.default_rng(0)
D, V = 8, 20
E = rng.standard_normal((V, D))  # token embeddings
P = rng.standard_normal((64, D)) * 0.5  # positional table (so positions matter)
Wo = rng.standard_normal((D, V))  # output head


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _ref(tokens, position_ids, bias, past):
    """One causal attention layer. `bias` is additive [n, L+n]; past is (K,V) or None."""
    x = E[np.asarray(tokens)] + P[np.asarray(position_ids)]
    pk, pv = past if past is not None else (np.zeros((0, D)), np.zeros((0, D)))
    k_all = np.concatenate([pk, x], 0)
    v_all = np.concatenate([pv, x], 0)
    scores = x @ k_all.T / np.sqrt(D) + bias
    out = _softmax(scores) @ v_all
    return out @ Wo, (k_all, v_all)


def _causal(n, left=0):
    """Additive mask [n, left+n]: all `left` past columns visible, causal among n."""
    m = np.triu(np.full((n, n), T.NEG), 1)
    return np.concatenate([np.zeros((n, left)), m], 1)


def test_tree_forward_equals_per_path_chains():
    L = 5
    past_tokens = rng.integers(0, V, L)
    _, past = _ref(past_tokens, np.arange(L), _causal(L), None)

    #            r(0)
    #           /    \
    #         a(1)   b(2)
    #          |
    #         c(3)
    tree = DraftTree([3, 7, 11, 4], [-1, 0, 0, 1])
    pos = T.positions(tree.parent, L)[0]
    bias = T.mask(tree.parent, L)[0, 0]
    tlogits, _ = _ref(tree.token_ids, pos, bias, past)

    for node in range(len(tree.token_ids)):
        path = []  # root -> node
        j = node
        while j != -1:
            path.append(j)
            j = tree.parent[j]
        path.reverse()
        toks = [tree.token_ids[i] for i in path]
        positions = list(range(L, L + len(path)))  # contiguous, like a real chain
        clogits, _ = _ref(toks, positions, _causal(len(toks), L), past)
        np.testing.assert_allclose(clogits[-1], tlogits[node], atol=1e-6)


def test_accept_descends_tree():
    #   r(0) -> a(1), b(2);  a -> c(3);  b -> d(4)
    tree = DraftTree([3, 7, 11, 4, 9], [-1, 0, 0, 1, 2])
    logits = np.full((5, V), -1.0)
    logits[0, 7] = 5  # r predicts 7 == token of node 1 (a) -> accept down a
    logits[1, 4] = 5  # a predicts 4 == token of node 3 (c) -> accept down c
    logits[3, 2] = 5  # c predicts 2 -> no child matches -> bonus, stop
    emitted, n_acc, path = T.accept(tree, logits)
    assert emitted == [7, 4, 2]
    assert n_acc == 2
    assert path == [0, 1, 3]


def test_causal_bias_matches_reference():
    """OrtDecoder._causal_bias (chain step on a tree-capable graph) must equal the
    reference additive causal mask."""
    from dejavuu.decoders.ort import _causal_bias

    np.testing.assert_array_equal(_causal_bias(3, 4)[0, 0], _causal(3, 4))


def test_gather_kv_keeps_committed_plus_path():
    L, M, path = 4, 5, [0, 1, 3]
    rowid = np.arange(L + M, dtype=np.float32).reshape(1, 1, L + M, 1)
    kept = T.gather_kv([(rowid.copy(), rowid.copy())], L, path)[0][0][0, 0, :, 0]
    assert list(kept) == [0, 1, 2, 3, 4, 5, 7]  # committed 0..3 + (L+0, L+1, L+3)


class ToyModel(Verifier):
    """A tree-capable Verifier: one causal attention layer in numpy. Stands in for a
    re-exported decoder so the engine's tree forward/gather path runs model-free."""

    @property
    def supports_tree(self) -> bool:
        return True

    def empty_kv(self) -> KVCache:
        z = np.zeros((1, 1, 0, D), np.float32)
        return [(z, z)]

    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        n = len(token_ids)
        if attn_bias is None:  # chain: contiguous positions + causal mask
            pos, bias = np.arange(past_len, past_len + n), _causal(n, past_len)
        else:
            pos, bias = position_ids[0], attn_bias[0, 0]
        x = E[np.asarray(token_ids)] + P[np.asarray(pos)]
        pk, pv = past[0][0][0, 0], past[0][1][0, 0]
        k_all, v_all = np.concatenate([pk, x]), np.concatenate([pv, x])
        out = _softmax(x @ k_all.T / np.sqrt(D) + bias) @ v_all
        return out @ Wo, [(k_all[None, None], v_all[None, None])], None


def test_engine_tree_path_is_lossless():
    """Greedy is exact, so tree verification must match the plain baseline token for
    token -- this exercises positions + mask + accept + KV gather through the engine."""
    model = ToyModel()
    prompt = [1, 2, 3, 4, 5, 6, 7]
    base = generate(model, prompt, 25)
    chain = generate(model, prompt, 25, TokenRecycling(), budget=8)
    tree = generate(model, prompt, 25, TokenRecycling(), budget=8, tree=True, width=3)
    assert chain.tokens == base.tokens
    assert tree.tokens == base.tokens


@pytest.mark.parametrize(
    "make",
    [
        lambda base: SuffixDecoding(min_match=1),
        lambda base: SAMDecoding(min_match=1),
        lambda base: REST(datastore=[base], min_match=1),  # static store of the answer
    ],
    ids=["suffix_decoding", "sam_decoding", "rest"],
)
def test_retrieval_drafters_tree_path_is_lossless(make):
    """The retrieval drafters' propose_tree (branch into top-width continuations) must
    stay token-identical to the greedy baseline through the engine's tree path."""
    model = ToyModel()
    prompt = [1, 2, 3, 4, 5, 6, 7]
    base = generate(model, prompt, 25)
    drafter = make(prompt + base.tokens)
    tree = generate(model, prompt, 25, drafter, budget=8, tree=True, width=3)
    assert tree.tokens == base.tokens


if __name__ == "__main__":
    test_tree_forward_equals_per_path_chains()
    test_accept_descends_tree()
    test_gather_kv_keeps_committed_plus_path()
    test_engine_tree_path_is_lossless()
    for mk in (
        lambda b: SuffixDecoding(min_match=1),
        lambda b: SAMDecoding(min_match=1),
        lambda b: REST(datastore=[b], min_match=1),
    ):
        test_retrieval_drafters_tree_path_is_lossless(mk)
    print("ok")


def test_grow_prefers_high_probability_paths():
    from dejavuu.drafters.base import DraftTree

    # token 0's children: 1 (p.9) is a near-certain path, 2 (p.1) unlikely.
    # 1 continues to 3 (p.8); 2 to 4 (p.8). With budget 3, the high-prob path
    # 0->1->3 must be filled before spending budget on the unlikely 2.
    succ = {0: [(1, 0.9), (2, 0.1)], 1: [(3, 0.8)], 2: [(4, 0.8)]}
    tree = DraftTree.grow(0, lambda t: succ.get(t, []), budget=3)
    assert tree.token_ids[0] == 0
    # nodes added in priority order: 1 (.9), 3 (.72), then 2 (.1)
    assert tree.token_ids == [0, 1, 3, 2]
    assert tree.parent == [-1, 0, 1, 0]
    # path scores are cumulative products
    assert abs(tree.score[2] - 0.72) < 1e-9


def test_grow_respects_budget_and_avoids_cycles():
    from dejavuu.drafters.base import DraftTree

    succ = {0: [(0, 0.99), (1, 0.5)]}  # 0->0 is a self-cycle, must be skipped
    tree = DraftTree.grow(0, lambda t: succ.get(t, []), budget=5)
    assert tree.token_ids == [0, 1]  # only the non-cycle child, then nothing left
