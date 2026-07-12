"""Lossless gate for the representation-aware drafters (PLD+, AdaPLD).

A tree-capable ToyModel that also emits per-token hidden states (its attention
output), so the engine's observe_hidden side channel is exercised. Greedy is exact,
so PLD+/AdaPLD must stay token-identical to the plain baseline in both chain and tree
mode -- the hidden-state rerank only picks *which* guess to copy, never the output.
"""

import numpy as np

from dejavuu.core import generate
from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.drafters import AdaPLD, PLDPlus

rng = np.random.default_rng(1)
D, V = 8, 20
E = rng.standard_normal((V, D))
P = rng.standard_normal((64, D)) * 0.5
Wo = rng.standard_normal((D, V))


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _causal(n, left=0):
    m = np.triu(np.full((n, n), -1e9), 1)
    return np.concatenate([np.zeros((n, left)), m], 1)


class HiddenToyModel(Verifier):
    """One causal attention layer; returns the attention output as hidden states."""

    @property
    def supports_tree(self) -> bool:
        return True

    def empty_kv(self) -> KVCache:
        z = np.zeros((1, 1, 0, D), np.float32)
        return [(z, z)]

    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        n = len(token_ids)
        if attn_bias is None:
            pos, bias = np.arange(past_len, past_len + n), _causal(n, past_len)
        else:
            pos, bias = position_ids[0], attn_bias[0, 0]
        x = E[np.asarray(token_ids)] + P[np.asarray(pos)]
        pk, pv = past[0][0][0, 0], past[0][1][0, 0]
        k_all, v_all = np.concatenate([pk, x]), np.concatenate([pv, x])
        out = _softmax(x @ k_all.T / np.sqrt(D) + bias) @ v_all
        return out @ Wo, [(k_all[None, None], v_all[None, None])], out


PROMPT = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]


def test_pld_plus_chain_is_lossless():
    model = HiddenToyModel()
    base = generate(model, PROMPT, 30)
    spec = generate(model, PROMPT, 30, PLDPlus(), budget=8)
    assert spec.tokens == base.tokens


def test_adapld_chain_and_tree_are_lossless():
    model = HiddenToyModel()
    base = generate(model, PROMPT, 30)
    chain = generate(model, PROMPT, 30, AdaPLD(), budget=8)
    tree = generate(model, PROMPT, 30, AdaPLD(), budget=8, tree=True, width=3)
    assert chain.tokens == base.tokens
    assert tree.tokens == base.tokens


def test_observe_hidden_populates_memory():
    """The side channel must actually fill the drafter's hidden memory."""
    model = HiddenToyModel()
    d = PLDPlus()
    generate(model, PROMPT, 10, d, budget=8)
    assert d.hid  # positions -> hidden rows got recorded


if __name__ == "__main__":
    test_pld_plus_chain_is_lossless()
    test_adapld_chain_and_tree_are_lossless()
    test_observe_hidden_populates_memory()
    print("ok")
