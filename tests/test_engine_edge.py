"""Engine edge cases: EOS termination, the streaming callback, tree->chain fallback,
degenerate budgets/lengths. Model-free -- a toy verifier with a deterministic cycle.
"""

import numpy as np

from dejavuu.core.engine import generate, generate_seeded
from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.drafters import PLD, Drafter, DraftTree, LogitSpec, SuffixDecoding


class _Cyc(Verifier):
    """next = (t + 1) % V. Deterministic repetitive cycle; supports_tree stays False
    (base default) so tree=True must fall back to chain."""

    V = 6

    def empty_kv(self) -> KVCache:
        return []

    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        logits = np.full((len(token_ids), self.V), -9.0, np.float32)
        for i, t in enumerate(token_ids):
            logits[i, (t + 1) % self.V] = 9.0
        return logits, [], None


class _PerfectTwoTokenDraft(Drafter):
    """Always proposes the deterministic cycle's next two tokens."""

    def propose(self, ctx, past_len, budget):
        del past_len, budget
        root = ctx[-1]
        return DraftTree.chain([root, (root + 1) % 6, (root + 2) % 6])


def test_eos_stops_generation():
    m = _Cyc()
    # cycle from 0: 1,2,3,4,5,0,1,... ; eos=3 must stop right after emitting 3.
    res = generate(m, [0], max_new=50, eos=3)
    assert res.tokens == [1, 2, 3]
    assert res.tokens[-1] == 3


def test_on_emit_reports_every_token_and_accept_flag():
    m = _Cyc()
    seen: list[tuple[int, bool]] = []
    res = generate(
        m, [0], max_new=10, drafter=SuffixDecoding(), on_emit=lambda t, a: seen.append((t, a))
    )
    assert [t for t, _ in seen] == res.tokens  # callback fires once per emitted token
    assert all(isinstance(a, bool) for _, a in seen)


def test_conditional_acceptance_tracks_each_reached_draft_position():
    """CAR has one opportunity and one hit per reached draft depth, not per tree node."""
    res = generate(_Cyc(), [0], max_new=4, drafter=_PerfectTwoTokenDraft(), budget=2)

    assert res.conditional_attempts == [2, 2]
    assert res.conditional_accepted == [2, 2]


def test_tree_falls_back_to_chain_losslessly_when_unsupported():
    m = _Cyc()
    assert m.supports_tree is False
    base = generate(m, [0], 20).tokens
    # tree=True with a non-tree model + a drafter: warns once, runs the chain path, stays exact.
    out = generate(m, [0], 20, drafter=PLD(), tree=True).tokens
    assert out == base


def test_budget_zero_is_plain_autoregressive():
    m = _Cyc()
    base = generate(m, [0], 15).tokens
    out = generate(m, [0], 15, drafter=PLD(), budget=0).tokens  # no room for any guess
    assert out == base


def test_max_new_zero_emits_nothing():
    assert generate(_Cyc(), [0], max_new=0).tokens == []


def test_single_token_prompt_prefill():
    # len<=1 prompt: prefill returns an empty KV with committed=0, then decodes normally.
    assert generate(_Cyc(), [2], 4).tokens == [3, 4, 5, 0]


def test_seeded_root_matches_plain_greedy_decode():
    """A root selected from prefill logits is target-correct, so verifying the
    descendants must preserve ordinary greedy decoding."""
    m = _Cyc()
    baseline = generate(m, [0, 1, 2], 20)
    seeded = generate_seeded(m, [0, 1, 2], 20, PLD())
    assert seeded.tokens == baseline.tokens


def test_seeded_root_keeps_logitspec_greedy_output_lossless():
    m = _Cyc()
    baseline = generate(m, [0, 1, 2], 20)
    seeded = generate_seeded(m, [0, 1, 2], 20, LogitSpec())
    assert seeded.tokens == baseline.tokens
