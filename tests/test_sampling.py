"""Model-free correctness for sampling verification.

The property that matters: under a sampler, spec-decode must emit the *same*
distribution as plain autoregressive sampling. The position-seeded coupling makes
that an exact identity -- spec-decode and the baseline draw the same uniform at each
absolute position -- so distribution preservation reduces to a token-for-token match,
no statistics needed. We also spot-check that the sampler's empirical frequencies
track the target probabilities, so the match isn't trivially "both always greedy".
"""

import numpy as np

from dejavuu.core import generate
from dejavuu.core.sampling import Sampler
from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.drafters import PLD, TokenRecycling

V = 16
LOGITS = np.random.default_rng(1).standard_normal((V, V))  # next-token logits by token


class ToyLM(Verifier):
    """Logits depend only on the last token, so a node's prediction is well defined
    from its own (accepted) token -- enough to exercise accept + sampler + coupling
    without any KV/position machinery."""

    def empty_kv(self) -> KVCache:
        return []

    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        return LOGITS[np.asarray(token_ids)], [], None


def test_sampling_spec_matches_baseline_token_for_token():
    model = ToyLM()
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    for seed in range(5):
        for temp, top_p in [(1.0, 1.0), (0.7, 1.0), (1.3, 0.9)]:
            s = Sampler(temp, top_p, seed)
            base = generate(model, prompt, 40, sampler=s)
            for drafter in (PLD(), TokenRecycling()):
                spec = generate(model, prompt, 40, drafter=drafter, budget=6, sampler=s)
                assert spec.tokens == base.tokens, (seed, temp, top_p, type(drafter))


def test_sampler_tracks_target_distribution():
    """Frequencies over positions (each an independent draw) should track softmax(p)."""
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05] + [1e-9] * (V - 4)))
    s = Sampler(temperature=1.0, seed=7)
    counts = np.bincount([s.token(logits, pos) for pos in range(8000)], minlength=V)
    freq = counts / counts.sum()
    np.testing.assert_allclose(freq[:4], [0.5, 0.3, 0.15, 0.05], atol=0.03)


def test_temperature_zero_is_greedy():
    s = Sampler(temperature=0.0)
    assert s.token(LOGITS[5], position=99) == int(LOGITS[5].argmax())


if __name__ == "__main__":
    test_sampling_spec_matches_baseline_token_for_token()
    test_sampler_tracks_target_distribution()
    test_temperature_zero_is_greedy()
    print("ok")
