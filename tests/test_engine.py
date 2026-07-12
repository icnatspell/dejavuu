"""Correctness guards. Loads the real 270m model once (cached after first run)."""

import numpy as np
import pytest

from dejavuu.core import generate
from dejavuu.decoders.text import Model, download
from dejavuu.drafters import PLD, REST, SAMDecoding, SuffixDecoding, TokenRecycling

# Whole module downloads + runs the real 270M model; opt-in via `pytest -m model`.
pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def model() -> Model:
    return Model(download("q4"), "q4")


# A repetitive prompt so PLD actually fires.
PROMPT = [2, 1235, 476, 9, 64, 49, 7, 235292, 1235, 476, 9, 64, 49, 7, 235292]


def test_kv_cache_matches_full_forward(model: Model):
    """Incremental KV decode must give the same logits as a single full pass --
    proves positions, 2D mask, and KV plumbing are correct."""
    full, _, _ = model.forward(PROMPT, model.empty_kv(), 0)
    _, past, _ = model.forward(PROMPT[:-1], model.empty_kv(), 0)
    inc, _, _ = model.forward(PROMPT[-1:], past, len(PROMPT) - 1)
    np.testing.assert_allclose(full[-1], inc[0], atol=1e-3)


# Drafters that fire on a single repetitive prompt (search the live context).
@pytest.mark.parametrize("drafter_cls", [PLD, TokenRecycling, SuffixDecoding, SAMDecoding])
def test_drafter_is_bit_exact_with_baseline(model: Model, drafter_cls):
    """Lossless gate: every drafter's greedy output is token-identical to baseline."""
    base = generate(model, PROMPT, 30)
    spec = generate(model, PROMPT, 30, drafter=drafter_cls(), budget=8)
    assert base.tokens == spec.tokens
    assert spec.steps < base.steps  # and it actually saved forward passes


def test_rest_is_bit_exact_and_drafts_from_datastore(model: Model):
    """REST ignores the live output, so seed its datastore with the baseline
    continuation; it must stay bit-exact and still save forward passes."""
    base = generate(model, PROMPT, 30)
    rest = REST(datastore=[PROMPT + base.tokens])
    spec = generate(model, PROMPT, 30, drafter=rest, budget=8)
    assert base.tokens == spec.tokens
    assert spec.steps < base.steps
