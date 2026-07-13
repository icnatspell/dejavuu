"""Correctness guards. Loads the real 270m model once (cached after first run)."""

import os
from pathlib import Path

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


def test_built_qwen_decoder_is_bit_exact_under_chain_and_tree():
    """Opt-in smoke for the documented Qwen3 build → benchmark artifact path."""
    from transformers import AutoTokenizer

    root = Path(
        os.environ.get(
            "DEJAVUU_QWEN_DECODER",
            Path.home() / ".cache" / "dejavuu" / "Qwen-Qwen3-0.6B",
        )
    )
    if not (root / "onnx" / "model_q4.onnx").exists():
        pytest.skip("build Qwen3 with scripts/build_decoder.sh or set DEJAVUU_QWEN_DECODER")
    qwen = Model(root, "q4")
    prompt = AutoTokenizer.from_pretrained(root)("Repeat: red blue red blue")["input_ids"]
    baseline = generate(qwen, prompt, 16).tokens
    for tree in (False, True):
        output = generate(qwen, prompt, 16, drafter=PLD(), tree=tree).tokens
        assert output == baseline
