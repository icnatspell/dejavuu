"""Reference-based response-quality scorers for diagnostic benchmarking.

When a backend's token stream diverges from its autoregressive baseline (a normal
property of quantized graphs -- see AGENTS.md), token identity no longer answers the
question "is the output still good?". A scorer answers that on the *text*: it compares a
method's decoded response against the baseline response and returns a value in [0, 1],
where 1.0 means indistinguishable from the baseline.

A scorer is any ``(prediction, reference) -> float`` callable, kept deliberately
dataset-agnostic: the reference is always the baseline text for the same case, so the
runner never embeds per-dataset logic. Register new scorers in ``SCORERS`` to have every
run compute them automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from difflib import SequenceMatcher

Scorer = Callable[[str, str], float]


def text_similarity(prediction: str, reference: str) -> float:
    """Alignment-based similarity of two responses in [0, 1].

    Uses ``difflib.SequenceMatcher`` (stdlib), which aligns matching blocks rather than
    comparing position-by-position. That matters for speculative decoding: a method can
    diverge from the baseline for a few tokens and then realign, which a positional
    token-overlap metric penalises but an alignment metric credits. Empty against empty
    is defined as identical.
    """
    if not prediction and not reference:
        return 1.0
    return SequenceMatcher(None, prediction, reference).ratio()


# Every registered scorer runs on every response; keep entries cheap and dependency-free.
SCORERS: dict[str, Scorer] = {"text_similarity": text_similarity}


def score_response(prediction: str, reference: str) -> dict[str, float]:
    """Apply every registered scorer to one response against its baseline reference."""
    return {name: scorer(prediction, reference) for name, scorer in SCORERS.items()}
