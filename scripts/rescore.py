#!/usr/bin/env python
"""Re-score existing benchmark bundles by *meaning*, offline.

Loose (lossy) verification trades token identity for speed, so the shipped
``text_similarity`` (character alignment) mismarks it: an early divergence that stays
on-topic scores near 0 even when the meaning is intact. This walks each bundle's
``responses.jsonl``, pairs every method response with its own baseline (same case /
turn / repetition), and reports the mean *semantic* similarity next to the aligned one
so you can tell "reworded but correct" from real drift -- the number that decides whether
a loose method is worth keeping.

Semantic similarity is the cosine of static sentence embeddings (model2vec: numpy-only,
no torch, reproducible/offline once the model is cached). model2vec is a dev-only tool,
deliberately kept out of the package deps -- install it just to run this:

    uv pip install model2vec
    uv run python scripts/rescore.py results/fp32-k1 results/fp32-k3 ...
    uv run python scripts/rescore.py results/*        # glob is fine
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from dejavuu.eval.scorers import text_similarity

_SEMANTIC_MODEL = "minishlab/potion-base-8M"
_encoder = None


def semantic_similarity(prediction: str, reference: str) -> float:
    """Cosine of static-embedding vectors for two responses, clamped to [0, 1].

    Unlike ``text_similarity`` (character alignment), this credits a reworded but
    semantically equivalent response and penalises a factual *drift* -- the distinction a
    loose verifier lives or dies by. Empty-vs-empty is 1.0.
    """
    global _encoder
    if not prediction and not reference:
        return 1.0
    if _encoder is None:
        try:
            from model2vec import StaticModel  # pyrefly: ignore[missing-import]
        except ImportError as exc:
            raise SystemExit("rescore needs model2vec: `uv pip install model2vec`") from exc
        _encoder = StaticModel.from_pretrained(_SEMANTIC_MODEL)
    a, b = _encoder.encode([prediction or " ", reference or " "])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return max(0.0, float(a @ b) / denom) if denom else 0.0


def _key(r: dict) -> tuple:
    return (r["case_id"], r["turn"], r["repetition"])


def rescore_bundle(bundle: Path) -> list[dict]:
    """One row per method in the bundle: mean aligned vs semantic similarity."""
    responses = [json.loads(line) for line in (bundle / "responses.jsonl").open()]
    baseline = {_key(r): r["text"] for r in responses if r.get("method") == "baseline"}

    agg: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in responses:
        method = r.get("method")
        if method == "baseline":
            continue
        ref = baseline.get(_key(r))
        if ref is None:  # no baseline for this case (shouldn't happen in a valid bundle)
            continue
        agg[method].append((text_similarity(r["text"], ref), semantic_similarity(r["text"], ref)))

    return [
        {
            "bundle": bundle.name,
            "method": method,
            "n": len(pairs),
            "aligned": sum(p[0] for p in pairs) / len(pairs),
            "semantic": sum(p[1] for p in pairs) / len(pairs),
        }
        for method, pairs in agg.items()
    ]


def main(argv: list[str]) -> int:
    bundles = [Path(a) for a in argv if (Path(a) / "responses.jsonl").exists()]
    if not bundles:
        print("no bundles with responses.jsonl in:", argv, file=sys.stderr)
        return 1
    print(f"{'bundle':16s} {'method':10s} {'n':>3s} {'aligned':>8s} {'semantic':>9s}")
    for bundle in bundles:
        for row in rescore_bundle(bundle):
            print(
                f"{row['bundle']:16s} {row['method']:10s} {row['n']:3d} "
                f"{row['aligned']:8.3f} {row['semantic']:9.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
