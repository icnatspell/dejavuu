"""Build a benchmarking corpus for REST / SAM-Decoding's static datastore.

    uv run python -m dejavuu.tools.build_specbench_corpus --out data/specbench_corpus.txt

Derived entirely from Spec-Bench itself (the same cached question.jsonl the harness
downloads), so it is regenerable and needs no hand-written text. Two halves:

  relevant -- the long source passages from summarization + rag. An extractive model
              re-emits their n-grams, so these are where retrieval drafting actually
              pays off. Taken HELD-OUT (skip the first --holdout per category) so the
              corpus doesn't contain the exact prompts a default eval run scores --
              same-domain overlap is the realistic win, exact-prompt match is leakage.
  noise    -- the diverse MT-bench prompts (writing/roleplay/coding/stem/...). Different
              vocabulary; stands in for the unrelated junk any real datastore carries.

One document per line, shuffled (fixed seed) so relevant/noise interleave.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

from loguru import logger

from dejavuu.eval.specbench import CACHE, SPEC_BENCH_URL

RELEVANT = ("summarization", "rag")  # long-context, output overlaps the passage
NOISE = ("writing", "roleplay", "reasoning", "coding", "stem", "humanities")


def _clean(turn: str) -> str:
    """Strip the task prefix so the doc is the bare passage, not the instruction."""
    for prefix in ("Summarize:", "Translate German to English:"):
        if turn.startswith(prefix):
            return turn[len(prefix) :].strip()
    return turn.strip()


def build(holdout: int) -> list[str]:
    path = CACHE / "spec_bench.jsonl"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        logger.info("downloading Spec-Bench -> {}", path)
        urllib.request.urlretrieve(SPEC_BENCH_URL, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    seen: Counter[str] = Counter()
    docs: list[str] = []
    for rec in rows:
        cat = rec["category"]
        if cat in RELEVANT:
            seen[cat] += 1
            if seen[cat] <= holdout:
                continue  # leave the first K per category for the eval to score on
            docs.append(_clean(rec["turns"][0]))
        elif cat in NOISE:
            docs.append(_clean(rec["turns"][0]))

    n_rel = sum(v - holdout for v in seen.values() if v > holdout)
    logger.info("{} docs ({} relevant held-out, {} noise)", len(docs), n_rel, len(docs) - n_rel)
    random.Random(0).shuffle(docs)
    return docs


def main() -> None:
    p = argparse.ArgumentParser("dejavuu.tools.build_specbench_corpus")
    p.add_argument("--out", type=Path, default=Path("data/specbench_corpus.txt"))
    p.add_argument(
        "--holdout",
        type=int,
        default=20,
        help="relevant prompts per category to withhold (must cover --n)",
    )
    args = p.parse_args()

    docs = build(args.holdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newlines inside a passage would split it into bogus docs -> one line each
    args.out.write_text("\n".join(d.replace("\n", " ") for d in docs) + "\n")
    logger.info("wrote {} -> {}", len(docs), args.out)


def _demo() -> None:
    docs = build(holdout=20)
    assert docs, "corpus must be non-empty"
    assert all("\n" not in d for d in docs), "docs must be single-line"
    assert len(set(docs)) > len(docs) * 0.9, "should be mostly unique passages"
    logger.info("demo ok: {} docs", len(docs))


if __name__ == "__main__":
    main()
