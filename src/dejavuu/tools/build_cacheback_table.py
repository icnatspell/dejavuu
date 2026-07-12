"""Build a portable Cacheback frozen table from a newline-delimited text corpus.

The result stores token ids, not text, and is therefore tied to the tokenizer named
in its metadata. Building is offline work; loading the JSON is an online-once cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from dejavuu.drafters import Cacheback


def build_payload(
    documents: list[list[int]],
    *,
    tokenizer: str,
    leader_len: int = 4,
    follower_len: int = 4,
    leader_capacity: int = 16_384,
    follower_capacity: int = 4,
) -> dict[str, object]:
    """Build one deterministic LRU table without letting n-grams cross documents."""
    cache = Cacheback(leader_len, follower_len, leader_capacity, follower_capacity)
    for doc in documents:
        cache.reset([])
        cache.update(doc)
    return {
        "schema_version": 1,
        "method": "cacheback",
        "tokenizer": tokenizer,
        "leader_len": leader_len,
        "follower_len": follower_len,
        "entries": [
            {"leader": list(leader), "followers": [list(follower) for follower in followers]}
            for leader, followers in cache.cache.items()
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser("dejavuu.tools.build_cacheback_table")
    p.add_argument("--input", type=Path, required=True, help="one text document per line")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", required=True, help="HF tokenizer id or local path")
    p.add_argument("--leader-len", type=int, default=4)
    p.add_argument("--follower-len", type=int, default=4)
    p.add_argument("--leader-capacity", type=int, default=16_384)
    p.add_argument("--follower-capacity", type=int, default=4)
    args = p.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    docs = [tok(line)["input_ids"] for line in args.input.read_text().splitlines() if line.strip()]
    payload = build_payload(
        docs,
        tokenizer=args.tokenizer,
        leader_len=args.leader_len,
        follower_len=args.follower_len,
        leader_capacity=args.leader_capacity,
        follower_capacity=args.follower_capacity,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("wrote {} entries -> {}", len(payload["entries"]), args.out)


if __name__ == "__main__":
    main()
