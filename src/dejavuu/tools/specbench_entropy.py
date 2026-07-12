"""SuffixDecoding 'structuredness' diagnostic on Spec-Bench, per category.

    uv run --extra vlm python -m dejavuu.tools.specbench_entropy --per-category 20

For each Spec-Bench topic, generate SmolVLM responses (text-only, baseline greedy)
and measure the occurrence-weighted next-token entropy of a suffix index built from
those responses. Low entropy = self-repetitive outputs = retrieval drafting
(PLD/SuffixDecoding/SAM) pays off; high entropy = open-ended, near-baseline. It is a
cheap pre-check: it predicts *which* categories the retrieval drafters will speed up,
from ~100 outputs, without running the full method sweep. See docs/methods.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from dejavuu.core import generate
from dejavuu.decoders.vlm import VLM, download
from dejavuu.drafters.suffix_index import SuffixIndex
from dejavuu.eval.specbench import load_specbench


def main() -> None:
    p = argparse.ArgumentParser("dejavuu.tools.specbench_entropy")
    p.add_argument("--workload", default="all", help="ignored if --per-category set")
    p.add_argument("--n", type=int, default=100)
    p.add_argument(
        "--per-category",
        type=int,
        default=20,
        help="K prompts per topic (overrides --n/--workload)",
    )
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--order", type=int, default=8, help="suffix-index n-gram order")
    p.add_argument("--variant", choices=["q4", "int8"], default="q4")
    p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    from transformers import AutoProcessor

    root = download(args.variant)
    vlm = VLM(Path(root), args.variant, args.provider, threads=args.threads)
    processor = AutoProcessor.from_pretrained(root)
    eos = processor.tokenizer.eos_token_id

    prompts = load_specbench(args.workload, args.n, args.per_category)
    logger.info("{} prompts | max_new={} order={}", len(prompts), args.max_new, args.order)

    # One suffix index per category, built from that category's SmolVLM responses.
    idxs: dict[str, SuffixIndex] = {}
    counts: dict[str, int] = {}
    for cat, prompt in tqdm(prompts, desc="prompts", unit="prompt"):
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
        )
        ids = processor(text=text, return_tensors="np")["input_ids"][0].tolist()
        r = generate(vlm, ids, args.max_new, None, 8, eos)  # baseline greedy, no drafter
        idx = idxs.setdefault(cat, SuffixIndex(args.order))
        idx.extend(r.tokens)
        idx.append(idx.SEP)  # each response is its own document
        counts[cat] = counts.get(cat, 0) + 1

    rows = sorted(
        ((cat, counts[cat], len(idx.buf), idx.weighted_entropy()) for cat, idx in idxs.items()),
        key=lambda r: r[3],
    )
    table = Table(
        title=f"Spec-Bench structuredness (SmolVLM {args.variant}, "
        f"n={len(prompts)}, max_new={args.max_new})"
    )
    for col in ("category", "prompts", "tokens", "entropy (bits)"):
        table.add_column(col, justify="right", no_wrap=True)
    table.caption = "lower entropy = more predictable outputs = retrieval drafting pays off"
    for cat, n, toks, h in rows:
        table.add_row(cat, str(n), str(toks), f"{h:.3f}")
    Console().print(table)

    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category", "prompts", "tokens", "entropy_bits"])
            for cat, n, toks, h in rows:
                w.writerow([cat, n, toks, f"{h:.4f}"])
        logger.info("csv -> {}", args.csv)


if __name__ == "__main__":
    main()
