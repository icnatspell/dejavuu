"""Spec-Bench text harness: baseline vs methods on Gemma (plan 8).

    uv run python -m dejavuu.eval.specbench --methods baseline,pld,token_recycling \
        --workload repetitive --n 20 --max-new 128

Greedy is lossless by construction, so the harness also asserts every method is
token-identical to the baseline -- a correctness gate, not just a speed table.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from dejavuu.core import Sampler, generate
from dejavuu.decoders.text import Model, download
from dejavuu.drafters import Cacheback
from dejavuu.eval.harness import (
    Agg,
    benchmark_metadata,
    load_datastore,
    make_drafter,
    render_table,
    write_run_manifest,
)

SPEC_BENCH_URL = (
    "https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl"
)
CACHE = Path.home() / ".cache" / "dejavuu"


def resolve_model_root(model_path: Path | None, variant: str) -> Path:
    """Return an explicit built-decoder directory or the bundled model snapshot.

    `variant` remains separate because one decoder directory contains fp32, int8,
    and q4 graphs. This keeps every quantized run comparable to its own baseline.
    """
    if model_path is not None:
        return model_path
    if variant == "fp32":
        raise ValueError("--variant fp32 requires --model-path to a built decoder directory")
    return download(variant)


# Group Spec-Bench's native categories by topic for reporting. The 8 MT-bench
# subcategories (the genuinely multi-turn rows) collapse into one topic; note
# `math` (MT-bench) != `math_reasoning` (GSM8K).
SPEC_TOPIC = {
    "summarization": "summarization",
    "translation": "translation",
    "qa": "question answering",
    "math_reasoning": "mathematical reasoning",
    "rag": "retrieval-augmented generation",
} | dict.fromkeys(
    ("writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"),
    "multi-turn conversation",
)

# repetitive = output overlaps a long context (where retrieval drafting pays off);
# diverse = open-ended generation (near-zero speedup expected, plan 7).
WORKLOADS = {
    "repetitive": {"summarization", "rag", "qa", "translation"},
    "diverse": {"writing", "roleplay", "reasoning", "stem", "humanities"},
    "all": set(SPEC_TOPIC),  # every native category
}


def load_specbench(workload: str, n: int, per_category: int = 0) -> list[tuple[str, str]]:
    """Return [(topic, prompt), ...], first turn only. `per_category>0` takes K from
    every *topic* (capped at availability); else first `n` of `workload`. Bucketed by
    SPEC_TOPIC -- the MT-bench subcategories collapse into multi-turn conversation."""
    path = CACHE / "spec_bench.jsonl"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        logger.info("downloading Spec-Bench -> {}", path)
        urllib.request.urlretrieve(SPEC_BENCH_URL, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if per_category:
        seen: Counter[str] = Counter()
        out = []
        for rec in rows:
            topic = SPEC_TOPIC.get(rec["category"], rec["category"])
            if seen[topic] < per_category:
                out.append((topic, rec["turns"][0]))
                seen[topic] += 1
        return out
    cats = WORKLOADS.get(workload, {workload})
    return [
        (SPEC_TOPIC.get(rec["category"], rec["category"]), rec["turns"][0])
        for rec in rows
        if rec["category"] in cats
    ][:n]


def main() -> None:
    p = argparse.ArgumentParser("dejavuu.eval.specbench")
    p.add_argument("--methods", default="baseline,pld,token_recycling")
    p.add_argument("--workload", default="repetitive")
    p.add_argument("--model-path", type=Path, default=None, help="built decoder directory")
    p.add_argument("--variant", choices=["fp32", "q4", "int8"], default="q4")
    p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--n", type=int, default=20)
    p.add_argument(
        "--per-category",
        type=int,
        default=0,
        help="K per topic (overrides --n/--workload)",
    )
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=8)
    p.add_argument(
        "--tree",
        action="store_true",
        help="tree verify; needs a tree-capable decoder, else falls back to chain",
    )
    p.add_argument("--width", type=int, default=2, help="max children/node in tree mode")
    p.add_argument(
        "--datastore",
        type=Path,
        default=None,
        help="corpus file (one doc per line) seeding REST / SAM-Decoding's static store",
    )
    p.add_argument(
        "--cacheback-frozen",
        type=Path,
        default=None,
        help="versioned Cacheback table; loaded once before the benchmark (cacheback only)",
    )
    p.add_argument(
        "--reset-drafter-per-prompt",
        action="store_true",
        help="construct each drafter before each prompt; use to measure cold caches",
    )
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument(
        "--threads", type=int, default=0, help="ORT intra-op threads per session (0 = ORT default)"
    )
    args = p.parse_args()
    sampler = Sampler(args.temperature, args.top_p, args.seed) if args.temperature > 0 else None

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logger.add(args.log, mode="a", format="{time:HH:mm:ss} {level} {message}")

    methods = args.methods.split(",")
    if "baseline" in methods:  # must run first: every method's exactness/speedup is vs it
        methods = ["baseline", *(m for m in methods if m != "baseline")]
    model = Model(
        resolve_model_root(args.model_path, args.variant),
        args.variant,
        args.provider,
        threads=args.threads,
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.root)
    prompts = load_specbench(args.workload, args.n, args.per_category)
    datastore = load_datastore(args.datastore, tok) if args.datastore else None
    logger.info(
        "{} prompts | budget={} max_new={}{}{} | methods={}",
        len(prompts),
        args.budget,
        args.max_new,
        f" tree(width={args.width})" if args.tree else "",
        f" datastore={len(datastore)}docs" if datastore else "",
        methods,
    )

    # One drafter instance per method for the whole run: stateful drafters (REST,
    # SuffixDecoding, Token Recycling) accumulate history across prompts.
    drafters = {
        m: (
            Cacheback.from_frozen(args.cacheback_frozen)
            if m == "cacheback" and args.cacheback_frozen
            else make_drafter(m, datastore)
        )
        for m in methods
    }
    aggs: dict[str, dict[str, Agg]] = {}  # topic -> method -> Agg
    baseline_out: dict[int, list[int]] = {}  # keyed by prompt index for the exactness gate
    baseline_tps: dict[int, float] = {}  # per-prompt baseline tps for the speedup mean
    for idx, (cat, prompt) in enumerate(tqdm(prompts, desc="prompts", unit="prompt")):
        cat_aggs = aggs.setdefault(cat, {m: Agg() for m in methods})
        ids = tok(prompt)["input_ids"]
        results: dict[str, tuple] = {}
        for m in methods:
            # Construction happens before timing: frozen-table loading is online-once,
            # not a decode-loop throughput gain or cost.
            drafter = (
                Cacheback.from_frozen(args.cacheback_frozen)
                if args.reset_drafter_per_prompt and m == "cacheback" and args.cacheback_frozen
                else make_drafter(m, datastore)
                if args.reset_drafter_per_prompt
                else drafters[m]
            )
            t = time.time()
            r = generate(
                model,
                ids,
                args.max_new,
                drafter,
                args.budget,
                tok.eos_token_id,
                tree=args.tree,
                width=args.width,
                sampler=sampler,
            )
            dt = time.time() - t
            cat_aggs[m].add(r, dt)
            ddt = dt - r.prefill_s  # decode-only time (prefill excluded)
            results[m] = (r, ddt)
            if m == "baseline":
                baseline_out[idx] = r.tokens
                baseline_tps[idx] = len(r.tokens) / ddt if ddt else 0.0
            elif idx in baseline_out:
                cat_aggs[m].compare(r.tokens, baseline_out[idx])
                cat_aggs[m].speedups(ddt, len(r.tokens), baseline_tps[idx])
        logger.info(
            "[{}] | {}",
            cat,
            " | ".join(f"{m}: {len(r.tokens) / ddt:.1f}tok/s" for m, (r, ddt) in results.items()),
        )

    scope = f"{args.per_category}/topic" if args.per_category else args.workload
    render_table(
        f"Spec-Bench / {scope} / {args.variant} (n={len(prompts)})",
        methods,
        aggs,
        args.log,
        csv_path=args.csv,
    )
    if args.csv:
        write_run_manifest(
            args.csv,
            benchmark_metadata(
                dataset="specbench",
                model=f"{model.root}:{args.variant}",
                provider=args.provider,
                threads=args.threads,
                budget=args.budget,
                tree=args.tree,
                width=args.width,
                max_new=args.max_new,
            ),
        )


if __name__ == "__main__":
    main()
