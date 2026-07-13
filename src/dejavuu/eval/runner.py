"""Shared single-request benchmark runner.

Dataset and model adapters meet here: callers provide normalized case metadata plus a
function that prepares token ids for one method invocation. The runner owns all
cross-method invariants and debugging artifacts.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from tqdm import tqdm

from dejavuu.core import Sampler, generate
from dejavuu.core.engine import GenResult
from dejavuu.eval.harness import Agg, first_divergence


@dataclass(frozen=True)
class RunCase:
    case_id: str
    category: str
    metadata: dict[str, object]


def run_cases(
    cases: list[RunCase],
    methods: list[str],
    model,
    prepare: Callable[[RunCase, str], list[int]],
    make_drafter: Callable[[str, RunCase], object],
    decode: Callable[[list[int]], str],
    *,
    max_new: int,
    budget: int,
    eos: int | None,
    tree: bool,
    width: int,
    sampler: Sampler | None = None,
) -> tuple[dict[str, dict[str, Agg]], list[dict[str, object]], list[dict[str, object]]]:
    """Run baseline first for every case and return aggregates, records, failures."""
    aggs: dict[str, dict[str, Agg]] = {}
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for case in tqdm(cases, desc="prompts", unit="prompt"):
        category_aggs = aggs.setdefault(case.category, {method: Agg() for method in methods})
        baseline: list[int] | None = None
        baseline_tps = 0.0
        for method in methods:
            started = time.time()
            result: GenResult = generate(
                model,
                prepare(case, method),
                max_new,
                make_drafter(method, case),
                budget,
                eos,
                tree=tree,
                width=width,
                sampler=sampler,
            )
            elapsed = time.time() - started
            category_aggs[method].add(result, elapsed)
            decode_s = elapsed - result.prefill_s
            if method == "baseline":
                baseline = result.tokens
                baseline_tps = len(result.tokens) / decode_s if decode_s else 0.0
            else:
                assert baseline is not None
                category_aggs[method].compare(result.tokens, baseline)
                category_aggs[method].speedups(decode_s, len(result.tokens), baseline_tps)
            record: dict[str, object] = {
                "case_id": case.case_id,
                "category": case.category,
                "metadata": case.metadata,
                "method": method,
                "tokens": result.tokens,
                "text": decode(result.tokens),
                "baseline_tokens": baseline,
                "exact": result.tokens == baseline,
                "first_divergence": first_divergence(result.tokens, baseline or []),
                "drafted": result.drafted,
                "accepted": result.accepted,
                "conditional_attempts": result.conditional_attempts,
                "conditional_accepted": result.conditional_accepted,
            }
            records.append(record)
            if method != "baseline" and not record["exact"]:
                failures.append(record)
    return aggs, records, failures
