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
from dejavuu.drafters import make_drafter as build_drafter
from dejavuu.eval.config import RunSpec
from dejavuu.eval.datasets import ConversationCase
from dejavuu.eval.harness import Agg, first_divergence
from dejavuu.eval.models import BenchmarkModel, ConversationHistory


@dataclass(frozen=True)
class RunCase:
    case_id: str
    category: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class Measurement:
    case_id: str
    category: str
    turn: int
    repetition: int
    method: str
    model_load_s: float
    prepare_s: float
    generation_s: float
    result: GenResult
    text: str
    exact: bool

    def as_record(self) -> dict[str, object]:
        """Serialize phase telemetry without duplicating generated response payloads."""
        decode_s = self.generation_s - self.result.prefill_s - self.result.draft_setup_s
        overhead_s = decode_s - self.result.draft_s - self.result.verify_s - self.result.learn_s
        return {
            "case_id": self.case_id,
            "category": self.category,
            "turn": self.turn,
            "repetition": self.repetition,
            "method": self.method,
            "model_load_s": self.model_load_s,
            "prepare_s": self.prepare_s,
            "generation_s": self.generation_s,
            "prefill_s": self.result.prefill_s,
            "draft_setup_s": self.result.draft_setup_s,
            "decode_s": decode_s,
            "draft_s": self.result.draft_s,
            "verify_s": self.result.verify_s,
            "learn_s": self.result.learn_s,
            "overhead_s": overhead_s,
            "tokens_emitted": len(self.result.tokens),
            "steps": self.result.steps,
            "drafted": self.result.drafted,
            "accepted": self.result.accepted,
            "root_proposals": self.result.root_proposals,
            "root_top1": self.result.root_top1,
            "root_top5": self.result.root_top5,
            "conditional_attempts": self.result.conditional_attempts,
            "conditional_accepted": self.result.conditional_accepted,
            "exact": self.exact,
        }


@dataclass
class RunResult:
    aggs: dict[str, dict[str, Agg]]
    measurements: list[Measurement]
    responses: list[dict[str, object]]
    failures: list[dict[str, object]]

    @property
    def valid(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class _Completed:
    result: GenResult
    generation_s: float
    text: str
    prepare_s: float
    model_load_s: float


class BenchmarkRunner:
    """Execute conversations with explicit timing, scheduling, and exactness policy."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock

    def run(
        self,
        spec: RunSpec,
        cases: list[ConversationCase],
        model: BenchmarkModel,
        datastore: list[list[int]] | None = None,
        drafter_builder: Callable[[str], object] | None = None,
        model_load_s: float = 0.0,
    ) -> RunResult:
        methods = list(spec.methods)
        aggs: dict[str, dict[str, Agg]] = {}
        measurements: list[Measurement] = []
        responses: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        construct = drafter_builder or (lambda method: build_drafter(method, datastore))
        run_drafters = {method: construct(method) for method in methods}
        warmed = False

        for repetition in range(spec.measurement.repetitions):
            for case_index, case in enumerate(cases):
                history: ConversationHistory = []
                for turn_index, turn in enumerate(case.turns):
                    invocation = None
                    shared_prepare_s = 0.0
                    if spec.measurement.model_memory == "warm":
                        started = self.clock()
                        invocation = model.prepare(case, turn, history)
                        shared_prepare_s = self.clock() - started
                        if (
                            spec.decode.tree
                            and len(methods) > 1
                            and not invocation.verifier.supports_tree
                        ):
                            raise ValueError(
                                "tree verification requested but model has no tree-capable decoder"
                            )
                        if not warmed:
                            for _ in range(spec.measurement.warmups):
                                model.warmup(invocation)
                            warmed = True
                            if spec.measurement.warmups:
                                # The first preparation initializes tokenizer/processor,
                                # embedding, vision, and decoder sessions. Measure the
                                # requested warm mode only after that online-once work.
                                started = self.clock()
                                invocation = model.prepare(case, turn, history)
                                shared_prepare_s = self.clock() - started

                    offset = (
                        case_index + repetition + turn_index + spec.measurement.order_seed
                    ) % len(methods)
                    schedule = [*methods[offset:], *methods[:offset]]
                    group: dict[str, _Completed] = {}
                    for method in schedule:
                        method_model = model
                        invocation_model_load_s = model_load_s
                        prepare_s = shared_prepare_s
                        method_invocation = invocation
                        if spec.measurement.model_memory == "cold":
                            started = self.clock()
                            method_model = model.cold_clone()
                            invocation_model_load_s = self.clock() - started
                            started = self.clock()
                            method_invocation = method_model.prepare(case, turn, history)
                            prepare_s = self.clock() - started
                        assert method_invocation is not None
                        if (
                            spec.decode.tree
                            and method != "baseline"
                            and not method_invocation.verifier.supports_tree
                        ):
                            raise ValueError(
                                "tree verification requested but model has no tree-capable decoder"
                            )
                        drafter = (
                            construct(method)
                            if spec.measurement.cache_scope == "request"
                            else run_drafters[method]
                        )
                        started = self.clock()
                        result = generate(
                            method_invocation.verifier,
                            method_invocation.prompt_ids,
                            spec.decode.max_new,
                            drafter,
                            spec.decode.budget,
                            method_invocation.eos_token_id,
                            tree=spec.decode.tree,
                            width=spec.decode.width,
                            sampler=(
                                Sampler(
                                    spec.decode.temperature,
                                    spec.decode.top_p,
                                    spec.decode.seed,
                                )
                                if spec.decode.temperature > 0
                                else None
                            ),
                        )
                        generation_s = self.clock() - started
                        group[method] = _Completed(
                            result,
                            generation_s,
                            method_model.decode(result.tokens),
                            prepare_s,
                            invocation_model_load_s,
                        )

                    baseline_run = group["baseline"]
                    baseline = baseline_run.result
                    baseline_s = baseline_run.generation_s
                    baseline_text = baseline_run.text
                    baseline_hot_s = baseline_s - baseline.prefill_s - baseline.draft_setup_s
                    baseline_tps = (
                        len(baseline.tokens) / baseline_hot_s if baseline_hot_s > 0 else 0.0
                    )
                    category_aggs = aggs.setdefault(
                        case.category, {method: Agg() for method in methods}
                    )
                    group_failed = False
                    sample_key = f"{case.case_id}:{turn_index}"
                    for method in methods:
                        completed = group[method]
                        result = completed.result
                        generation_s = completed.generation_s
                        text = completed.text
                        exact = result.tokens == baseline.tokens
                        category_aggs[method].add(
                            result,
                            generation_s,
                            completed.prepare_s,
                            completed.model_load_s,
                            sample_key,
                        )
                        if method != "baseline":
                            category_aggs[method].compare(result.tokens, baseline.tokens)
                            hot_s = generation_s - result.prefill_s - result.draft_setup_s
                            category_aggs[method].speedups(
                                hot_s, len(result.tokens), baseline_tps, sample_key
                            )
                        record: dict[str, object] = {
                            "case_id": case.case_id,
                            "category": case.category,
                            "turn": turn_index,
                            "repetition": repetition,
                            "method": method,
                            "tokens": result.tokens,
                            "text": text,
                            "baseline_tokens": baseline.tokens,
                            "exact": exact,
                            "first_divergence": first_divergence(result.tokens, baseline.tokens),
                            "model_load_s": completed.model_load_s,
                            "prepare_s": completed.prepare_s,
                            "prefill_s": result.prefill_s,
                            "draft_setup_s": result.draft_setup_s,
                            "decode_s": generation_s - result.prefill_s - result.draft_setup_s,
                        }
                        responses.append(record)
                        measurements.append(
                            Measurement(
                                case.case_id,
                                case.category,
                                turn_index,
                                repetition,
                                method,
                                completed.model_load_s,
                                completed.prepare_s,
                                generation_s,
                                result,
                                text,
                                exact,
                            )
                        )
                        if method != "baseline" and not exact:
                            failures.append(record)
                            group_failed = True
                    if group_failed:
                        break
                    if hasattr(model, "extend_history"):
                        history = model.extend_history(history, turn, baseline_text)
                    else:  # compatibility for lightweight third-party adapters
                        history.extend(
                            [
                                {"role": "user", "content": turn.text},
                                {"role": "assistant", "content": baseline_text},
                            ]
                        )
        for category_aggs in aggs.values():
            for agg in category_aggs.values():
                agg.finalize_repetitions()
        return RunResult(aggs, measurements, responses, failures)


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
            prompt_ids = prepare(case, method)
            drafter = make_drafter(method, case)
            started = time.perf_counter()
            result: GenResult = generate(
                model,
                prompt_ids,
                max_new,
                drafter,
                budget,
                eos,
                tree=tree,
                width=width,
                sampler=sampler,
            )
            elapsed = time.perf_counter() - started
            category_aggs[method].add(result, elapsed)
            decode_s = elapsed - result.prefill_s - result.draft_setup_s
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
