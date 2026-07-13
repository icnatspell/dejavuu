"""Unified benchmark runner behavior through its public interface."""

import numpy as np
import pytest

from dejavuu.core.verifier import Verifier
from dejavuu.eval.config import DecodeSpec, MeasurementSpec, RunSpec
from dejavuu.eval.datasets import ConversationCase, Turn
from dejavuu.eval.models import ModelIdentity, PreparedInvocation
from dejavuu.eval.runner import BenchmarkRunner


class _Cycle(Verifier):
    def empty_kv(self):
        return []

    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        logits = np.full((len(token_ids), 5), -9.0, np.float32)
        for index, token in enumerate(token_ids):
            logits[index, (token + 1) % 5] = 9.0
        return logits, [], None


class _TextModel:
    identity = ModelIdentity("text_onnx", "toy", "cpu", ("CPUExecutionProvider",))

    def __init__(self):
        self.verifier = _Cycle()

    def prepare(self, case, turn, history):
        return PreparedInvocation(self.verifier, [0], None)

    def decode(self, token_ids):
        return " ".join(map(str, token_ids))

    def warmup(self, invocation):
        pass

    def cold_clone(self):
        return _TextModel()


class _LengthVariant(_Cycle):
    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        logits, present, hidden = super().forward(
            token_ids, past, past_len, position_ids, attn_bias
        )
        if past_len and len(token_ids) > 1:
            logits[:] = np.roll(logits, 1, axis=1)
        return logits, present, hidden


class _BrokenModel(_TextModel):
    def __init__(self):
        self.verifier = _LengthVariant()

    def prepare(self, case, turn, history):
        return PreparedInvocation(self.verifier, [0, 1, 2, 0, 1, 2], None)


def test_runner_excludes_preparation_and_balances_against_an_exact_baseline():
    times = iter([0.0, 5.0, 10.0, 12.0, 20.0, 23.0])
    runner = BenchmarkRunner(clock=lambda: next(times))
    spec = RunSpec(
        methods=("pld",),
        decode=DecodeSpec(max_new=4),
        measurement=MeasurementSpec(warmups=0),
    )
    case = ConversationCase("case-1", "reasoning", (Turn("Question"),))

    result = runner.run(spec, [case], _TextModel())

    assert not result.has_divergences
    assert [m.method for m in result.measurements] == ["baseline", "pld"]
    assert [m.prepare_s for m in result.measurements] == [5.0, 5.0]
    assert [m.generation_s for m in result.measurements] == [2.0, 3.0]
    assert all(m.exact for m in result.measurements)


def test_divergence_is_recorded_but_never_stops_or_invalidates_the_run():
    # Divergence between multi-token and incremental decoding is a backend numerical
    # property, not a failure. It is recorded as a diagnostic; the run stays usable and
    # keeps measuring every case (AGENTS.md: divergence never invalidates a benchmark).
    spec = RunSpec(
        methods=("pld",),
        decode=DecodeSpec(max_new=6),
        measurement=MeasurementSpec(warmups=0),
    )
    cases = [
        ConversationCase("case-1", "reasoning", (Turn("Question"),)),
        ConversationCase("case-2", "reasoning", (Turn("Question"),)),
    ]

    result = BenchmarkRunner().run(spec, cases, _BrokenModel())

    assert result.has_divergences
    diverged = [d for d in result.divergences if d["method"] == "pld"]
    assert diverged
    assert diverged[0]["exact"] is False
    assert diverged[0]["first_divergence"] is not None
    # Every case was measured despite the first case diverging.
    assert {m.case_id for m in result.measurements} == {"case-1", "case-2"}
    assert all(m.generation_s > 0 for m in result.measurements if m.method == "pld")


def test_cold_model_load_and_preparation_are_separate_from_decode():
    times = iter([0, 4, 10, 15, 20, 22, 30, 34, 40, 45, 50, 53])
    spec = RunSpec(
        methods=("pld",),
        decode=DecodeSpec(max_new=4),
        measurement=MeasurementSpec(warmups=0, model_memory="cold"),
    )
    case = ConversationCase("case-1", "reasoning", (Turn("Question"),))

    result = BenchmarkRunner(clock=lambda: next(times)).run(spec, [case], _TextModel())

    assert [m.model_load_s for m in result.measurements] == [4, 4]
    assert [m.prepare_s for m in result.measurements] == [5, 5]
    assert [m.generation_s for m in result.measurements] == [2, 3]


def test_warm_model_load_is_reported_separately_from_every_decode():
    spec = RunSpec(
        methods=("pld",),
        decode=DecodeSpec(max_new=2),
        measurement=MeasurementSpec(warmups=0),
    )
    case = ConversationCase("case-1", "reasoning", (Turn("Question"),))

    result = BenchmarkRunner().run(spec, [case], _TextModel(), model_load_s=7.5)

    assert [m.model_load_s for m in result.measurements] == [7.5, 7.5]


def test_diagnostic_run_scores_output_quality_against_the_baseline_text():
    # When tokens diverge, task quality is judged on the emitted text vs the baseline
    # text, not token identity. Every response carries reference-based scores.
    spec = RunSpec(
        methods=("pld",),
        decode=DecodeSpec(max_new=6),
        measurement=MeasurementSpec(warmups=0),
    )
    case = ConversationCase("case-1", "reasoning", (Turn("Question"),))

    result = BenchmarkRunner().run(spec, [case], _BrokenModel())

    by_method = {r["method"]: r["scores"] for r in result.responses}
    assert by_method["baseline"]["text_similarity"] == 1.0  # reference vs itself
    assert 0.0 <= by_method["pld"]["text_similarity"] <= 1.0


def test_requested_tree_verification_fails_instead_of_silently_falling_back():
    spec = RunSpec(
        methods=("pld",),
        decode=DecodeSpec(max_new=4, tree=True),
        measurement=MeasurementSpec(warmups=0),
    )
    case = ConversationCase("case-1", "reasoning", (Turn("Question"),))

    with pytest.raises(ValueError, match="tree-capable"):
        BenchmarkRunner().run(spec, [case], _TextModel())
