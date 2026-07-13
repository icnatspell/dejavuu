"""Unified benchmark CLI writes one complete immutable run."""

import json
import sys

import numpy as np

from dejavuu.core.verifier import Verifier
from dejavuu.eval import bench
from dejavuu.eval.datasets import ConversationCase, DatasetProvenance, Turn
from dejavuu.eval.models import ModelIdentity, PreparedInvocation


class _Verifier(Verifier):
    def empty_kv(self):
        return []

    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        logits = np.full((len(token_ids), 3), -1.0, np.float32)
        logits[:, 1] = 1.0
        return logits, [], None


class _Model:
    identity = ModelIdentity("text_onnx", "fixture", "cpu", ("CPUExecutionProvider",))
    artifacts = ({"role": "model", "source_revision": "model-commit"},)

    def __init__(self):
        self.verifier = _Verifier()

    def prepare(self, case, turn, history):
        return PreparedInvocation(self.verifier, [0], None)

    def decode(self, token_ids):
        return "answer"

    def warmup(self, invocation):
        pass

    def extend_history(self, history, turn, response):
        return history


def test_cli_writes_complete_run_without_individual_output_flags(monkeypatch, tmp_path):
    case = ConversationCase(
        "1",
        "qa",
        (Turn("Question"),),
        provenance=DatasetProvenance("fixture", "abc123", "test", "deadbeef"),
    )
    monkeypatch.setattr(bench, "load_cases", lambda *args, **kwargs: [case])
    monkeypatch.setattr(bench, "load_benchmark_model", lambda *args, **kwargs: _Model())
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--dataset",
            "specbench",
            "--methods",
            "baseline",
            "--max-new",
            "2",
            "--warmups",
            "0",
            "--out",
            str(out),
        ],
    )

    bench.main()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "valid"
    assert manifest["run"]["decode"]["max_new"] == 2
    assert manifest["datasets"][0]["revision"] == "abc123"
    assert manifest["model_artifacts"][0]["source_revision"] == "model-commit"
    assert (out / "summary.csv").exists()
    assert (out / "responses.jsonl").exists()
    assert (out / "failures.jsonl").exists()
    measurement = json.loads((out / "measurements.jsonl").read_text().splitlines()[0])
    assert measurement["generation_s"] >= measurement["decode_s"]
    assert measurement["tokens_emitted"] == 2
    assert "baseline_tokens" not in measurement
