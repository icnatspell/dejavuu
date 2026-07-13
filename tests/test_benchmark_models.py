"""Benchmark-model adapters prepare request-local verifier state."""

import numpy as np
import pytest

from dejavuu.decoders import vlm as vlm_module
from dejavuu.decoders.vlm import PreparedVLMVerifier
from dejavuu.eval.config import ModelSpec
from dejavuu.eval.datasets import ConversationCase, Turn
from dejavuu.eval.models import TextOnnxBenchmarkModel, load_benchmark_model
from dejavuu.tools.artifact import write_manifest


class _FakeVLM:
    supports_tree = True

    def __init__(self):
        self.prefills = []

    def empty_kv(self):
        return []

    def forward_embeds(self, embeds, past, past_len, position_ids=None, attn_bias=None):
        self.prefills.append(np.asarray(embeds).copy())
        return np.zeros((len(embeds), 4)), [], None


def test_prepared_vlm_verifiers_keep_prefill_embeddings_request_local():
    model = _FakeVLM()
    first = PreparedVLMVerifier(model, np.asarray([[1.0], [2.0]]))
    second = PreparedVLMVerifier(model, np.asarray([[8.0], [9.0]]))

    first.prefill([10, 11])
    second.prefill([20, 21])

    np.testing.assert_array_equal(model.prefills[0], [[1.0]])
    np.testing.assert_array_equal(model.prefills[1], [[8.0]])


class _Tokenizer:
    chat_template = "template"
    eos_token_id = 2

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return [7, 8]

    def decode(self, token_ids, **kwargs):
        return "decoded"


def test_text_model_adapter_applies_chat_template_to_conversation_history():
    tokenizer = _Tokenizer()
    adapter = TextOnnxBenchmarkModel(_FakeVLM(), tokenizer, "toy")
    case = ConversationCase("1", "chat", (Turn("Follow up"),))
    history = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Answer"},
    ]

    prepared = adapter.prepare(case, case.turns[0], history)

    assert prepared.prompt_ids == [7, 8]
    assert tokenizer.messages[-1] == {"role": "user", "content": "Follow up"}


def test_text_model_adapter_accepts_transformers_mapping_token_output():
    class MappingTokenizer(_Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [[7, 8]]}

    case = ConversationCase("1", "chat", (Turn("Question"),))
    adapter = TextOnnxBenchmarkModel(_FakeVLM(), MappingTokenizer(), "toy")

    prepared = adapter.prepare(case, case.turns[0], [])

    assert prepared.prompt_ids == [7, 8]


def test_explicit_model_artifact_is_verified_before_loading(tmp_path):
    with pytest.raises(ValueError, match="unverified model artifact"):
        load_benchmark_model(
            ModelSpec(path=str(tmp_path)),
            dataset="specbench",
            protocol="conversation",
        )


def _incompatible_artifact(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"weights")
    write_manifest(
        tmp_path,
        {
            "model_kind": "text_onnx",
            "variants": {
                "q4": {
                    "file": "model.onnx",
                    "speculative_compatible": False,
                }
            },
        },
    )
    return ModelSpec(path=str(tmp_path), variant="q4")


def test_a_variant_marked_incompatible_with_speculation_loads_past_the_gate(tmp_path):
    # A `speculative_compatible: false` variant is never rejected: divergence is a
    # recorded diagnostic, not a validity gate. Loading here still fails on the
    # fixture's missing tokenizer, and matching that error (not a gate error) proves
    # the compatibility gate itself let the artifact through.
    with pytest.raises(ValueError, match="tokenizer"):
        load_benchmark_model(
            _incompatible_artifact(tmp_path),
            dataset="specbench",
            protocol="first-turn-workload",
        )


def test_vlm_rejects_an_unverified_external_decoder(monkeypatch, tmp_path):
    root = tmp_path / "vlm"
    root.mkdir()
    (root / "processor.json").write_text("{}")
    write_manifest(root, {"model_kind": "smolvlm_onnx"})
    decoder = tmp_path / "decoder" / "model.onnx"
    decoder.parent.mkdir()
    decoder.write_bytes(b"weights")

    class FakeVLM:
        def __init__(self, *args, **kwargs):
            self.decoder_path = decoder

    monkeypatch.setattr(vlm_module, "VLM", FakeVLM)

    with pytest.raises(ValueError, match="external VLM decoder"):
        load_benchmark_model(
            ModelSpec(path=str(root), kind="smolvlm_onnx"),
            dataset="mmspec",
            protocol="conversation",
        )
