from pathlib import Path

import pytest

from dejavuu.decoders.ort import make_session
from dejavuu.decoders.text import resolve_graph_path
from dejavuu.decoders.vlm import resolve_vlm_graph_path
from dejavuu.tools.artifact import verify_manifest, write_manifest


def test_manifest_roundtrip_and_drift(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"weights")
    (tmp_path / "model_int8.onnx").write_bytes(b"quant")
    write_manifest(tmp_path, {"src": "test"})
    assert verify_manifest(tmp_path) == []

    (tmp_path / "model_int8.onnx").write_bytes(b"tampered")
    assert any("mismatch" in p for p in verify_manifest(tmp_path))


def test_missing_and_untracked(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"w")
    write_manifest(tmp_path, {})
    (tmp_path / "model.onnx").unlink()
    assert any("missing" in p for p in verify_manifest(tmp_path))

    (tmp_path / "model.onnx").write_bytes(b"w")
    (tmp_path / "new.onnx").write_bytes(b"x")
    assert any("untracked" in p for p in verify_manifest(tmp_path))


def test_no_manifest(tmp_path: Path) -> None:
    assert verify_manifest(tmp_path) == [f"no manifest.json in {tmp_path}"]


def test_manifest_tracks_nested_model_and_tokenizer_files(tmp_path: Path) -> None:
    graph = tmp_path / "onnx" / "model_q4.onnx"
    graph.parent.mkdir()
    graph.write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_text("{}")

    manifest = write_manifest(tmp_path, {"source": "Qwen/Qwen3-0.6B"})

    assert set(__import__("json").loads(manifest.read_text())["files"]) == {
        "onnx/model_q4.onnx",
        "tokenizer.json",
    }
    graph.write_bytes(b"changed")
    assert verify_manifest(tmp_path) == ["sha256 mismatch onnx/model_q4.onnx"]


def test_cuda_provider_request_fails_instead_of_silently_using_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dejavuu.decoders.ort.ort.get_available_providers", lambda: ["CPUExecutionProvider"]
    )

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        make_session(tmp_path / "model.onnx", "cuda")


def test_explicit_provider_fallback_exposes_the_actual_cpu_provider(monkeypatch, tmp_path):
    class Session:
        def __init__(self, path, sess_options, providers):
            self.providers = providers

        def get_providers(self):
            return self.providers

    monkeypatch.setattr(
        "dejavuu.decoders.ort.ort.get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr("dejavuu.decoders.ort.ort.InferenceSession", Session)

    session = make_session(tmp_path / "model.onnx", "cuda", allow_provider_fallback=True)

    assert session.get_providers() == ["CPUExecutionProvider"]


def test_text_decoder_resolves_variant_from_artifact_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        '{"provenance":{"variants":{"q4":{"file":"graphs/qwen.onnx"}}},"files":{}}'
    )

    assert resolve_graph_path(tmp_path, "q4") == tmp_path / "graphs" / "qwen.onnx"


def test_vlm_frontend_graph_roles_come_from_the_artifact_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        '{"provenance":{"graphs":{"vision_encoder":{"q4":"graphs/vision.onnx"}}},"files":{}}'
    )

    assert resolve_vlm_graph_path(tmp_path, "vision_encoder", "q4") == (
        tmp_path / "graphs" / "vision.onnx"
    )
