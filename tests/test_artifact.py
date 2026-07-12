from pathlib import Path

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
