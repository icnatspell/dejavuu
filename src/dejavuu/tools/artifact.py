"""Provenance + integrity manifest for exported ONNX artifacts.

Hand-run export scripts drop unversioned files in ~/.cache, so a run can't prove
which weights it loaded or notice a half-written / drifted file. A manifest.json next
to the model records sha256 of every artifact plus how it was built; `verify` recomputes
and flags drift. Lazy on purpose: one JSON, no registry, no signing.

    python -m dejavuu.tools.artifact <dir> [<dir> ...]   # verify
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from loguru import logger

MANIFEST = "manifest.json"
_ARTIFACT_SUFFIXES = (".onnx", ".onnx_data", ".data")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _artifacts(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.name.endswith(_ARTIFACT_SUFFIXES))


def write_manifest(d: Path, provenance: dict) -> Path:
    """Stamp manifest.json: sha256 of every artifact in `d` + how it was built."""
    m = {"provenance": provenance, "files": {p.name: _sha256(p) for p in _artifacts(d)}}
    out = d / MANIFEST
    out.write_text(json.dumps(m, indent=2))
    return out


def verify_manifest(d: Path) -> list[str]:
    """Return a list of problems (empty == artifacts match the manifest)."""
    mf = d / MANIFEST
    if not mf.exists():
        return [f"no {MANIFEST} in {d}"]
    want = json.loads(mf.read_text())["files"]
    problems = []
    for name, sha in want.items():
        p = d / name
        if not p.exists():
            problems.append(f"missing {name}")
        elif _sha256(p) != sha:
            problems.append(f"sha256 mismatch {name}")
    for p in _artifacts(d):  # an artifact the manifest never recorded is also drift
        if p.name not in want:
            problems.append(f"untracked {p.name}")
    return problems


def _demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "model.onnx").write_bytes(b"weights")
        write_manifest(d, {"src": "test"})
        assert verify_manifest(d) == [], "fresh manifest must verify clean"
        (d / "model.onnx").write_bytes(b"tampered")
        assert any("mismatch" in p for p in verify_manifest(d)), "must catch drift"
        (d / "extra.onnx").write_bytes(b"x")
        assert any("untracked" in p for p in verify_manifest(d)), "must catch new file"
    logger.info("artifact manifest self-check OK")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _demo()
    else:
        bad = False
        for arg in sys.argv[1:]:
            problems = verify_manifest(Path(arg))
            if problems:
                logger.error("{}: {}", arg, "; ".join(problems))
            else:
                logger.info("{}: OK", arg)
            bad |= bool(problems)
        sys.exit(1 if bad else 0)
