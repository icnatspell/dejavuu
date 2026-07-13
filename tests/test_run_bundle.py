"""Immutable benchmark result bundle behavior."""

import json

import pytest

from dejavuu.eval.run_bundle import RunBundle


def test_run_bundle_is_atomic_immutable_and_writes_responses_without_csv(tmp_path):
    target = tmp_path / "qwen-speed"
    bundle = RunBundle.create(target, {"dataset": "speedbench"})
    bundle.write_responses([{"case_id": "1", "method": "pld", "text": "ok"}])
    bundle.finalize("valid")

    assert json.loads((target / "manifest.json").read_text()) == {
        "dataset": "speedbench",
        "status": "valid",
    }
    assert json.loads((target / "responses.jsonl").read_text()) == {
        "case_id": "1",
        "method": "pld",
        "text": "ok",
    }
    with pytest.raises(FileExistsError):
        RunBundle.create(target, {})
