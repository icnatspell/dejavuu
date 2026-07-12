"""Public benchmark-artifact metadata checks."""

import json

from dejavuu.eval.harness import write_run_manifest


def test_write_run_manifest_records_configuration_next_to_csv(tmp_path):
    csv_path = tmp_path / "specbench.csv"
    manifest = write_run_manifest(
        csv_path,
        {"model": "gemma-q4", "provider": "cpu", "threads": 2, "tree": False},
    )

    assert manifest == tmp_path / "specbench.manifest.json"
    assert json.loads(manifest.read_text()) == {
        "model": "gemma-q4",
        "provider": "cpu",
        "threads": 2,
        "tree": False,
    }
