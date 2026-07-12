"""Public benchmark-artifact metadata checks."""

import json

from dejavuu.eval.harness import benchmark_metadata, write_run_manifest


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


def test_benchmark_metadata_keeps_execution_settings_explicit():
    metadata = benchmark_metadata(
        dataset="specbench",
        model="gemma-q4",
        provider="cpu",
        threads=2,
        budget=8,
        tree=True,
        width=2,
        max_new=128,
    )

    assert metadata["benchmark"] == "specbench"
    assert metadata["model"] == "gemma-q4"
    assert metadata["decode"] == {"budget": 8, "max_new": 128, "tree": True, "width": 2}
    assert metadata["runtime"] == {"provider": "cpu", "threads": 2}
    assert {"cpu_count", "machine", "processor"} <= metadata["host"].keys()
