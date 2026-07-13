"""Public benchmark-artifact metadata checks."""

import csv
import json
import subprocess
from pathlib import Path

from dejavuu.eval.harness import (
    Agg,
    benchmark_metadata,
    create_run_dir,
    write_car_profile,
    write_response_jsonl,
    write_run_manifest,
)
from dejavuu.eval.specbench import resolve_model_root


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


def test_specbench_explicit_model_path_bypasses_default_download():
    """A locally built decoder directory is a first-class Spec-Bench target."""
    decoder = Path("/tmp/qwen3-decoder")

    assert resolve_model_root(decoder, "fp32") == decoder


def test_reproducible_decoder_scripts_are_valid_bash():
    """The documented build and tree-benchmark entry points parse before a long run."""
    root = Path(__file__).parents[1]

    for script in ("scripts/build_decoder.sh", "scripts/bench_tree.sh"):
        assert subprocess.run(["bash", "-n", root / script], check=False).returncode == 0


def test_car_profile_is_long_form_by_category_method_and_depth(tmp_path):
    csv_path = tmp_path / "run.csv"
    agg = Agg(conditional_attempts=[4, 2], conditional_accepted=[3, 1])

    profile = write_car_profile(csv_path, {"coding": {"pld": agg}})

    assert profile == tmp_path / "run.car.csv"
    assert list(csv.DictReader(profile.open())) == [
        {
            "category": "coding",
            "method": "pld",
            "depth": "1",
            "opportunities": "4",
            "accepted": "3",
            "conditional_acceptance": "0.7500",
        },
        {
            "category": "coding",
            "method": "pld",
            "depth": "2",
            "opportunities": "2",
            "accepted": "1",
            "conditional_acceptance": "0.5000",
        },
    ]


def test_response_jsonl_preserves_case_method_tokens_and_text(tmp_path):
    path = tmp_path / "run.responses.jsonl"

    write_response_jsonl(
        path,
        [{"case_id": "speed-1", "method": "pld", "tokens": [4, 5], "text": "hello"}],
    )

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"case_id": "speed-1", "method": "pld", "tokens": [4, 5], "text": "hello"}
    ]


def test_run_directory_is_immutable_and_contains_manifest(tmp_path):
    run = create_run_dir(tmp_path, "qwen speed/tree", {"dataset": "speedbench", "budget": 8})

    assert run.name == "qwen-speed-tree"
    assert json.loads((run / "manifest.json").read_text()) == {"budget": 8, "dataset": "speedbench"}
