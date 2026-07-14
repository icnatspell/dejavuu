"""Unified benchmark CLI: dataset and model selection are independent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from loguru import logger

from dejavuu.eval.config import (
    DatasetSpec,
    DecodeSpec,
    MeasurementSpec,
    ModelSpec,
    OutputSpec,
    RunSpec,
)
from dejavuu.eval.datasets import load_cases
from dejavuu.eval.harness import load_datastore, render_table, write_car_profile
from dejavuu.eval.models import load_benchmark_model
from dejavuu.eval.run_bundle import RunBundle
from dejavuu.eval.runner import BenchmarkRunner


def _parser(default_dataset: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("dejavuu.eval.bench")
    parser.add_argument(
        "--dataset",
        choices=["specbench", "speedbench", "mmspec"],
        default=default_dataset,
        required=default_dataset is None,
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument(
        "--protocol",
        choices=["conversation", "first-turn-workload", "official"],
        default="conversation",
    )
    parser.add_argument("--workload", default="all")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--model-kind", choices=["auto", "text_onnx", "smolvlm_onnx"], default="auto"
    )
    parser.add_argument("--variant", default="q4")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--allow-provider-fallback", action="store_true")
    parser.add_argument("--allow-unverified-artifact", action="store_true")
    parser.add_argument("--methods", default="baseline,pld,token_recycling")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument(
        "--accept-top-k",
        type=int,
        default=1,
        help="loose (lossy) acceptance: 1 = exact lossless (default); >1 accepts a "
        "drafted token in the target's top-k, trading token identity for speed "
        "(quality cost is measured against the greedy baseline via response scorers)",
    )
    parser.add_argument(
        "--accept-entropy-gate",
        type=float,
        default=0.0,
        help="FLy-style gate (0-1, 0=off): only apply --accept-top-k where the target's "
        "normalized entropy exceeds this, so confident positions stay exact "
        "(superseded by --accept-min-prob-ratio)",
    )
    parser.add_argument(
        "--accept-min-prob-ratio",
        type=float,
        default=0.0,
        help="plausibility gate (0-1, 0=off): accept a non-argmax draft only when its "
        "probability is >= this factor of the argmax's (a near-tie). Sharper than the "
        "entropy gate at bounding semantic drift; the recommended lever "
        "(recommended loose recipe: --accept-top-k 3 --accept-min-prob-ratio 0.3)",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--cache-scope", choices=["request", "run"], default="run")
    parser.add_argument("--model-memory", choices=["cold", "warm"], default="warm")
    parser.add_argument("--datastore", type=Path, default=None)
    parser.add_argument("--cacheback-frozen", type=Path, default=None)
    parser.add_argument("--reset-drafter-per-prompt", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    # One-cycle compatibility: copies are non-appendable and the run bundle remains canonical.
    parser.add_argument("--csv", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--log", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--responses", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def _run_spec(args: argparse.Namespace) -> RunSpec:
    split = (
        args.split
        or {
            "speedbench": "qualitative",
            "mmspec": "testmini",
            "specbench": "test",
        }[args.dataset]
    )
    target = args.out
    if target is None:
        if args.csv is not None:
            target = args.csv.with_suffix("")
        else:
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            target = Path("results") / f"{args.dataset}-{stamp}"
    return RunSpec(
        dataset=DatasetSpec(
            name=args.dataset,
            split=split,
            revision=args.dataset_revision,
            protocol=args.protocol,
            workload=args.workload,
            n=args.n,
            per_category=args.per_category,
            image_size=args.image_size,
        ),
        model=ModelSpec(
            path=str(args.model_path) if args.model_path else None,
            kind=args.model_kind,
            variant=args.variant,
            provider=args.provider,
            threads=args.threads,
            allow_provider_fallback=args.allow_provider_fallback,
            allow_unverified_artifact=args.allow_unverified_artifact,
        ),
        methods=tuple(part.strip() for part in args.methods.split(",") if part.strip()),
        decode=DecodeSpec(
            max_new=args.max_new,
            budget=args.budget,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            tree=args.tree,
            width=args.width,
            accept_top_k=args.accept_top_k,
            accept_entropy_gate=args.accept_entropy_gate,
            accept_min_prob_ratio=args.accept_min_prob_ratio,
        ),
        measurement=MeasurementSpec(
            warmups=args.warmups,
            repetitions=args.repetitions,
            order_seed=args.order_seed,
            cache_scope="request" if args.reset_drafter_per_prompt else args.cache_scope,
            model_memory=args.model_memory,
        ),
        output=OutputSpec(path=str(target)),
    )


def _copy_legacy(source: Path, target: Path | None) -> None:
    if target is None:
        return
    if target.exists():
        raise FileExistsError(f"legacy benchmark output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except Exception:  # pragma: no cover - editable/unpacked environments
        return "unknown"


def main(default_dataset: str | None = None) -> None:
    args = _parser(default_dataset).parse_args()
    spec = _run_spec(args)  # validate before model or dataset loading
    cases = load_cases(
        spec.dataset.name,
        split=spec.dataset.split,
        revision=spec.dataset.revision,
        protocol=spec.dataset.protocol,
        n=spec.dataset.n,
        per_category=spec.dataset.per_category,
        workload=spec.dataset.workload,
    )
    model_started = time.perf_counter()
    model = load_benchmark_model(
        spec.model,
        dataset=spec.dataset.name,
        protocol=spec.dataset.protocol,
        image_size=spec.dataset.image_size,
    )
    model_load_s = time.perf_counter() - model_started
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None and hasattr(model, "processor"):
        tokenizer = model.processor.tokenizer
    datastore = load_datastore(args.datastore, tokenizer) if args.datastore else None
    drafter_builder = None
    if args.cacheback_frozen is not None:
        from dejavuu.drafters import Cacheback, make_drafter

        def frozen_drafter_builder(method: str):
            return (
                Cacheback.from_frozen(args.cacheback_frozen)
                if method == "cacheback"
                else make_drafter(method, datastore)
            )

        drafter_builder = frozen_drafter_builder
    result = BenchmarkRunner().run(
        spec,
        cases,
        model,
        datastore,
        drafter_builder,
        model_load_s=model_load_s,
    )

    provenance = [asdict(case.provenance) for case in cases if case.provenance is not None]
    unique_provenance = list({tuple(sorted(item.items())): item for item in provenance}.values())
    serialized_spec = spec.model_dump_json()
    artifact_manifest = None
    if spec.model.path:
        path = Path(spec.model.path) / "manifest.json"
        if path.exists():
            artifact_manifest = json.loads(path.read_text())
    model_artifacts = getattr(model, "artifacts", ())
    manifest = {
        "schema_version": 2,
        "run_id": Path(spec.output.path).name,
        "config_sha256": hashlib.sha256(serialized_spec.encode()).hexdigest(),
        "run": spec.model_dump(mode="json"),
        "model": asdict(model.identity),
        "model_load_s": model_load_s,
        "model_artifact": artifact_manifest,
        "model_artifacts": model_artifacts,
        "datasets": unique_provenance,
        "software": {
            package: _installed_version(package)
            for package in ("dejavuu", "onnxruntime", "transformers")
        },
        "host": {
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "phase_definitions": {
            "prepare": "online once: render/tokenize/image decode/processor/vision encode",
            "prefill": "online once: prompt KV construction",
            "draft_setup": "online once: drafter reset for the request",
            "decode": "online hot path: draft + verify + learn + overhead",
        },
    }
    bundle = RunBundle.create(Path(spec.output.path), manifest)
    summary = bundle.path("summary.csv")
    log = bundle.path("logs/runner.log")
    render_table(
        f"{spec.dataset.name} / {spec.model.variant} (n={len(cases)})",
        list(spec.methods),
        result.aggs,
        log,
        strict=False,  # divergence is diagnostic: show token-match %, never a pass/fail gate
        csv_path=summary,
    )
    car = write_car_profile(summary, result.aggs)
    car.rename(bundle.path("car.csv"))
    bundle.write_responses(result.responses)
    bundle.write_divergences(result.divergences)
    bundle.write_jsonl(
        "measurements.jsonl", [measurement.as_record() for measurement in result.measurements]
    )
    # Divergence never invalidates a run; it is a recorded diagnostic. The bundle is
    # always valid -- infra errors would already have raised before reaching here.
    status = "valid_with_divergences" if result.has_divergences else "valid"
    final = bundle.finalize(status)
    _copy_legacy(final / "summary.csv", args.csv)
    _copy_legacy(final / "logs/runner.log", args.log)
    _copy_legacy(final / "responses.jsonl", args.responses)
    logger.info("run bundle -> {}", final)
    if result.has_divergences:
        logger.info(
            "{} speculative outputs diverged from baseline; recorded in divergences.jsonl, "
            "all performance and quality metrics retained",
            len(result.divergences),
        )


if __name__ == "__main__":
    main()
