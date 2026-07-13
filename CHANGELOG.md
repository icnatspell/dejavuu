# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Unified benchmark runner with validated run specifications, full-conversation dataset
  adapters, independent text/VLM model adapters, warm/cold memory modes, repetitions,
  cache scopes, balanced method scheduling, and immutable result bundles.
- Pinned SpecBench, SPEED-Bench, MMSpec, Gemma, and SmolVLM source revisions plus
  recursive model-artifact integrity manifests and manifest-selected ONNX graph roles.
- Separate response, failure, and phase-measurement JSONL records, including selected
  VLM graph and external-decoder provenance in every run bundle.

### Changed
- Benchmark throughput now excludes model preparation, KV prefill, and per-request
  drafter setup; all online-once costs are reported separately from the decode hot path.
- Every benchmark modality now requires bit-exact output against its own autoregressive
  baseline. Divergent VLM runs are retained as invalid diagnostics, never valid speedups.
- CUDA provider requests fail when CUDA is unavailable unless fallback is explicit and
  the actual provider is recorded.
- Text adapters normalize list, tensor, and mapping tokenizer outputs across supported
  Transformers versions; externally selected VLM decoders must pass integrity checks.
- Decoder builds now measure batched-causal versus incremental KV-cache agreement and
  mark sequence-length-sensitive quantized variants incompatible with strict
  speculative benchmarks.

## [0.1.0] - 2026-07-12

First public release of `dejavuu`.

### Added
- Hugging Face transformers backend (`backend="hf"`, `hf` extra): run any
  `AutoModelForCausalLM` through the spec-decode engine with no ONNX export. Chain **and
  tree** verification, both bit-exact with greedy and tested lossless on every drafter.
  Tree needs no re-export: eager attention honours the engine's 4D additive mask +
  explicit position_ids, so tree verification and the representation drafters
  (`pld_plus`/`adapld`) work on any HF causal LM. KV management moved behind the backend
  interface (`Verifier.rollback_kv`/`gather_kv`, numpy default), so a torch cache plugs
  in without touching the engine. `attn_implementation` is selectable ("eager" default,
  "sdpa" validated lossless for GPU perf). Validated bit-exact on real weights
  (SmolLM2-135M) as well as a random tiny model, chain and tree.
- The PyPI distribution, import package, and CLI are all named `dejavuu`.
- `dejavuu.drafters.DRAFTERS` registry and `make_drafter` now live in the library, so
  the CLI, the `DejaVu` API, and the benchmark harnesses share one source of truth and
  a plain `import dejavuu` no longer pulls the benchmark dependencies.
- `vlm`, `bench`, and `build` optional-dependency extras. The base install is the
  text-path library only (no torch, onnx, or pillow).
- Chain and tree verification for every registered method; `pld` and the `asam` family
  now emit genuine branching drafts under tree verification.
- Registry conformance suite (`tests/test_conformance.py`): every method must be valid
  and bit-exact with the baseline under both chain and tree verification.
- Packaging metadata (keywords, classifiers, project URLs) and a clean-env install-smoke
  CI job that proves the base wheel imports without the extras.
- Tooling gate: ruff (full ruleset), pyrefly, deptry, pip-audit, prek pre-commit hooks,
  and coverage gating at 90%.
- Validated configuration via pydantic (`dejavuu.config`): `GenerationConfig` and
  `ModelConfig` check bounds (temperature, top_p, budget, ...), the method name, and the
  backend/device combination at the API and CLI boundary. Unknown methods raise a clear
  error listing valid names instead of a bare `KeyError`.
- `DejaVu.generate` now exposes `tree=`/`width=` for tree verification; decode arguments
  are keyword-only.

### Changed
- Registry entries are typed `DrafterSpec`s (factory + capabilities) instead of a bare
  name->class map, removing the `None` baseline sentinel, the `partial` presets, and the
  separate `_DATASTORE_METHODS` set. `METHODS` lists `baseline` plus every drafter.
- VLM backends are identified by `Verifier.is_vlm`/`prepare` on the contract rather than
  by `hasattr` duck-typing in the API.
- Dependency floors relaxed from newest-release pins to conservative lower bounds on the
  APIs actually used.
- Moved the offline model-build toolchain dependencies (`onnx`, `onnxruntime-genai`,
  `onnxscript`, `onnx-ir`) out of the base install into the `build` extra.

### Removed
- Unused `num2words` dependency.

[Unreleased]: https://github.com/icnatspell/dejavuu/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/icnatspell/dejavuu/releases/tag/v0.1.0
