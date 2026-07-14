# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `ngram_backoff` drafter: a memory-bounded, multi-order n-gram cache that stores
  single-token continuations at every order in a range and drafts longest-context-first
  with high->low backoff (inspired by NG+, issue #7). Distinct from `cacheback` (single
  fixed order) and `ngram_trie` (prompt-rebuilt). Registered, chain/tree conformance
  bit-exact; default `min_order=2` (unigram backoff drafts low-confidence guesses).

## [0.2.0] - 2026-07-14

### Added
- **New drafters**, all registered in `DRAFTERS` (so they wire into the CLI, both
  benches, and the chain/tree conformance suite automatically): `copyspec`
  (earliest-occurrence k-gram continuation copying), `cacheback` (bounded online n-gram
  cache, plus loadable versioned frozen tables via `Cacheback.from_frozen` /
  `tools.build_cacheback_table`), `logit_spec` (verifier-logit candidates extended by
  n-gram retrieval), `ngram_trie` (prompt continuation trie with deep tree branches),
  `stand` (Gumbel-ranked sampled tree drafting), and a `hybrid` family (retrieval with a
  verifier-logit fallback, plus grafted-merge and tail-extension variants).
- **GPU decode path.** Device-resident KV via ONNX Runtime io-binding keeps the KV on the
  accelerator across steps for chain and tree (`OrtDecoder(device_kv=True)`, auto-enabled
  on `provider=cuda`; the CPU/numpy path stays byte-identical and conformance untouched).
  genai fused chain variants (`int4_genai` primary, `fp16_genai` fallback) built by
  `dejavuu.tools.build_genai_chain`, each gated on batched-vs-incremental self-consistency
  and top-1 fidelity vs the eager fp32 reference. Guarded CUDA runtime-library preload and
  a per-run draft-source attribution counter for the hybrid methods.
- Loose (lossy) verification: an opt-in accept rule that trades token identity for speed.
  `accept_top_k > 1` accepts a drafted token in the target's top-k; `accept_min_prob_ratio`
  is a plausibility gate that only accepts a non-argmax draft when its probability is within
  that factor of the argmax (a near-tie), directly bounding semantic drift. Off by default
  (`accept_top_k=1` stays exact and bit-exact under conformance). On SPEED-Bench qualitative
  (Qwen3-0.6B fp32, tree, budget 4), `--accept-top-k 3 --accept-min-prob-ratio 0.3` is a
  Pareto win over bare top-k -- higher decode speedup (1.66x vs 1.60x) and higher semantic
  fidelity at nearly equal acceptance. `scripts/rescore.py` re-scores existing bundles by
  meaning (model2vec static embeddings) so divergent output is judged on semantics, not
  character overlap. See `docs/methods.md` -> "Loose (lossy) verification".
- Token divergence from the baseline is recorded as a diagnostic and never fails a
  benchmark run. The runner writes first-divergence position, exact-match, and overlap
  to `divergences.jsonl`, marks the bundle `valid_with_divergences`, and keeps measuring
  latency/throughput/acceptance/phase costs; a `speculative_compatible: false` variant
  loads with a warning instead of being rejected. Model-free chain/tree conformance
  (`tests/test_conformance.py`) stays bit-exact and is where lossless correctness is
  enforced.
- Reference-based response-quality scorers (`eval/scorers.py`): every response records
  `scores` against its baseline text so diverging output can still be judged on task
  quality. Ships a stdlib `text_similarity` (alignment-based) scorer; register more in
  `SCORERS`.

- Unified reproducible benchmark runner with validated run specifications, full-conversation
  dataset adapters, independent text/VLM model adapters, warm/cold memory modes,
  repetitions, cache scopes, balanced method scheduling, and immutable result bundles.
  Adds SPEED-Bench support, benchmarking of built tree decoders, per-prompt progress
  logging, and recorded run-time metadata.
- Pinned SpecBench, SPEED-Bench, MMSpec, Gemma, and SmolVLM source revisions plus
  recursive model-artifact integrity manifests and manifest-selected ONNX graph roles.
- Separate response, failure, and phase-measurement JSONL records, including selected
  VLM graph and external-decoder provenance in every run bundle.
- A `learn` profiler phase timing the post-verify drafter callbacks (`observe`/`update`),
  partitioning the decode total alongside `prefill/draft/verify/overhead`.
- `scripts/gap_rank.py`: ranks methods across a sweep by speedup and labels each one's
  dominant reducible gap (acceptance / learn / overhead / verify-bound) from the phase split.
- CI publishes a GitHub release when a `v*` tag is pushed.

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
- `pld` default `n_max` 3->4 and `copyspec` now tries shorter k-grams, both raising
  acceptance on int8 where the gains convert to speedup.
- Benchmark internals refactored: shared text/VLM benchmark loops and the removal of the
  dead legacy `run_cases` path. Method fidelity matrix documented (`docs/method-fidelity.md`).

### Fixed
- `asd_verify`'s verify-aware draft sizing no longer collapses to length-1 drafts on
  launch-bound backends. Its cost model fits `verify_s ~ c0 + c1*submitted`, but the
  no-draft (M=1) steps were folded into that fit; on a backend whose verify cost is a
  step function (a large fixed jump from M=1 to M>=2, then a small per-token slope — e.g.
  the int4_genai GPU chain path, measured ~7ms at M=1, +6.6ms at M>=2, ~1ms/token after),
  those cheap samples smeared the jump into the per-token slope and over-penalised length.
  Only drafted (M>=2) steps now inform the fit, so the sizer recovers the true (cheap)
  marginal cost and drafts long again (danube int4_genai: +14% decode speedup at budget 4,
  and it scales with budget instead of regressing). Lossless behaviour is unchanged —
  sizing only shortens a pure-retrieval draft; chain/tree conformance stays bit-exact.

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

[Unreleased]: https://github.com/icnatspell/dejavuu/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/icnatspell/dejavuu/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/icnatspell/dejavuu/releases/tag/v0.1.0
