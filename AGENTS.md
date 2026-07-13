# Working agreement

## Issue-first integration

Before starting any integration work, check GitHub for an existing issue with
`gh issue list --state all`. Link the work to that issue when one exists; otherwise,
create an issue that describes the method or integration, intended benefit, expected
drafter/verifier impact, and primary paper or code reference before implementation.
Do this before creating an implementation branch or changing code.

## Engineering standards

- Write comments, docstrings, and benchmark explanations for an AI/ML engineer.
  When audience expertise is uncertain, target a junior AI/ML engineer: define
  spec-decoding terms, explain why a measurement or constraint matters, and avoid
  unexplained runtime jargon.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
  messages and [Semantic Versioning](https://semver.org/) for released versions.
- Run `uv run prek run --all-files` before handing off changes. CI runs the same
  suite (plus `pip-audit`) and is the final enforcement layer.

## The drafter contract is the standardization boundary

Every drafter is a drop-in: it works on raw token-id lists only (never model
internals) and emits a `DraftTree`, so the same instance drives the LLM (`Model`)
and the VLM (`VLM`) through the shared `Verifier`/engine. Preserve this:

- A new drafter subclasses `Drafter` and is registered in `eval.harness.DRAFTERS`.
  Registration alone wires it into the CLI, both benches, and the conformance
  suite -- no per-drafter test scaffolding needed.
- **Every method must be usable under both chain and tree verification.**
  `propose` is the chain path; `propose_tree` is the branching path (default: the
  chain, so tree support is free unless you have ranked candidates to branch on).
  `tests/test_conformance.py` enforces this: it runs every `DRAFTERS` entry
  through a model-free verifier under `tree=False` and `tree=True` and asserts a
  valid tree shape plus bit-exact output vs the autoregressive baseline. If your
  drafter can't pass both, it isn't done.
- Losslessness is the verifier's job, not the drafter's. A drafter may propose
  anything; correctness comes from greedy accept / seeded-sample coupling in the
  engine. Preserve that acceptance logic in model-free conformance tests; do not
  change it merely to hide a mismatch from a particular backend.

## Treat backend exactness as a metric, not a universal validity gate

The model-free conformance suite remains bit-exact because it protects the engine's
acceptance and KV-path invariants. Real inference backends, especially quantized
graphs, may select different tokens for incremental and multi-token execution because
their kernels use different numerical paths. That does not automatically invalidate a
performance benchmark or indict a drafter.

- Benchmark divergence is **always diagnostic, never a failure**. Record exact-match
  rate, first-divergence position, and token overlap alongside task-quality metrics, but
  keep measuring latency, throughput, acceptance, and phase costs after a divergence. No
  benchmark code path may mark a run invalid, reject an artifact, or exit nonzero because
  generated tokens differ from the baseline.
- Lossless correctness is enforced only in the model-free conformance suite
  (`tests/test_conformance.py`), which stays bit-exact to protect the engine's accept/KV
  invariants. That is the place for a bit-exact claim -- not a mode on the practical
  benchmark runner.
- Compare methods against the same model artifact, provider, precision, prompt set,
  decoding policy, and draft budget. Do not compare a quantized method run with an FP32
  baseline and call the difference a drafter regression.
- Describe numerical behavior accurately: distinguish backend precision or
  sequence-shape variation from a broken accept/KV implementation. Use task-level or
  semantic quality evaluation when token identity is not the product requirement.

## Classify every cost into one tier

1. **Offline:** export, conversion, quantization, tree-capable re-export.
2. **Online once:** session setup, artifact load, prompt prefill / KV precompute.
3. **Online hot path:** per-step draft, verify forward, accept, KV composition,
   sampling.

Never present a tier-one or tier-two saving as a tier-three latency/TPS gain, and
report `tok/s`/speedup as decode-only (prefill excluded -- see `render_table`).

## Never call a technique dead on an unoptimized implementation

A negative result only indicts the version you measured. Before writing "no-go",
"dead", or any verdict that stops a line of work, confirm you are assessing an
*adequately optimized* implementation of the idea.

- Separate "this **implementation** is slow/low quality" from "the **technique**
  cannot work here". Only the second justifies stopping; most first attempts are
  the first.
- State the optimization level alongside the result (e.g. "brute-force cosine, no
  ANN"). A number without that label is not a verdict.
- Prefer "conditional / needs X" over "no-go" whenever a named lever is untried.
  Reserve "dead" for when the theoretical ceiling itself is unfavorable.

## Profile before you optimize

Optimize what the numbers indict, not what you suspect. The harness (`render_table`,
both benches, the CSV) already splits every decode step into
`prefill / draft / verify / learn / overhead` from the timers on `GenResult` --
proposing, the model forward, the post-verify drafter callbacks (`observe`/`update`),
and accept/KV bookkeeping. Read that split first and attack the dominant bucket. If it
is too coarse, **extend the profiler, not a one-off script:** add the timer to
`GenResult` and surface it through `Agg` / `render_table` / the CSV so it covers every
method, and have the new bucket partition the same decode total (subtract it out of
`overhead`) so the columns still sum -- the `learn` bucket was added exactly this way.
After improving, re-run the same bench and quote the before/after split, confirming the
bucket shrank and reporting any acceptance, exactness, or task-quality change.

Across a whole sweep, `scripts/gap_rank.py <csv>` ranks methods by speedup and labels
each one's dominant reducible gap (acceptance / learn / overhead / verify-bound) off the
same phase split -- that ranked list *is* the "concrete remaining levers" to report.

When *comparing* methods, hold the draft budget -- and hence the drafted/verified token
count -- equal. The verifier forward is not flat in sequence length (on the ONNX CPU
path a 1-token and a multi-token step differ by a large fixed jump, then cost roughly
linearly per extra draft token), so a method that submits more tokens per step pays more
verify time regardless of draft quality; a gap between methods at different budgets
mostly measures the budget. Compare at the same `--budget`, chain/tree, and `--width`,
and lean on acceptance and per-emitted-token time -- which normalize for draft length --
when a method genuinely needs a different length to shine.

## Test-first delivery

Use a vertical red-green-refactor loop: write one public-interface test for the
highest-value behavior, watch it fail, implement the minimum to pass, repeat one
behavior at a time. Tests assert observable behavior, not private internals. For
performance-sensitive code, test correctness and protocol invariants separately
from machine-specific timing thresholds.

## Method integration loop

For every new decoding method, work autonomously through this loop: check or create
the tracking issue; write a public-interface test; implement in small vertical slices;
run chain/tree conformance and sampled-distribution checks where applicable; benchmark
against relevant existing methods; then report measured benefit, optimization level,
known paper deviations, and concrete remaining levers. Do not stop for routine
approval between these steps.

Report every column `render_table` emits (the phase split above, `tok/s`, accepted
length, acceptance rate, submitted/verified counts, time per emitted token), plus, for
proposal quality, top-1 and top-5 target-agreement at the draft root -- or say clearly
when a backend can't expose one. Add method-specific metrics as useful, and report cold
and warm cache/memory modes separately.
