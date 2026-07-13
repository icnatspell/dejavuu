# Seeded-root LogitSpec prototype

## Question

Can LogitSpec improve its draft tree by using the target's final prefill logits to
select a known next-token root, then verifying only descendants of that root?

## Prototype

Commit `e359800` adds `generate_seeded()`. It keeps the complete prompt in the KV
cache during prefill, takes greedy argmax from the final target logits as the first
uncommitted root, and verifies a drafter's children beneath that root. The correction
from the final accepted node becomes the next root. The ordinary `generate()` loop is
unchanged.

The implementation covers text and VLM prefill. The VLM path retains prepared image
embeddings through full-prompt prefill, so it does not silently replace image features
with token embeddings.

## Correctness gate

Two model-free tests compare seeded greedy output to ordinary greedy output: one with
PLD and one with LogitSpec. The full suite passed with 142 selected tests, as did
`uv run prek run --all-files`.

## Direct ONNX Runtime probe

Gemma 3 270M q4, CPU, two ORT threads, one repetitive Spec-Bench prompt, 128 new
tokens. Throughput excludes prefill, matching the repository's benchmark convention.

| Mode | Decode tok/s | Verifier steps | Accepted drafts | Output |
|---|---:|---:|---:|---|
| Baseline | 44.5 | 128 | 0 | exact |
| Current LogitSpec | 52.2 | 51 | 85 | exact |
| Seeded-root LogitSpec | 20.3 | 73 | 59 | exact |

## Interpretation

This prototype is lossless under greedy decoding, but its candidate construction is
not competitive with the current anchor-root LogitSpec: it accepts fewer guesses and
therefore performs more verifier forwards. This result evaluates this narrow
implementation on one prompt. It does not show that seeded-root verification or the
LogitSpec technique is intrinsically unhelpful.

The feature remains an experiment and is not wired into the public API or benchmark
CLI. Revisit it only with a materially different child-candidate policy; otherwise the
next implementation effort should target another method.
