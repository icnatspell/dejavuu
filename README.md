# dejavuu

Training-free, lossless speculative decoding for
[ONNX Runtime](https://onnxruntime.ai/) and Hugging Face PyTorch models—with one
benchmarking workflow for text LLMs and vision-language models.

[![CI](https://github.com/icnatspell/dejavuu/actions/workflows/ci.yml/badge.svg)](https://github.com/icnatspell/dejavuu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/icnatspell/dejavuu/blob/main/LICENSE)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)

Speculative decoding drafts several tokens cheaply, then checks them in one model
forward pass. The target model's accept rule keeps only the tokens it would have
produced anyway, so a wrong guess wastes one pass and never changes the output. The
speedup comes from replacing many single-token passes with fewer multi-token ones.

`dejavuu` is model-free in the drafting sense: it removes the small auxiliary draft
model that speculative decoding usually needs. Its drafters copy guesses from text the
target has already seen—the prompt, the generation so far, or a fixed corpus. A guess
is an index lookup, not a second network. Every drafter works on raw token IDs, so the
same method drives a text LLM or vision-language model through one verifier interface.

Why use it:

- **No draft-model tax:** nothing extra to train, load, or keep in accelerator memory.
- **Losslessness is enforced:** every speculative response is compared token-for-token
  with the same backend's autoregressive baseline, for text and vision.
- **Backends are interchangeable:** use a local ONNX graph or any Hugging Face causal
  LM; drafters never depend on model internals.
- **Methods are genuinely drop-in:** every registered drafter supports chain and tree
  verification through the same raw-token `DraftTree` contract.
- **Benchmarks are comparable:** Spec-Bench, SPEED-Bench, and MMSpec share one validated
  configuration, scheduler, profiler, divergence diagnostics, and output schema.
- **Runs are reproducible:** datasets and source models are revision-pinned, artifacts
  are SHA-256 verified, provider fallback is explicit, and output bundles are immutable.

[docs/methods.md](docs/methods.md) explains how the drafters differ and when each one wins.

## Where dejavuu fits

Dejavuu is designed for **input-grounded** generation: tasks where a useful part of the
answer is present in, or closely constrained by, the prompt or an attached datastore.
That includes retrieval-augmented generation (RAG), function calling and other
structured output, summarization, code completion, and similar copy- or
context-heavy workloads. These methods can turn repeated token patterns into drafts
without training or loading a second model.

For tasks whose answer is weakly grounded in the input—such as open-ended translation
or generation—a learned, model-based drafter such as EAGLE-3 or DFlash may be a better
fit. This is a workload distinction, not a universal ranking: measure acceptance,
verify cost, and end-to-end decode throughput on the target model and prompts before
choosing a method.

## Quickstart

Install the library from PyPI (Python 3.13+). The base install is the text path and is
torch-free; the import package is `dejavuu`:

```bash
pip install dejavuu          # library + text path + the `dejavuu` CLI
pip install "dejavuu[hf]"    # run any transformers model (torch), no ONNX export
pip install "dejavuu[vlm]"   # add a vision-language model (torch, torchvision)
pip install "dejavuu[build]" # add the offline model-build toolchain
```

Run a local ONNX export (the default backend):

```python
from dejavuu import DejaVu

model = DejaVu.from_pretrained("onnx-community/gemma-3-270m-ONNX", method="pld")
print(model.generate("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return"))
```

Or run any Hugging Face causal LM with no export (`backend="hf"`, explicit `device`).
Chain and tree verification both work with no re-export, on any model:

```python
model = DejaVu.from_pretrained(
    "meta-llama/Llama-3.2-1B", backend="hf", device="cuda", method="pld"
)
print(model.generate("The capital of France is"))
```

The same generation from the CLI, no VLM build required:

```bash
dejavuu "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return" --method pld
```

`--method` (or `method=`) takes any name from the [table below](#methods). Add
`--temperature`, `--top-p`, or `--seed` to switch from greedy to sampling. To hack on
the repo instead of installing, use `uv sync` and prefix commands with `uv run`.

### Benchmark in two minutes

The smallest end-to-end run downloads a pinned ONNX model and one pinned Spec-Bench
case, compares PLD with its baseline, and writes a self-contained result bundle:

```bash
git clone https://github.com/icnatspell/dejavuu.git
cd dejavuu
uv sync --extra bench

uv run python -m dejavuu.eval.bench \
    --dataset specbench --protocol first-turn-workload \
    --methods baseline,pld --n 1 --max-new 16 \
    --warmups 0 --out results/quickstart
```

Open `results/quickstart/summary.csv` for the comparison and
`results/quickstart/manifest.json` for the exact model, dataset, provider, software,
and measurement configuration. Output directories are immutable; choose a new
`--out` path for each run. The quick start uses `first-turn-workload` because the
small default Gemma model is a base model without a chat template; instruct models
such as Qwen use the full `conversation` protocol by default.

## Results

Token divergence is diagnostic, never a failure. When a speculative method's tokens
differ from the same backend's autoregressive baseline — a quantized graph's multi-token
forward can pick a different argmax than incremental decoding — that is a backend
numerical property, not a broken drafter, and it never invalidates a run. The runner
records every divergence (exact-match, first-divergence position, token overlap) in
`divergences.jsonl`, marks the bundle `valid_with_divergences`, and keeps measuring
latency, throughput, acceptance, and phase costs. Each response also carries reference
`scores` (e.g. `text_similarity`) against its baseline text, so diverging output is still
judged on task quality rather than token identity. A variant flagged
`speculative_compatible: false` loads with a warning, not a rejection. Compare methods
only against the same model artifact, provider, precision, prompt set, and draft budget.

Bit-exactness is still enforced where it protects correctness: the model-free chain/tree
conformance suite (`tests/test_conformance.py`) asserts every drafter reproduces the
autoregressive baseline exactly, guarding the verifier's accept/KV invariants.

## Methods

Every drafter runs on raw token ids, so the same instance drives both the LLM and the
VLM. REST, SuffixDecoding, SAM, and ASAM share one reusable token-only `SuffixIndex`.

| method | idea |
|---|---|
| `baseline` | plain autoregressive, no drafter |
| `pld` | prompt-lookup: longest suffix match within the context |
| `copyspec` | earliest matching k-gram continuation copying from prompt/history |
| `pld_plus` | `pld` plus hidden-state reranking of matches (needs a hidden-emitting decoder; else falls back to `pld`) |
| `adapld` | `pld_plus` with a semantic fallback, plus a branched tree under `--tree` |
| `anpd` | adaptive n-gram draft length |
| `lookahead` | multi-candidate n-gram pool |
| `logit_spec` | verifier-logit candidates extended by n-gram retrieval |
| `ngram_trie` | prompt n-gram continuation trie with deep tree branches |
| `token_recycling` | tree drafts from the verifier's own logits |
| `stand` | probability-ranked n-gram trees learned from verifier logits |
| `cacheback` | bounded LRU cache of recent leader/follower n-grams |
| `rest` | retrieval from a static datastore |
| `suffix_decoding` | online suffix index over global and per-request history |
| `sam_decoding` | static datastore plus live generation; drafts from the longer match |
| `asam` / `asam_verify` | adaptive SAM with an acceptance-calibrated cap, plus verify-cost-aware sizing |
| `asd` / `asd_verify` | `asam` without a datastore, so an adaptive suffix decoder |

The drafters are lossless under greedy decoding: they only choose tokens to propose,
while the verifier decides what is emitted. Every method emits a chain by default and a branching tree under `--tree`
(`pld_plus` and `anpd` fall back to a chain, since neither has a natural fork).

## How it works

Each decode step drafts several tokens, verifies them in one forward pass, and keeps
only the prefix the model would have produced on its own:

```
prompt --prefill--> KV cache, anchor token
loop until max_new / EOS:
  1. drafter.propose(ctx) ------------------> DraftTree   (chain, or a branching tree)
  2. flatten the tree -----------------------> position_ids + attention mask
  3. backend.forward(tokens, kv, mask) ------> logits (+ optional hidden states)
  4. accept: descend while the model's pick --> emitted tokens + accepted path
     matches a drafted child                   (argmax, or a position-seeded sample)
  5. roll the KV cache back to the accepted path; the last token is the next anchor
```

A wrong guess costs one forward pass and is discarded at step 4, so the output is
identical to plain decoding. Steps 1, 2, and 4 are **backend-agnostic** (they work on
token ids and a logits array). Only step 3 (the forward) and the KV rollback in step 5
are **backend-specific**, and both live behind the `Verifier` contract:

```
Verifier                       # the backend seam (dejavuu/core/verifier.py)
  forward(tokens, kv, pos, mask) -> logits, kv, hidden
  rollback_kv / gather_kv        # chain / tree KV trim (numpy default; a backend overrides)
```

Two backends implement it. **ORT** (`backend="ort"`, default) runs a local ONNX export
and auto-derives its I/O contract from the graph. **HF** (`backend="hf"`) runs any
transformers `AutoModelForCausalLM` with no export; eager attention honours the tree's
4D mask, so chain and tree verification and the hidden-state drafters all work on any
model. A drafter never sees a tensor or a model config, so the same instance drives both
backends and both text and vision.

## Reproduce the benchmark

### Benchmark compatibility

| workload | accepted model adapter | default protocol | what this runner reports |
|---|---|---|---|
| Spec-Bench | text ONNX or SmolVLM | full conversation | quality, exactness, and single-request decode performance |
| SPEED-Bench `qualitative` | text ONNX | full conversation | semantically diverse quality and single-request performance |
| SPEED-Bench `throughput_*` | text ONNX | first-turn workload only | controlled long-context single-request measurements |
| MMSpec | SmolVLM | full conversation with images | multimodal quality, exactness, and decode performance |

NVIDIA's **official** SPEED-Bench throughput protocol measures concurrent serving on
vLLM, SGLang, or TensorRT-LLM. This ONNX Runtime runner rejects
`--protocol official` for throughput splits instead of presenting single-request
numbers as server throughput. See the
[SPEED-Bench dataset](https://huggingface.co/datasets/nvidia/SPEED-Bench) and
[official measurement framework](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/specdec_bench)
for that serving experiment.

A fresh clone reaches the full numbers in a few steps. Both benchmarks run the one
SmolVLM2 model, text Spec-Bench and vision MMSpec, so the results compare across
datasets.

```bash
# 1. install (--extra vlm pulls torch/torchvision, for image preprocessing only)
uv sync --extra vlm

# 2. sanity: unit tests + a drafter self-check
uv run pytest -q
uv run python -m dejavuu.drafters.asam

# 3. one-time: build the SmolVLM2 tree+hidden decoder (weights auto-download from HF;
#    this decoder isn't published, so we build it). Emits fp32 + int8 + int4 (int4 body
#    with an int8 lm-head, ~234MB) into ~/.cache/dejavuu/smolvlm2_decoder_tree_embeds/
#    (NOT in the repo -- rebuild it per machine). The runtime auto-prefers int4. It is
#    tree-capable AND emits hidden states, so pld_plus/adapld and --tree both work.
#    --with onnx_ir enables the int4 step (without it you still get fp32 + int8).
uv run --extra vlm --with onnx_ir --index-strategy unsafe-best-match \
    python -m dejavuu.tools.build_tree_decoder
#    re-quantize only (fast, skips the ~4-min torch re-export) if the fp32 model exists:
#    uv run --extra vlm --with onnx_ir --index-strategy unsafe-best-match \
#        python -m dejavuu.tools.build_tree_decoder --quant-only

# 4. one-time: build the retrieval datastore (regenerable; gitignored)
uv run python -m dejavuu.tools.build_specbench_corpus   # -> data/specbench_corpus.txt

# 5. smoke: 1 prompt/topic, both benches, all methods, downscaled images
./scripts/bench_all.sh 1 512

# 6. full run: all samples/topic, detached so it survives your shell
nohup ./scripts/bench_all.sh 80 512 > results/run.out 2>&1 &
tail -f results/run.out

# 7. same sweep with tree-based verification instead of chain (4th arg = 1)
./scripts/bench_all.sh 80 512 0 1
```

To build a tree-capable text decoder for another conventional causal language model
(Qwen3, Llama, or Mistral) and benchmark it, use the reproducible wrappers. The
builder writes `fp32`, `int8`, and mixed-body `q4` graphs plus the tokenizer into one
decoder directory. The tree bench runs every registered drafter against each graph that
exists, always comparing a speculative method with that graph's own baseline.
The build gate also compares multi-token causal logits with incremental KV-cache
decoding. A quantized graph that changes next-token choices with sequence length is
annotated `speculative_compatible: false`; benchmarks load it with a warning and record
the resulting divergences as diagnostics rather than rejecting it.

```bash
./scripts/build_decoder.sh Qwen/Qwen3-0.6B
./scripts/bench_tree.sh ~/.cache/dejavuu/Qwen-Qwen3-0.6B
```

For a direct text run, `--model-path` selects that decoder directory and `--variant`
selects the graph. `fp32` is available only from a built directory; the published
Gemma snapshot still supplies the legacy `q4` and `int8` defaults.

```bash
uv run python -m dejavuu.eval.specbench \
    --model-path ~/.cache/dejavuu/Qwen-Qwen3-0.6B --variant fp32 \
    --methods baseline,pld,pld_plus,adapld --per-category 20 --tree --width 2 \
    --out results/qwen-specbench
```

Use the unified selector for benchmark datasets. `speedbench` defaults to NVIDIA's
diverse qualitative split; its throughput-named splits are context-length workloads in
this single-request ONNX Runtime harness, not batched-server throughput claims.
The default `conversation` protocol preserves every turn and applies the model chat
template. Use `--protocol first-turn-workload` only to compare with older first-turn
runs; `--protocol official` rejects a throughput split when the runner cannot satisfy
its batched-serving requirements.

```bash
uv run --extra bench python -m dejavuu.eval.bench --dataset speedbench \
    --model-path ~/.cache/dejavuu/Qwen-Qwen3-0.6B --variant q4 --per-category 20 \
    --threads 4 --budget 8 --tree --width 2 --out results/qwen-speed
uv run --extra bench python -m dejavuu.eval.bench --dataset mmspec \
    --per-category 10 --threads 4 --budget 8 --tree --width 2 \
    --out results/smol-mmspec
```

Every run is an immutable directory containing `manifest.json`, `summary.csv`,
`car.csv`, `measurements.jsonl`, `responses.jsonl`, `divergences.jsonl`, and the runner
log. The manifest pins the model and dataset provenance, execution settings, provider,
software versions, and cost-tier definitions. `measurements.jsonl` contains phase-level
telemetry; generated text and token IDs live in `responses.jsonl`; every token
divergence from the baseline is recorded in `divergences.jsonl`. Divergence never fails
the run: the command exits zero and the bundle is `valid` (or `valid_with_divergences`).

Results land under a timestamped `results/bench-*` directory. The four knobs are positional,
`scripts/bench_all.sh <K> <IMG> <THREADS> <TREE>`:

- `K` is prompts per topic (80 covers all of Spec-Bench; MMSpec saturates at 10 or more).
- `IMG` is the MMSpec image longest-edge in pixels. Smaller means fewer, smaller tiles
  and a faster run (512 is one tile, 256 downscales further, `0` is full resolution).
  The text bench ignores it.
- `THREADS` is ORT intra-op threads on CPU (`0` is the ORT default, `4` pins to 4 cores).
- `TREE` is `0` for chain verify (default) or `1` for tree-based verification.

```bash
./scripts/bench_all.sh 80 512 4          # 80/topic, 4 CPU threads, chain verify
./scripts/bench_all.sh 80 512 0 1        # tree-based verification (needs the step-3 decoder)
./scripts/bench_all.sh 80 256 4 1        # all knobs: 256px images, 4 threads, tree ON
./scripts/bench_all.sh 80 256 4 0        # same but chain (the with/without-tree pair)
```

Dataset and model selection are independent. To run both datasets through SmolVLM,
select its adapter explicitly for text Spec-Bench; drop `--tree` for chain:

```bash
uv run --extra vlm python -m dejavuu.eval.bench --dataset specbench \
    --model-kind smolvlm_onnx --methods baseline,pld,pld_plus,adapld \
    --per-category 80 --threads 4 --out results/smol-specbench                   # chain
uv run --extra vlm python -m dejavuu.eval.bench --dataset mmspec \
    --methods baseline,pld,pld_plus,adapld --per-category 80 \
    --image-size 256 --threads 4 --tree --width 2 --out results/smol-mmspec      # tree
```

`--provider cpu` selects `CPUExecutionProvider`, and `--threads N` sets its
`intra_op_num_threads`. `--provider cuda` fails when CUDA is unavailable unless
`--allow-provider-fallback` is explicit. `--tree` requires a tree-capable decoder and
fails instead of silently changing the requested verification mode.

## How results are reported

`eval/harness.py` renders a per-topic table and CSV. `tok/s` and speedups are hot-path
decode-only. Model load, prompt/image preparation, KV prefill, and drafter setup are
reported separately as online-once costs. The hot path partitions into
`draft + verify + learn + overhead`. Workload dispersion over per-case means and
within-case repetition variance are separate columns. Each text and vision method is
compared against its own backend baseline; token divergence is reported as a diagnostic
and never fails the run. Model-free chain/tree conformance stays bit-exact regardless —
it guards the verifier's acceptance and KV invariants.

## Layout

```
dejavuu/
  api.py            DejaVu.from_pretrained(...).generate(...)  (drop-in)
  cli.py            the `dejavuu` single-generation command
  core/             model-agnostic spec-decode: verifier (contract), engine,
                    tree, sampling
  decoders/         Verifier implementations: ort, text (Model), vlm (VLM)
  drafters/         the method zoo (base, suffix_index, one file per method)
  eval/             unified config, dataset/model adapters, runner, metrics, run bundles
  tools/            build_tree_decoder (tree+hidden, quantized), build_vlm_decoder,
                    build_specbench_corpus, specbench_entropy, eval_tree
scripts/bench_all.sh  one-command specbench + mmspec sweep over all methods
```

## Development

`AGENTS.md` is the working agreement: the drafter contract, the cost tiers, and how new
methods stay lossless under both chain and tree verification.

```bash
uv run prek run --all-files   # the full gate: ruff, pyrefly, deptry, tests, coverage
uv run pytest                 # offline unit tests only (add -m model for the Gemma run)
```
