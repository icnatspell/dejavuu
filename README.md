# dejavuu

Training-free speculative decoding for [ONNX Runtime](https://onnxruntime.ai/) and
Hugging Face PyTorch models.

[![CI](https://github.com/icnatspell/dejavuu/actions/workflows/ci.yml/badge.svg)](https://github.com/icnatspell/dejavuu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/icnatspell/dejavuu/blob/main/LICENSE)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)

Speculative decoding drafts several tokens cheaply, then checks them in one model
forward pass. The target model's accept rule keeps only the tokens it would have
produced anyway, so a wrong guess wastes one pass and never changes the output. The
speedup comes from replacing many single-token passes with fewer multi-token ones.

`dejavuu` is model-free in the drafting sense: it drops the small auxiliary draft model
that speculative decoding usually needs. Its drafters copy their guesses straight from
text the target model has already seen—the prompt, the generation so far, or a fixed
corpus. A guess is an index lookup, not a second network. Every drafter works on raw
token ids, so one instance drives both a text LLM and a vision-language model through
the same verifier.

- No draft model to load, train, or keep in memory.
- Strictly lossless on the text path, checked bit-for-bit in CI.
- One model spans both modalities: SmolVLM2 runs the text and the vision benchmark.
- Chain and tree verification both work for every method.

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

## Results

SmolVLM2 (int4) on Spec-Bench, 480 prompts, CPU. Speedup is decode-only throughput
over the plain autoregressive baseline. Retrieval pays off most where the output
repeats the input, so the best method varies by task:

| task | best method | speedup | token match |
|---|---|---:|---:|
| retrieval-augmented generation | `suffix_decoding` | 1.66x | 79% |
| multi-turn conversation | `sam_decoding` | 1.32x | 55% |
| summarization | `asam` | 1.31x | 73% |
| mathematical reasoning | `anpd` | 1.29x | 49% |
| question answering | `anpd` | 1.15x | 48% |
| translation | `anpd` | 2.00x | 27% |

The strongest methods average about 1.2x across the six tasks. The clean wins are RAG
and summarization, where the drafted output tracks the baseline closely. Translation
and QA decode faster but diverge more, because the quantized SmolVLM decoder is not
length-invariant: its speculative output is *near*-lossless (a token-match percentage
against the baseline), not bit-exact. The Gemma text path stays strictly lossless, and
the bit-exactness unit tests guard it. Reproduce these numbers in
[Reproduce the benchmark](#reproduce-the-benchmark).

## Methods

Every drafter runs on raw token ids, so the same instance drives both the LLM and the
VLM. REST, SuffixDecoding, SAM, and ASAM share one reusable token-only `SuffixIndex`.

| method | idea |
|---|---|
| `baseline` | plain autoregressive, no drafter |
| `pld` | prompt-lookup: longest suffix match within the context |
| `pld_plus` | `pld` plus hidden-state reranking of matches (needs a hidden-emitting decoder; else falls back to `pld`) |
| `adapld` | `pld_plus` with a semantic fallback, plus a branched tree under `--tree` |
| `anpd` | adaptive n-gram draft length |
| `lookahead` | multi-candidate n-gram pool |
| `logit_spec` | verifier-logit candidates extended by n-gram retrieval |
| `token_recycling` | tree drafts from the verifier's own logits |
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
rm -f results/specbench.* results/mmspec.*
nohup ./scripts/bench_all.sh 80 512 > results/run.out 2>&1 &
tail -f results/run.out

# 7. same sweep with tree-based verification instead of chain (4th arg = 1)
./scripts/bench_all.sh 80 512 0 1
```

Results land in `results/{specbench,mmspec}.{csv,log}`. The four knobs are positional,
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

To call a bench directly (both datasets go through the one SmolVLM harness), drop
`--tree` for chain, and set `--width` for max children per node under a tree:

```bash
uv run --extra vlm python -m dejavuu.eval.mmspec --dataset specbench \
    --methods baseline,pld,pld_plus,adapld --per-category 80 --threads 4          # chain
uv run --extra vlm python -m dejavuu.eval.mmspec --dataset mmspec \
    --methods baseline,pld,pld_plus,adapld --per-category 80 \
    --image-size 256 --threads 4 --tree --width 2                                 # tree
```

`--provider cpu` selects `CPUExecutionProvider`, and `--threads N` sets its
`intra_op_num_threads`. `--tree` needs the tree+hidden decoder from step 3, and warns
and falls back to chain without it.

## How results are reported

`eval/harness.py` renders a per-topic table and CSV. `tok/s` and the speedups are
decode-only, since prefill is a one-time prompt tax that has nothing to do with
speculative decoding and would dilute the decode-loop speedup (the table reports it in
its own column). Every per-prompt column is `mean ± std`, and the per-step time splits
as `total = prefill + draft + verify + overhead`. A strict exactness gate, or a
token-match percentage for the VLM, guards every method against its baseline.

## Layout

```
dejavuu/
  api.py            DejaVu.from_pretrained(...).generate(...)  (drop-in)
  cli.py            the `dejavuu` single-generation command
  core/             model-agnostic spec-decode: verifier (contract), engine,
                    tree, sampling
  decoders/         Verifier implementations: ort, text (Model), vlm (VLM)
  drafters/         the method zoo (base, suffix_index, one file per method)
  eval/             benchmark harnesses: harness (shared), specbench, mmspec
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
