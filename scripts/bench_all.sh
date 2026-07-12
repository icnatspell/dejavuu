#!/usr/bin/env bash
# Launch both benches on ONE model (SmolVLM2): text Spec-Bench (--dataset specbench)
# and vision MMSpec (--dataset mmspec), over every method, seeded with the static
# datastore. Results -> results/{specbench,mmspec}.{csv,log}.
#   ./scripts/bench_all.sh                # 20 prompts/topic, 512px images, chain verify
#   ./scripts/bench_all.sh 40             # 40 prompts/topic
#   ./scripts/bench_all.sh 80 512 4       # cap ORT to 4 intra-op threads (CPUExecutionProvider)
#   ./scripts/bench_all.sh 80 512 0 1     # enable tree-based verification (default is chain)
set -euo pipefail

METHODS=baseline,pld,pld_plus,adapld,anpd,lookahead,token_recycling,rest,suffix_decoding,sam_decoding,asd,asd_verify,asam,asam_verify
K=${1:-20}                       # prompts per category
IMG=${2:-512}                    # mmspec: cap longest_edge px (512 = 1 tile, fastest; 0 = full res)
THREADS=${3:-0}                  # ORT intra-op threads on CPUExecutionProvider (0 = ORT default)
TREE=${4:-0}                     # 0 = chain verify (default), 1 = tree-based verification
STORE=data/specbench_corpus.txt
mkdir -p results

# tree needs the tree+hidden decoder (tools/build_tree_decoder.py); pld_plus/adapld also
# need its hidden states -- without it they degrade to plain PLD. See README.
TREE_FLAG=()
[ "$TREE" = "1" ] && TREE_FLAG=(--tree --width 2)

[ -f "$STORE" ] || uv run python -m dejavuu.tools.build_specbench_corpus --out "$STORE"

# text Spec-Bench on SmolVLM
uv run --extra vlm python -m dejavuu.eval.mmspec --dataset specbench \
    --methods "$METHODS" --per-category "$K" --datastore "$STORE" --threads "$THREADS" \
    "${TREE_FLAG[@]}" \
    --csv results/specbench.csv --log results/specbench.log

# vision MMSpec on SmolVLM
uv run --extra vlm python -m dejavuu.eval.mmspec --dataset mmspec \
    --methods "$METHODS" --per-category "$K" --datastore "$STORE" --threads "$THREADS" \
    --image-size "$IMG" "${TREE_FLAG[@]}" \
    --csv results/mmspec.csv --log results/mmspec.log
