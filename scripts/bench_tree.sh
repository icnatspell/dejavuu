#!/usr/bin/env bash
# Run every drafter through tree verification for each graph available in a decoder dir.
# Usage: ./scripts/bench_tree.sh ~/.cache/dejavuu/qwen-qwen3-0.6b
# Optional environment: K=80 BUDGET=8 WIDTH=2 THREADS=0 PROVIDER=cpu MAX_NEW=128.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <decoder-dir>" >&2
    exit 2
fi

DECODER=$1
K=${K:-20}
BUDGET=${BUDGET:-8}
WIDTH=${WIDTH:-2}
THREADS=${THREADS:-0}
PROVIDER=${PROVIDER:-cpu}
MAX_NEW=${MAX_NEW:-128}
STORE=${STORE:-data/specbench_corpus.txt}
METHODS=baseline,pld,copyspec,pld_plus,adapld,anpd,cacheback,lookahead,logit_spec,ngram_trie,token_recycling,rest,suffix_decoding,sam_decoding,stand,asam,asam_verify,asd,asd_verify

mkdir -p results
[ -f "$STORE" ] || uv run python -m dejavuu.tools.build_specbench_corpus --out "$STORE"

for VARIANT in fp32 int8 q4; do
    GRAPH="$DECODER/onnx/model_$VARIANT.onnx"
    [ -f "$GRAPH" ] || continue
    STEM=$(basename "$DECODER")
    uv run python -m dejavuu.eval.specbench \
        --model-path "$DECODER" --variant "$VARIANT" --provider "$PROVIDER" \
        --methods "$METHODS" --per-category "$K" --max-new "$MAX_NEW" \
        --budget "$BUDGET" --tree --width "$WIDTH" --threads "$THREADS" \
        --datastore "$STORE" \
        --csv "results/specbench_${STEM}_${VARIANT}_tree.csv" \
        --log "results/specbench_${STEM}_${VARIANT}_tree.log"
done
