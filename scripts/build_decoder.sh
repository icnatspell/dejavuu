#!/usr/bin/env bash
# Build fp32, int8, and q4 tree+hidden decoder graphs for a HF causal LM.
# Usage: ./scripts/build_decoder.sh Qwen/Qwen3-0.6B
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <hf-model-id>" >&2
    exit 2
fi

MODEL=$1
NAME=${MODEL//\//-}
OUT=${DEJAVUU_DECODER_DIR:-"$HOME/.cache/dejavuu/$NAME"}

uv run --extra vlm --with onnx_ir --index-strategy unsafe-best-match \
    python -m dejavuu.tools.build_decoder --model "$MODEL" --out "$OUT" --quant both

echo "decoder: $OUT"
