"""Raw onnxruntime text decoder, as a Verifier. Just a snapshot + OrtDecoder.

All graph specifics (layer/head counts, KV naming, position_ids, tree support) are
auto-derived by OrtDecoder from the ONNX I/O -- nothing here is Gemma-specific beyond
the default repo, so any conventional causal-LM export works by pointing `root` at it.
KV is plain numpy, sliced on accept -- ponytail: the accept-slice is 0.08% of the
forward (0.02 ms vs 24 ms, 270m/q4/cpu). The forward *does* scale with context
(~0.4 ms/MB of KV: 12.5 ms @ 64 tok -> 42.8 ms @ 2048 tok), but that cost is inside
the graph -- the past->present concat + attention over the growing cache. A prototype
that keeps KV as bound OrtValues across steps (no numpy round-trip) recovered only a
flat ~2 ms (1.06-1.13x, not scaling), so the numpy boundary is not the bottleneck.
Removing the in-graph concat copy needs a past_present_share_buffer (genai-built)
export, and even then the attention read over the KV is irreducible. Not worth it on
270m/q4/cpu; revisit on a bigger model / GPU where the copy/compute ratio shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.decoders.ort import OrtDecoder, make_session

REPO = "onnx-community/gemma-3-270m-ONNX"
ONNX_FILES = {"q4": "onnx/model_q4.onnx", "int8": "onnx/model_int8.onnx"}


def download(variant: str = "q4") -> Path:
    """Fetch tokenizer + one onnx variant; return the snapshot dir."""
    return Path(
        snapshot_download(
            REPO,
            allow_patterns=["*.json", "tokenizer*", ONNX_FILES[variant]],
        )
    )


@dataclass
class Model(Verifier):
    root: Path
    variant: str = "q4"
    provider: str = "cpu"
    threads: int = 0

    @cached_property
    def _dec(self) -> OrtDecoder:
        path = Path(self.root) / ONNX_FILES[self.variant]
        return OrtDecoder(make_session(path, self.provider, self.threads))

    @property
    def supports_tree(self) -> bool:
        return self._dec.supports_tree

    def empty_kv(self) -> KVCache:
        return self._dec.empty_kv()

    def forward(
        self,
        token_ids: list[int],
        past: KVCache,
        past_len: int,
        position_ids: np.ndarray | None = None,
        attn_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, KVCache, np.ndarray | None]:
        return self._dec.run(token_ids, past, past_len, position_ids, attn_bias)
