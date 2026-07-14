"""Raw onnxruntime text decoder, as a Verifier. Just a snapshot + OrtDecoder.

All graph specifics (layer/head counts, KV naming, position_ids, tree support) are
auto-derived by OrtDecoder from the ONNX I/O -- nothing here is Gemma-specific beyond
the default repo, so any conventional causal-LM export works by pointing `root` at it.
On CPU, KV is plain numpy, sliced on accept -- the accept-slice is 0.08% of the forward
(0.02 ms vs 24 ms, 270m/q4/cpu), and a prototype that kept KV as bound OrtValues across
steps recovered only a flat ~2 ms there, because the "numpy round-trip" is a cheap host
memcpy -- the numpy boundary is not the CPU bottleneck.

On **GPU that round-trip is a PCIe copy** and it dominates: profiling fp32/qwen3-0.6b/CUDA,
the host<->device KV copy is ~60 ms of a 79 ms step at 1024 tokens and scales with context,
while the compute floor is launch-bound and flat at ~18 ms. So for `provider=cuda` the
decoder (`OrtDecoder(device_kv=True)`) keeps KV device-resident via io-binding and trims it
on-device (issue #46): measured 1.6x @ 256 tok rising to 5.2x @ 2048 tok, decode step flat
at ~25 ms. CPU is unchanged (device_kv stays off; the numpy path is byte-identical).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.decoders.ort import OrtDecoder, make_session

REPO = "onnx-community/gemma-3-270m-ONNX"
REVISION = "f432e832a60ad22394057c85d45ed5007da3f571"
ONNX_FILES = {
    "fp32": "onnx/model_fp32.onnx",
    "q4": "onnx/model_q4.onnx",
    "int8": "onnx/model_int8.onnx",
}


def download(variant: str = "q4") -> Path:
    """Fetch tokenizer + one onnx variant; return the snapshot dir."""
    return Path(
        snapshot_download(
            REPO,
            revision=REVISION,
            allow_patterns=["*.json", "tokenizer*", ONNX_FILES[variant]],
        )
    )


def resolve_graph_path(root: Path, variant: str) -> Path:
    """Resolve a variant by manifest role, with the legacy layout as fallback."""
    root = Path(root)
    manifest = root / "manifest.json"
    if manifest.exists():
        provenance = json.loads(manifest.read_text()).get("provenance", {})
        entry = provenance.get("variants", {}).get(variant)
        if isinstance(entry, dict) and entry.get("file"):
            return root / entry["file"]
    return root / ONNX_FILES[variant]


@dataclass
class Model(Verifier):
    root: Path
    variant: str = "q4"
    provider: str = "cpu"
    threads: int = 0
    allow_provider_fallback: bool = False

    @cached_property
    def _dec(self) -> OrtDecoder:
        path = resolve_graph_path(self.root, self.variant)
        session = make_session(
            path,
            self.provider,
            self.threads,
            allow_provider_fallback=self.allow_provider_fallback,
        )
        # On GPU the host<->device KV copy dominates the verify forward at real sequence
        # lengths (issue #46), so keep KV device-resident there. CPU is unchanged: the
        # numpy round-trip is a cheap host memcpy and a prior prototype recovered nothing.
        device_kv = "CUDAExecutionProvider" in session.get_providers()
        return OrtDecoder(session, device_kv=device_kv)

    @property
    def supports_tree(self) -> bool:
        return self._dec.supports_tree

    def empty_kv(self) -> KVCache:
        return self._dec.empty_kv()

    def rollback_kv(self, kv: KVCache, committed: int) -> KVCache:
        # Delegate so device_kv sessions trim on-device; the decoder falls back to the
        # numpy prefix-trim otherwise (identical to the Verifier default).
        return self._dec.rollback_kv(kv, committed)

    def gather_kv(self, kv: KVCache, committed: int, path: list[int]) -> KVCache:
        return self._dec.gather_kv(kv, committed, path)

    def forward(
        self,
        token_ids: list[int],
        past: KVCache,
        past_len: int,
        position_ids: np.ndarray | None = None,
        attn_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, KVCache, np.ndarray | None]:
        return self._dec.run(token_ids, past, past_len, position_ids, attn_bias)
