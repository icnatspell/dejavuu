"""SmolVLM2 as a Verifier: same spec-decode engine, multimodal prefill.

Three ONNX graphs:
  embed_tokens:   input_ids -> inputs_embeds[.,960]
  vision_encoder: pixel_values -> image_features[tiles, 64, 960]
  decoder:        inputs_embeds + position_ids + past_kv -> logits + present

The decode loop is identical to the text path -- it just embeds tokens first.
Only prefill differs: vision features are scattered into the prompt embeds at the
image-token slots. Requires the optional `vlm` extra (torch, for preprocessing
only; inference stays pure onnxruntime).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download

from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.decoders.ort import OrtDecoder, make_session

REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GRAPHS = ("embed_tokens", "vision_encoder", "decoder_model_merged")

# onnxruntime-genai-built decoder (GQA, handles seq>1 with past -> chain
# spec-decode works). Built by tools/build_vlm_decoder.py. The published
# transformers.js decoder is locked to seq_len=1 with past (baseline only).
GENAI_DECODER = Path.home() / ".cache" / "dejavuu" / "smolvlm2_decoder_genai" / "model.onnx"
# Tree-capable + hidden-state-emitting decoder (torch.onnx export, inputs_embeds).
# Built by tools/build_tree_decoder.py; preferred over GENAI when present since it
# is strictly more capable (chain, tree, and hidden states for PLD+/AdaPLD).
VLM_TREE_DECODER = (
    Path.home() / ".cache" / "dejavuu" / "smolvlm2_decoder_tree_embeds" / "model.onnx"
)


def download(variant: str = "q4") -> Path:
    files = [f"onnx/{g}_{variant}.onnx" for g in GRAPHS]
    return snapshot_download(REPO, allow_patterns=["*.json", "*.txt", *files])


@dataclass
class VLM(Verifier):
    root: Path
    variant: str = "q4"
    provider: str = "cpu"
    threads: int = 0
    image_token_id: int = 49190
    decoder_path: Path | None = None  # genai GQA decoder; None -> stock (baseline only)
    _prefill_embeds: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.decoder_path is None:
            # tree + hidden states, smallest quant first (int4 lm-head-fp32 > int8 > fp32)
            for name in ("model_int4.onnx", "model_int8.onnx", "model.onnx"):
                cand = VLM_TREE_DECODER.with_name(name)
                if cand.exists():
                    self.decoder_path = cand
                    break
            else:
                if GENAI_DECODER.exists():
                    self.decoder_path = GENAI_DECODER  # chain-only fallback

    def _sess(self, graph: str) -> ort.InferenceSession:
        path = Path(self.root) / "onnx" / f"{graph}_{self.variant}.onnx"
        return make_session(path, self.provider, self.threads)

    @cached_property
    def _embed(self):
        return self._sess("embed_tokens")

    @cached_property
    def _vision(self):
        return self._sess("vision_encoder")

    @cached_property
    def _dec(self) -> OrtDecoder:
        path = self.decoder_path or (
            Path(self.root) / "onnx" / f"decoder_model_merged_{self.variant}.onnx"
        )
        return OrtDecoder(make_session(path, self.provider, self.threads))

    @property
    def supports_tree(self) -> bool:
        return self._dec.supports_tree

    @property
    def is_vlm(self) -> bool:
        return True

    def empty_kv(self) -> KVCache:
        return self._dec.empty_kv()

    def embed(self, token_ids: list[int]) -> np.ndarray:
        out = self._embed.run(None, {"input_ids": np.asarray([token_ids], dtype=np.int64)})
        return out[0][0]  # [N, 960]

    def forward(
        self,
        token_ids: list[int],
        past: KVCache,
        past_len: int,
        position_ids: np.ndarray | None = None,
        attn_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, KVCache, np.ndarray | None]:
        # The genai GQA decoder is causal-only (attn_bias raises in OrtDecoder.run);
        # the tree-embeds decoder honours the 4D mask and also emits hidden states.
        embeds = self.embed(token_ids)
        return self._dec.run(embeds, past, past_len, position_ids, attn_bias)

    def prepare(self, processor_out: dict) -> list[int]:
        """Splice vision features into the prompt embeds; stash for prefill.
        `processor_out` is the dict from the HF processor (numpy tensors)."""
        ids = processor_out["input_ids"][0]
        embeds = self.embed(ids.tolist())  # [seq, 960]
        feats = self._vision.run(
            None,
            {
                "pixel_values": processor_out["pixel_values"].astype(np.float32),
                "pixel_attention_mask": processor_out["pixel_attention_mask"].astype(bool),
            },
        )[0]  # [tiles, 64, 960]
        feats = feats.reshape(-1, feats.shape[-1])
        slots = ids == self.image_token_id
        assert slots.sum() == feats.shape[0], (
            f"{int(slots.sum())} image slots != {feats.shape[0]} vision features"
        )
        embeds[slots] = feats
        self._prefill_embeds = embeds
        return ids.tolist()

    def prefill(self, prompt_ids: list[int]) -> tuple[KVCache, int]:
        embeds = self._prefill_embeds
        self._prefill_embeds = None
        if embeds is None:  # text-only prompt, no image prepared
            return super().prefill(prompt_ids)
        _, past, _ = self._dec.run(embeds[:-1], self.empty_kv(), 0)
        return past, len(prompt_ids) - 1

    def prefill_seeded(self, prompt_ids: list[int]) -> tuple[KVCache, int, np.ndarray]:
        """Seeded-root prefill, preserving prepared image embeddings through the
        complete prompt so the final target logits can select the first root."""
        embeds = self._prefill_embeds
        self._prefill_embeds = None
        if embeds is None:
            return super().prefill_seeded(prompt_ids)
        logits, past, _ = self._dec.run(embeds, self.empty_kv(), 0)
        return past, len(prompt_ids), logits[-1]
