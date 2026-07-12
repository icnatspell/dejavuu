"""Drop-in entry point: `DejaVu.from_pretrained(repo_or_dir).generate(...)`.

Same shape as optimum's ORTModelForCausalLM, so it swaps in for plain ONNX
generation and adds speculative decoding for free -- greedy stays bit-exact, sampling
stays distribution-exact. The decoder self-describes (OrtDecoder); the only thing to
detect is text vs VLM (a VLM snapshot ships a vision_encoder graph)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from dejavuu.config import GenerationConfig, ModelConfig
from dejavuu.core.engine import GenResult, generate
from dejavuu.core.sampling import Sampler
from dejavuu.core.verifier import Verifier
from dejavuu.drafters import make_drafter, require_method


def load(root: str | Path, variant: str = "q4", provider: str = "cpu") -> Verifier:
    root = Path(root)
    if list(root.glob("onnx/vision_encoder*.onnx")):
        from dejavuu.decoders.vlm import VLM  # lazy: VLM pulls the optional `vlm` extra (torch)

        return VLM(root, variant, provider)
    from dejavuu.decoders.text import Model

    return Model(root, variant, provider)


def _fetch(repo_or_dir: str | Path, variant: str) -> Path:
    p = Path(repo_or_dir)
    if p.is_dir():
        return p  # already a local snapshot
    return Path(
        snapshot_download(
            str(repo_or_dir),
            allow_patterns=["*.json", "tokenizer*", f"onnx/*{variant}*.onnx"],
        )
    )


@dataclass
class DejaVu:
    """Drop-in: `from_pretrained` then `generate`. Text or VLM -- pass `image=` to
    generate against a VLM; the model type is auto-detected at load."""

    model: Verifier
    proc: object  # AutoTokenizer (text) or AutoProcessor (VLM)
    method: str  # a registered drafter name, or "baseline"
    is_vlm: bool = False

    @property
    def _tok(self):  # the tokenizer either way, for encode/decode/eos
        return self.proc.tokenizer if self.is_vlm else self.proc

    @classmethod
    def from_pretrained(
        cls,
        repo_or_dir: str | Path = "onnx-community/gemma-3-270m-ONNX",
        method: str = "token_recycling",
        variant: str = "q4",
        provider: str = "cpu",
        backend: str = "ort",
        device: str | None = None,
        dtype: str | None = None,
        attn_implementation: str = "eager",
    ) -> DejaVu:
        """`backend="ort"` (default) runs a local ONNX export; `backend="hf"` runs any
        transformers `AutoModelForCausalLM` with no export -- pass an explicit
        `device` ("cuda"/"cpu") and optional `dtype` ("bfloat16", ...).
        `attn_implementation` is the HF attention kernel ("eager" default, "sdpa" for
        GPU perf). `variant`/`provider` apply to the ORT backend only."""
        require_method(method)  # fail fast on a typo, before any model download
        cfg = ModelConfig(
            backend=backend,
            variant=variant,
            provider=provider,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        if cfg.backend == "hf":
            from transformers import AutoTokenizer

            from dejavuu.decoders.hf import HFBackend

            assert cfg.device is not None  # ModelConfig validates this for the hf backend
            model: Verifier = HFBackend(
                str(repo_or_dir),
                device=cfg.device,
                dtype=cfg.dtype,
                attn_implementation=cfg.attn_implementation,
            )
            proc = AutoTokenizer.from_pretrained(repo_or_dir)
            return cls(model, proc, method, is_vlm=False)

        root = _fetch(repo_or_dir, cfg.variant)
        model = load(root, cfg.variant, cfg.provider)
        is_vlm = model.is_vlm
        if is_vlm:
            from transformers import AutoProcessor

            proc = AutoProcessor.from_pretrained(root)
        else:
            from transformers import AutoTokenizer

            proc = AutoTokenizer.from_pretrained(root)
        return cls(model, proc, method, is_vlm)

    def _encode(self, prompt: str, image) -> list[int]:
        """Text -> ids; VLM -> chat-template + splice vision (via model.prepare)."""
        if not self.is_vlm:
            return self._tok(prompt)["input_ids"]
        content = ([{"type": "image"}] if image is not None else []) + [
            {"type": "text", "text": prompt}
        ]
        text = self.proc.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True
        )
        if image is None:
            return self.proc(text=text, return_tensors="np")["input_ids"][0].tolist()
        from PIL import Image  # lazy: only the VLM image path needs it

        img = image if not isinstance(image, str) else Image.open(image).convert("RGB")
        proc_out = dict(self.proc(text=text, images=[img], return_tensors="np"))
        return self.model.prepare(proc_out)

    def generate(
        self,
        prompt: str,
        image=None,
        *,
        max_new: int = 64,
        budget: int = 8,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        tree: bool = False,
        width: int = 2,
        stream: bool = False,
        color: bool = True,
    ) -> str:
        """Returns the completion. `image` (path or PIL) drives a VLM. `tree=True` uses
        tree verification when the backend supports it (else it falls back to chain).
        `stream=True` prints tokens live; with `color`, accepted draft guesses are green
        and bonus/correction tokens default-colored -- a visual of where speculation pays.
        Decode parameters are validated by `GenerationConfig`."""
        cfg = GenerationConfig(
            method=self.method,
            max_new=max_new,
            budget=budget,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            tree=tree,
            width=width,
        )
        ids = self._encode(prompt, image)
        sampler = Sampler(cfg.temperature, cfg.top_p, cfg.seed) if cfg.temperature > 0 else None

        def emit(tok: int, accepted: bool) -> None:
            piece = self._tok.decode([tok])
            if color and accepted:
                piece = f"\033[32m{piece}\033[0m"  # green = accepted speculation
            print(piece, end="", flush=True)

        res: GenResult = generate(
            self.model,
            ids,
            cfg.max_new,
            make_drafter(cfg.method),
            cfg.budget,
            self._tok.eos_token_id,
            tree=cfg.tree,
            width=cfg.width,
            sampler=sampler,
            on_emit=emit if stream else None,
        )
        if stream:
            print()
        return self._tok.decode(res.tokens)
