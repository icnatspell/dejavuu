"""Build a GQA decoder that handles seq>1 with past for a vision-language model, so
chain speculative decoding works (many published VLM text decoders are locked to
seq_len=1 with past, which is baseline-only).

Two steps: (1) extract the causal-LM text backbone from the multimodal checkpoint as a
standalone `*ForCausalLM`; (2) run the onnxruntime-genai model builder with
exclude_embeds=true (so it takes inputs_embeds, matching our vision splice). The output
runs under raw onnxruntime in vlm.py; genai is a build-time tool only.

Defaults to SmolVLM2. Any other VLM works if (a) its text backbone is one of the
architectures the genai model builder supports (Llama, Qwen2, Gemma, Mistral, Phi, ...)
and (b) its text tower is reachable by one of the attribute paths in `_TEXT_PATHS`.
A new model is worth a smoke check: confirm the built decoder is token-identical to the
checkpoint's own HF text path before trusting it.

Run (needs the vlm extra + the build-only tools):
    uv run --with onnxruntime-genai --with onnx_ir --index-strategy unsafe-best-match \
        python -m dejavuu.tools.build_vlm_decoder [--repo REPO] [--out model.onnx]
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from dejavuu.decoders.vlm import GENAI_DECODER, REPO

# Where the text tower hangs off a multimodal checkpoint. Layout is architecture-
# specific, so probe the common paths; first match wins.
# ponytail: a lookup list, not model detection. Add a path when a new VLM doesn't fit.
_TEXT_PATHS = (
    "model.text_model",
    "model.language_model",
    "text_model",
    "language_model",
    "model.model.language_model",
)


def _text_backbone(vlm: torch.nn.Module) -> torch.nn.Module:
    """Locate the causal-LM text backbone inside a multimodal model."""
    for path in _TEXT_PATHS:
        obj: object = vlm
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.Module):
            return obj
    raise SystemExit(
        f"could not find the text backbone in {type(vlm).__name__}; "
        "add its attribute path to _TEXT_PATHS"
    )


def main() -> None:
    p = argparse.ArgumentParser("dejavuu.tools.build_vlm_decoder")
    p.add_argument("--repo", default=REPO, help="HF VLM checkpoint (default: SmolVLM2)")
    p.add_argument(
        "--out",
        type=Path,
        default=GENAI_DECODER,
        help="output model.onnx path (its sibling .onnx.data is written too)",
    )
    p.add_argument("--precision", default="int4", help="genai builder precision")
    p.add_argument("--device", default="cpu", help="genai builder exec provider")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, built = Path(tmp) / "text", Path(tmp) / "onnx"

        vlm = AutoModelForImageTextToText.from_pretrained(args.repo, dtype=torch.float32)
        text = _text_backbone(vlm)
        # Rebuild the text tower as a standalone causal LM: get_text_config picks the
        # right architecture (not just Llama) and from_config builds the matching
        # *ForCausalLM. The genai builder needs a plain causal-LM checkpoint as input.
        lm = AutoModelForCausalLM.from_config(vlm.config.get_text_config())
        # strict=False tolerates non-persistent buffers (rotary inv_freq); anything else
        # not transferred means the backbones drifted -> the decoder would ship garbage
        # weights, so surface it instead of building silently on top of it.
        info = lm.model.load_state_dict(text.state_dict(), strict=False)
        drift = [k for k in info.missing_keys if "inv_freq" not in k] + list(info.unexpected_keys)
        if drift:
            logger.warning("backbone weight drift -- keys not transferred: {}", drift)
        head = getattr(vlm, "lm_head", None) or getattr(text, "lm_head", None)
        if head is not None:
            lm.lm_head.load_state_dict(head.state_dict())
        lm.tie_weights()  # no-op unless the config ties the head to the embeddings
        lm.save_pretrained(ckpt)
        # Drop the source (fast) tokenizer into the input dir so the genai builder
        # reuses tokenizer.json instead of a slow->fast convert (which needs
        # sentencepiece/tiktoken and otherwise fails the build at the config step).
        AutoTokenizer.from_pretrained(args.repo).save_pretrained(ckpt)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "onnxruntime_genai.models.builder",
                "-i",
                str(ckpt),
                "-o",
                str(built),
                "-p",
                args.precision,
                "-e",
                args.device,
                "-c",
                str(Path(tmp) / "cache"),
                "--extra_options",
                "exclude_embeds=true",
            ],
            check=True,
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        for f in ("model.onnx", "model.onnx.data"):
            shutil.copy(built / f, args.out.parent / f)
    logger.info("decoder -> {}", args.out)


if __name__ == "__main__":
    main()
