"""Model adapters used by the benchmark runner.

The adapter owns prompt rendering and online-once preparation. The returned verifier
still satisfies the core token-only contract, so drafters never see model internals.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dejavuu.core.verifier import Verifier
from dejavuu.eval.config import ModelSpec
from dejavuu.eval.datasets import ConversationCase, Turn
from dejavuu.tools.artifact import verify_manifest

ConversationHistory = list[dict[str, object]]


def _token_ids(value: object) -> list[int]:
    """Normalize tokenizer output across Transformers list/mapping/tensor APIs."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]  # type: ignore[union-attr]


@dataclass(frozen=True)
class ModelIdentity:
    kind: str
    source: str
    requested_provider: str
    actual_providers: tuple[str, ...]


@dataclass(frozen=True)
class PreparedInvocation:
    verifier: Verifier
    prompt_ids: list[int]
    eos_token_id: int | None


class BenchmarkModel(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    @property
    def artifacts(self) -> tuple[dict[str, object], ...]: ...

    def prepare(
        self, case: ConversationCase, turn: Turn, history: ConversationHistory
    ) -> PreparedInvocation: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def warmup(self, invocation: PreparedInvocation) -> None: ...

    def cold_clone(self) -> BenchmarkModel: ...

    def extend_history(
        self, history: ConversationHistory, turn: Turn, response: str
    ) -> ConversationHistory: ...


@dataclass
class TextOnnxBenchmarkModel:
    verifier: Verifier
    tokenizer: object
    source: str
    protocol: str = "conversation"
    requested_provider: str = "cpu"

    @property
    def identity(self) -> ModelIdentity:
        decoder = getattr(self.verifier, "_dec", None)
        providers = tuple(decoder.session.get_providers()) if decoder is not None else ()
        return ModelIdentity("text_onnx", self.source, self.requested_provider, providers)

    @property
    def artifacts(self) -> tuple[dict[str, object], ...]:
        root = Path(self.source)
        manifest = root / "manifest.json"
        return (
            {
                "role": "model",
                "root": str(root),
                "manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
            },
        )

    def prepare(
        self, case: ConversationCase, turn: Turn, history: ConversationHistory
    ) -> PreparedInvocation:
        del case
        tokenizer = self.tokenizer
        if self.protocol != "first-turn-workload":
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError(
                    "conversation protocol requires a tokenizer chat template; "
                    "select first-turn-workload for raw prompts"
                )
            messages = [*history, {"role": "user", "content": turn.text}]
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        else:
            ids = tokenizer(turn.text)["input_ids"]
        return PreparedInvocation(self.verifier, _token_ids(ids), tokenizer.eos_token_id)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def warmup(self, invocation: PreparedInvocation) -> None:
        invocation.verifier.prefill(invocation.prompt_ids)

    def cold_clone(self) -> TextOnnxBenchmarkModel:
        from transformers import AutoTokenizer

        from dejavuu.decoders.text import Model

        current = self.verifier
        verifier = Model(
            Path(self.source),
            current.variant,
            current.provider,
            current.threads,
            current.allow_provider_fallback,
        )
        return TextOnnxBenchmarkModel(
            verifier,
            AutoTokenizer.from_pretrained(self.source),
            self.source,
            self.protocol,
            self.requested_provider,
        )

    def extend_history(
        self, history: ConversationHistory, turn: Turn, response: str
    ) -> ConversationHistory:
        return [
            *history,
            {"role": "user", "content": turn.text},
            {"role": "assistant", "content": response},
        ]


@dataclass
class SmolVlmOnnxBenchmarkModel:
    verifier: object
    processor: object
    source: str
    protocol: str = "conversation"
    requested_provider: str = "cpu"
    image_size: int = 0

    @property
    def identity(self) -> ModelIdentity:
        decoder = getattr(self.verifier, "_dec", None)
        providers = tuple(decoder.session.get_providers()) if decoder is not None else ()
        return ModelIdentity("smolvlm_onnx", self.source, self.requested_provider, providers)

    @property
    def artifacts(self) -> tuple[dict[str, object], ...]:
        from dejavuu.decoders.vlm import REPO, REVISION, resolve_vlm_graph_path

        root = Path(self.source)
        decoder = self.verifier.decoder_path or resolve_vlm_graph_path(
            root, "decoder", self.verifier.variant
        )
        source_manifest = root / "manifest.json"
        decoder_manifest = Path(decoder).parent / "manifest.json"
        source: dict[str, object] = {
            "role": "model",
            "root": str(root),
            "source_model": REPO,
            "source_revision": REVISION,
            "selected_graphs": {
                role: str(resolve_vlm_graph_path(root, role, self.verifier.variant))
                for role in ("embed_tokens", "vision_encoder")
            },
            "manifest": json.loads(source_manifest.read_text())
            if source_manifest.exists()
            else None,
        }
        selected_decoder: dict[str, object] = {
            "role": "decoder",
            "path": str(decoder),
            "manifest": json.loads(decoder_manifest.read_text())
            if decoder_manifest.exists()
            else None,
        }
        return source, selected_decoder

    def prepare(
        self, case: ConversationCase, turn: Turn, history: ConversationHistory
    ) -> PreparedInvocation:
        del case
        from PIL import Image

        content: list[dict[str, object]] = [
            *({"type": "image", "path": str(path)} for path in turn.images),
            {"type": "text", "text": turn.text},
        ]
        messages = [*history, {"role": "user", "content": content}]
        image_paths: list[Path] = []
        template_messages: list[dict[str, object]] = []
        for message in messages:
            rendered_content = message["content"]
            if isinstance(rendered_content, list):
                rendered_content = []
                for item in message["content"]:
                    clean = dict(item)
                    if clean.get("type") == "image":
                        image_paths.append(Path(str(clean.pop("path"))))
                    rendered_content.append(clean)
            template_messages.append({"role": message["role"], "content": rendered_content})
        text = self.processor.apply_chat_template(template_messages, add_generation_prompt=True)
        if not image_paths:
            ids = self.processor(text=text, return_tensors="np")["input_ids"][0].tolist()
            return PreparedInvocation(self.verifier, ids, self.processor.tokenizer.eos_token_id)
        images = [Image.open(path).convert("RGB") for path in image_paths]
        options = (
            {"images_kwargs": {"size": {"longest_edge": self.image_size}}}
            if self.image_size
            else {}
        )
        processor_out = dict(
            self.processor(text=text, images=images, return_tensors="np", **options)
        )
        ids, prepared = self.verifier.prepare_request(processor_out)
        return PreparedInvocation(prepared, ids, self.processor.tokenizer.eos_token_id)

    def decode(self, token_ids: list[int]) -> str:
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)

    def warmup(self, invocation: PreparedInvocation) -> None:
        invocation.verifier.prefill(invocation.prompt_ids)

    def cold_clone(self) -> SmolVlmOnnxBenchmarkModel:
        from transformers import AutoProcessor

        from dejavuu.decoders.vlm import VLM

        current = self.verifier
        verifier = VLM(
            Path(self.source),
            current.variant,
            current.provider,
            current.threads,
            current.allow_provider_fallback,
            current.image_token_id,
            current.decoder_path,
        )
        return SmolVlmOnnxBenchmarkModel(
            verifier,
            AutoProcessor.from_pretrained(self.source),
            self.source,
            self.protocol,
            self.requested_provider,
            self.image_size,
        )

    def extend_history(
        self, history: ConversationHistory, turn: Turn, response: str
    ) -> ConversationHistory:
        content: list[dict[str, object]] = [
            *({"type": "image", "path": str(path)} for path in turn.images),
            {"type": "text", "text": turn.text},
        ]
        return [
            *history,
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]


def _manifest_kind(root: Path) -> str | None:
    path = root / "manifest.json"
    if not path.exists():
        return None
    provenance = json.loads(path.read_text()).get("provenance", {})
    return provenance.get("model_kind")


def load_benchmark_model(
    spec: ModelSpec,
    *,
    dataset: str,
    protocol: str,
    image_size: int = 0,
) -> BenchmarkModel:
    """Load a benchmark model independently of dataset selection."""
    root = Path(spec.path).expanduser() if spec.path else None
    if root is not None:
        problems = verify_manifest(root)
        if problems and not spec.allow_unverified_artifact:
            raise ValueError(f"unverified model artifact {root}: {'; '.join(problems)}")
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            provenance = json.loads(manifest_path.read_text()).get("provenance", {})
            variant = provenance.get("variants", {}).get(spec.variant, {})
            if (
                isinstance(variant, dict)
                and variant.get("speculative_compatible") is False
                and not spec.allow_unverified_artifact
            ):
                raise ValueError(
                    f"model variant {spec.variant!r} is not sequence-length consistent; "
                    "use a compatible graph or explicitly allow an unverified artifact"
                )
    kind = spec.kind
    if kind == "auto":
        kind = (_manifest_kind(root) if root is not None else None) or (
            "smolvlm_onnx" if dataset == "mmspec" else "text_onnx"
        )
    if dataset == "mmspec" and kind == "text_onnx":
        raise ValueError("MMSpec cases contain images and require a vision-language adapter")
    if kind == "text_onnx":
        from transformers import AutoTokenizer

        from dejavuu.decoders.text import Model, download

        model_root = root or download(spec.variant)
        verifier = Model(
            model_root,
            spec.variant,
            spec.provider,
            spec.threads,
            spec.allow_provider_fallback,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_root)
        return TextOnnxBenchmarkModel(verifier, tokenizer, str(model_root), protocol, spec.provider)
    if kind == "smolvlm_onnx":
        from transformers import AutoProcessor

        from dejavuu.decoders.vlm import VLM, download

        model_root = root or Path(download(spec.variant))
        verifier = VLM(
            model_root,
            spec.variant,
            spec.provider,
            spec.threads,
            spec.allow_provider_fallback,
        )
        if verifier.decoder_path is not None:
            decoder = Path(verifier.decoder_path)
            try:
                decoder.relative_to(model_root)
            except ValueError:
                problems = verify_manifest(decoder.parent)
                if problems and not spec.allow_unverified_artifact:
                    raise ValueError(
                        f"unverified external VLM decoder {decoder.parent}: {'; '.join(problems)}"
                    ) from None
        processor = AutoProcessor.from_pretrained(model_root)
        return SmolVlmOnnxBenchmarkModel(
            verifier,
            processor,
            str(model_root),
            protocol,
            spec.provider,
            image_size,
        )
    raise ValueError(f"unsupported benchmark model kind {kind!r}")
