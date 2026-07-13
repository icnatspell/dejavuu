"""Dataset adapters that normalize benchmark rows into prompt cases.

Adapters are deliberately model-agnostic: they retain source metadata and return raw
prompt text. Text and vision runners own tokenizer/chat-template rendering, so every
drafter still receives only token ids through the shared verifier contract.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPEEDBENCH_REVISION = "487aa718444e816458d1a0a52bfce7a454285cf4"
SPECBENCH_REVISION = "fd2c1cd7d2201ef71db4c5f4e455008f017967bf"
MMSPEC_REVISION = "290486861eabd746075ca776adb66175636c1fc0"

SPEC_TOPIC = {
    "summarization": "summarization",
    "translation": "translation",
    "qa": "question answering",
    "math_reasoning": "mathematical reasoning",
    "rag": "retrieval-augmented generation",
} | dict.fromkeys(
    ("writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"),
    "multi-turn conversation",
)
SPEC_WORKLOADS = {
    "repetitive": {"summarization", "rag", "qa", "translation"},
    "diverse": {"writing", "roleplay", "reasoning", "stem", "humanities"},
    "all": set(SPEC_TOPIC),
}
MMSPEC_TOPIC = {
    "chart understanding": "chart vqa",
    "complex reasoning pro": "complex reasoning",
    "multi-turn conversation": "multi-turn conversation",
    "general vqa": "general vqa",
    "text vqa": "text vqa",
    "image captioning": "image captioning",
}


@dataclass(frozen=True)
class Turn:
    """One user turn and any media assets attached to it."""

    text: str
    images: tuple[Path, ...] = ()


@dataclass(frozen=True)
class DatasetProvenance:
    source: str
    revision: str
    split: str
    checksum: str | None = None


@dataclass(frozen=True)
class ConversationCase:
    """One benchmark conversation plus stable source provenance."""

    case_id: str
    category: str
    turns: tuple[Turn, ...]
    metadata: dict[str, object] = field(default_factory=dict)
    provenance: DatasetProvenance | None = None

    @property
    def prompt(self) -> str:
        """Compatibility view used by the legacy first-turn runners."""
        return self.turns[0].text


BenchmarkCase = ConversationCase


def select_cases(
    cases: list[ConversationCase], n: int, per_category: int = 0
) -> list[ConversationCase]:
    """Apply one deterministic selection policy across every dataset adapter."""
    if not per_category:
        return cases[:n]
    seen: Counter[str] = Counter()
    selected: list[ConversationCase] = []
    for case in cases:
        if seen[case.category] < per_category:
            selected.append(case)
            seen[case.category] += 1
    return selected


def _download(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(f"{path.name}.sha256")
    if sidecar.exists():
        expected = sidecar.read_text().strip()
        if checksum != expected:
            raise RuntimeError(f"cached dataset artifact failed sha256 verification: {path}")
    else:
        sidecar.write_text(checksum + "\n")
    return checksum


def load_specbench_cases(
    *,
    n: int = 20,
    per_category: int = 0,
    workload: str = "all",
    revision: str = SPECBENCH_REVISION,
    cache_root: Path | None = None,
) -> list[ConversationCase]:
    """Load pinned Spec-Bench conversations, preserving every user turn."""
    root = cache_root or Path.home() / ".cache" / "dejavuu" / "datasets"
    path = root / "specbench" / revision / "question.jsonl"
    url = (
        f"https://raw.githubusercontent.com/hemingkx/Spec-Bench/{revision}/"
        "data/spec_bench/question.jsonl"
    )
    checksum = _download(url, path)
    allowed = SPEC_WORKLOADS.get(workload, {workload})
    cases: list[ConversationCase] = []
    for index, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        if row["category"] not in allowed:
            continue
        cases.append(
            ConversationCase(
                str(row.get("question_id", index)),
                SPEC_TOPIC.get(row["category"], row["category"]),
                tuple(Turn(str(text)) for text in row["turns"]),
                {"native_category": row["category"]},
                DatasetProvenance("hemingkx/Spec-Bench", revision, "test", checksum),
            )
        )
    return select_cases(cases, n, per_category)


def load_mmspec_cases(
    *,
    n: int = 20,
    per_category: int = 0,
    revision: str = MMSPEC_REVISION,
    cache_root: Path | None = None,
) -> list[ConversationCase]:
    """Load pinned MMSpec conversations and their referenced images."""
    root = cache_root or Path.home() / ".cache" / "dejavuu" / "datasets"
    base = f"https://raw.githubusercontent.com/killthefullmoon/MMSpec/{revision}/dataset/MMSpec/testmini"
    meta = root / "mmspec" / revision / "mmspec.jsonl"
    checksum = _download(f"{base}/mmspec.jsonl", meta)
    rows = [json.loads(line) for line in meta.read_text().splitlines()]
    if per_category:
        seen: Counter[str] = Counter()
        selected_rows: list[dict[str, Any]] = []
        for row in rows:
            topic = MMSPEC_TOPIC.get(row["topic"], row["topic"])
            if seen[topic] < per_category:
                selected_rows.append(row)
                seen[topic] += 1
        rows = selected_rows
    else:
        rows = rows[:n]
    candidates: list[ConversationCase] = []
    for index, row in enumerate(rows):
        image = root / "mmspec" / revision / "images" / row["image"]
        _download(f"{base}/images/{row['image']}", image)
        topic = MMSPEC_TOPIC.get(row["topic"], row["topic"])
        turns = tuple(
            Turn(str(text), (image,) if turn_index == 0 else ())
            for turn_index, text in enumerate(row["turns"])
        )
        candidates.append(
            ConversationCase(
                str(row.get("question_id", index)),
                topic,
                turns,
                {"native_topic": row["topic"], "image": row["image"]},
                DatasetProvenance("killthefullmoon/MMSpec", revision, "testmini", checksum),
            )
        )
    return candidates


def cases_from_speed_rows(rows: list[dict[str, Any]]) -> list[BenchmarkCase]:
    """Normalize SPEED-Bench qualitative rows without discarding provenance.

    The current single-request harness evaluates the first user turn. Multi-turn rows
    remain labelled in ``metadata`` rather than pretending this measures the paper's
    full conversational serving regime; that needs shared generated history/batching.
    """
    cases: list[BenchmarkCase] = []
    for row in rows:
        turns = row["turns"]
        if not turns:
            continue
        cases.append(
            ConversationCase(
                case_id=str(row["question_id"]),
                category=str(row["category"]),
                turns=tuple(Turn(str(turn)) for turn in turns),
                metadata={
                    "source": row.get("source"),
                    "sub_category": row.get("sub_category"),
                    "difficulty": row.get("difficulty"),
                    "multiturn": bool(row.get("multiturn", False)),
                    "turn_count": len(turns),
                },
                provenance=DatasetProvenance(
                    source="nvidia/SPEED-Bench",
                    revision=SPEEDBENCH_REVISION,
                    split="qualitative",
                ),
            )
        )
    return cases


def load_speedbench(
    split: str = "qualitative", revision: str = SPEEDBENCH_REVISION
) -> list[ConversationCase]:
    """Load a pinned logical SPEED-Bench split through the optional benchmark extra.

    ``qualitative`` is the default because it measures diverse single-request
    speculative decoding. The ``throughput_*`` splits are accepted as controlled
    context-length prompt sets, not as a claim of batched-server throughput.
    """
    try:
        from datasets import load_dataset  # pyrefly: ignore[missing-import]
    except ImportError as exc:  # pragma: no cover - exercised only without bench extra
        raise RuntimeError("SPEED-Bench requires `uv sync --extra bench`") from exc
    rows = list(load_dataset("nvidia/SPEED-Bench", split, split="test", revision=revision))
    cases = cases_from_speed_rows(rows)
    return [
        ConversationCase(
            case.case_id,
            case.category,
            case.turns,
            case.metadata,
            DatasetProvenance("nvidia/SPEED-Bench", revision, split),
        )
        for case in cases
    ]


def load_cases(
    name: str,
    *,
    split: str,
    revision: str | None,
    protocol: str,
    n: int,
    per_category: int,
    workload: str = "all",
) -> list[ConversationCase]:
    """Unified dataset interface used by the benchmark CLI."""
    if name == "specbench":
        if split != "test":
            raise ValueError("Spec-Bench exposes only the test split")
        cases = load_specbench_cases(
            n=n,
            per_category=per_category,
            workload=workload,
            revision=revision or SPECBENCH_REVISION,
        )
    elif name == "speedbench":
        if protocol == "official" and split.startswith("throughput"):
            raise ValueError(
                "official SPEED-Bench throughput needs a batched serving runner; "
                "use first-turn-workload for single-request context measurements"
            )
        cases = select_cases(
            load_speedbench(split, revision or SPEEDBENCH_REVISION), n, per_category
        )
    elif name == "mmspec":
        if split != "testmini":
            raise ValueError("MMSpec exposes only the testmini split")
        cases = load_mmspec_cases(
            n=n,
            per_category=per_category,
            revision=revision or MMSPEC_REVISION,
        )
    else:
        raise ValueError(f"unknown benchmark dataset {name!r}")
    if protocol == "first-turn-workload":
        return [
            ConversationCase(
                case.case_id,
                case.category,
                case.turns[:1],
                case.metadata,
                case.provenance,
            )
            for case in cases
        ]
    return cases
