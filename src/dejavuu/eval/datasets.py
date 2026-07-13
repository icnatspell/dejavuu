"""Dataset adapters that normalize benchmark rows into prompt cases.

Adapters are deliberately model-agnostic: they retain source metadata and return raw
prompt text. Text and vision runners own tokenizer/chat-template rendering, so every
drafter still receives only token ids through the shared verifier contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    """One independently generated benchmark prompt plus stable provenance."""

    case_id: str
    category: str
    prompt: str
    metadata: dict[str, object] = field(default_factory=dict)


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
            BenchmarkCase(
                case_id=str(row["question_id"]),
                category=str(row["category"]),
                prompt=str(turns[0]),
                metadata={
                    "source": row.get("source"),
                    "sub_category": row.get("sub_category"),
                    "difficulty": row.get("difficulty"),
                    "multiturn": bool(row.get("multiturn", False)),
                    "turn_count": len(turns),
                },
            )
        )
    return cases


def load_speedbench(split: str = "qualitative") -> list[BenchmarkCase]:
    """Load a pinned logical SPEED-Bench split through the optional benchmark extra.

    ``qualitative`` is the default because it measures diverse single-request
    speculative decoding. The ``throughput_*`` splits are accepted as controlled
    context-length prompt sets, not as a claim of batched-server throughput.
    """
    try:
        from datasets import load_dataset  # pyrefly: ignore[missing-import]
    except ImportError as exc:  # pragma: no cover - exercised only without bench extra
        raise RuntimeError("SPEED-Bench requires `uv sync --extra bench`") from exc
    rows = list(load_dataset("nvidia/SPEED-Bench", split, split="test"))
    return cases_from_speed_rows(rows)
