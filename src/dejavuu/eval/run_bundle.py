"""Atomic, non-appendable result bundle for one benchmark configuration."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunBundle:
    target: Path
    staging: Path
    manifest: dict[str, object] = field(default_factory=dict)
    _finalized: bool = False

    @classmethod
    def create(cls, target: Path, manifest: dict[str, object]) -> RunBundle:
        target = Path(target)
        if target.exists():
            raise FileExistsError(f"benchmark run already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
        staging.mkdir()
        (staging / "logs").mkdir()
        return cls(target, staging, dict(manifest))

    def write_jsonl(self, name: str, records: list[dict[str, object]]) -> Path:
        if self._finalized:
            raise RuntimeError("run bundle is already finalized")
        path = self.staging / name
        with path.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return path

    def path(self, name: str) -> Path:
        if self._finalized:
            return self.target / name
        return self.staging / name

    def write_responses(self, records: list[dict[str, object]]) -> Path:
        return self.write_jsonl("responses.jsonl", records)

    def write_divergences(self, records: list[dict[str, object]]) -> Path:
        return self.write_jsonl("divergences.jsonl", records)

    def finalize(self, status: str) -> Path:
        if self._finalized:
            raise RuntimeError("run bundle is already finalized")
        if self.target.exists():
            raise FileExistsError(f"benchmark run already exists: {self.target}")
        self.manifest["status"] = status
        manifest = self.staging / "manifest.json"
        manifest.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")
        # Atomic on the same filesystem: readers never observe a half-written run.
        os.replace(self.staging, self.target)
        self._finalized = True
        return self.target
