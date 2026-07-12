"""CopySpec-style k-gram continuation copying from prompt and run history."""

from __future__ import annotations

from dejavuu.drafters.base import Drafter, DraftTree


class CopySpec(Drafter):
    """Copy the continuation after the earliest matching k-gram.

    This is the model-free copying component of CopySpec. The paper's optional
    model-drafter fallback is intentionally omitted so this stays a raw-token,
    lossless drop-in drafter for both repository backends.
    """

    def __init__(self, gamma: int = 5):
        self.gamma = gamma
        self.history: list[int] = []

    def reset(self, prompt_ids: list[int]) -> None:
        self.history = list(prompt_ids)

    def update(self, accepted: list[int]) -> None:
        self.history.extend(accepted)

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        if budget <= 0 or len(ctx) < self.gamma:
            return DraftTree.chain([ctx[-1]])
        key = ctx[-self.gamma :]
        for start in range(len(self.history) - self.gamma):
            if self.history[start : start + self.gamma] == key:
                cont = self.history[start + self.gamma : start + self.gamma + budget]
                if cont:
                    return DraftTree.chain([ctx[-1], *cont])
        return DraftTree.chain([ctx[-1]])
