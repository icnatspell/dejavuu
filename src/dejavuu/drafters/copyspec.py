"""CopySpec-style k-gram continuation copying from prompt and run history."""

from __future__ import annotations

from dejavuu.drafters.base import Drafter, DraftTree


class CopySpec(Drafter):
    """Copy the continuation after a matching k-gram of the context so far.

    This is the model-free copying component of CopySpec. The paper's optional
    model-drafter fallback is intentionally omitted so this stays a raw-token,
    lossless drop-in drafter for both repository backends.

    A fixed match length trades recall for precision: a long k-gram rarely recurs, so
    the drafter proposes nothing on most steps. We instead try the *longest* match
    first and fall back to shorter ones down to ``gamma_min`` -- the longest available
    match keeps precision high where it exists, while the shorter fallbacks recover the
    steps a fixed length would have skipped. Because losslessness is the verifier's job,
    proposing more aggressively can only raise accepted length, never correctness.
    """

    def __init__(self, gamma: int = 5, gamma_min: int = 3):
        # gamma is the longest match tried; gamma_min the shortest. Clamp so an explicit
        # small gamma (e.g. CopySpec(gamma=2)) still matches at that length.
        self.gamma, self.gamma_min = gamma, min(gamma_min, gamma)
        self.history: list[int] = []

    def reset(self, prompt_ids: list[int]) -> None:
        self.history = list(prompt_ids)

    def update(self, accepted: list[int]) -> None:
        self.history.extend(accepted)

    def _continuations(self, key: list[int], budget: int) -> list[list[int]]:
        """Continuations after each earliest occurrence of ``key``, one per distinct
        next token (earliest occurrence kept), in match order."""
        n = len(key)
        conts: list[list[int]] = []
        seen_next: set[int] = set()
        for start in range(len(self.history) - n):
            if self.history[start : start + n] != key:
                continue
            cont = self.history[start + n : start + n + budget]
            if cont and cont[0] not in seen_next:
                seen_next.add(cont[0])
                conts.append(cont)
        return conts

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        root = ctx[-1]
        if budget <= 0 or len(ctx) < self.gamma_min:
            return DraftTree.chain([root])
        for g in range(min(self.gamma, len(ctx)), self.gamma_min - 1, -1):
            conts = self._continuations(ctx[-g:], budget)
            if conts:
                return DraftTree.chain([root, *conts[0]])
        return DraftTree.chain([root])

    # No propose_tree override: at practical budgets, splitting the budget into shallow
    # branches loses more accepted length on a single deep, high-confidence copy than the
    # extra branches recover (roleplay accept 2.40 -> 1.97 when branched). CopySpec's edge
    # is one long precise match, so tree verification reuses the deep chain (base default).
