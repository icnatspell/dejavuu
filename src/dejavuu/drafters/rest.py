"""REST -- Retrieval over an offline datastore (+ completed generations)."""

from __future__ import annotations

from dejavuu.drafters.base import Drafter, DraftTree
from dejavuu.drafters.suffix_index import SuffixIndex


class REST(Drafter):
    """Retrieval over an offline datastore (a domain corpus passed at construction
    and/or completed generations rolled in per request), longest-suffix exact match
    -> continuation chain. Unlike PLD/SuffixDecoding it ignores the in-progress
    output and draws only on the persistent store. Token-only -> LLM and VLM share it."""

    def __init__(
        self,
        datastore: list[list[int]] | None = None,
        order: int = 8,
        min_match: int = 2,
    ):
        self.index = SuffixIndex(order)
        self.min_match = min_match
        for doc in datastore or []:
            self.index.extend(doc)
            self.index.append(self.index.SEP)
        self._cur: list[int] = []  # current generation, rolled into the store at reset

    def reset(self, prompt_ids: list[int]) -> None:
        if self._cur:
            self.index.extend(self._cur)
            self.index.append(self.index.SEP)
            self._cur = []

    def update(self, accepted: list[int]) -> None:
        self._cur.extend(accepted)

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        cont, n = self.index.continuation(ctx, budget)
        if n < self.min_match:
            return DraftTree.chain([ctx[-1]])
        return DraftTree.chain([ctx[-1], *cont])

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        conts, n = self.index.continuations(ctx, budget, width)
        if n < self.min_match:
            return DraftTree.chain([ctx[-1]])
        return DraftTree.branches(ctx[-1], conts, budget)
