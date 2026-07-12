"""SuffixDecoding -- online longest-suffix drafting over the run's own history."""

from __future__ import annotations

from dejavuu.drafters.base import Drafter, DraftTree
from dejavuu.drafters.suffix_index import SuffixIndex


class SuffixDecoding(Drafter):
    """Online suffix decoding: one growing index over every sequence seen this run
    -- prior generations *and* the live prompt+output, SEP-separated (the 'global +
    per-request' corpus). Longest-suffix match, frequency-scored continuation, and
    adaptive depth (draft length tracks match length). Subsumes PLD's within-prompt
    lookup and adds cross-request memory. Token-only -> LLM and VLM share it."""

    def __init__(self, order: int = 8, min_match: int = 2):
        self.index = SuffixIndex(order)
        self.min_match = min_match
        self.seen = 0  # ctx tokens of the current request already indexed

    def reset(self, prompt_ids: list[int]) -> None:
        self.index.append(self.index.SEP)  # close the previous request
        self.seen = 0

    def _ingest(self, ctx: list[int]) -> None:
        self.index.extend(ctx[self.seen :])
        self.seen = len(ctx)

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        self._ingest(ctx)
        cont, n = self.index.continuation(ctx, budget, by_freq=True)
        if n < self.min_match:
            return DraftTree.chain([ctx[-1]])
        return DraftTree.chain([ctx[-1], *cont[:n]])  # adaptive: longer match, longer draft

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        self._ingest(ctx)
        conts, n = self.index.continuations(ctx, budget, width)
        if n < self.min_match:
            return DraftTree.chain([ctx[-1]])
        return DraftTree.branches(ctx[-1], conts, budget)
