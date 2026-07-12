"""SAM-Decoding -- static datastore + live generation, match-length source selection."""

from __future__ import annotations

from dejavuu.drafters.base import Drafter, DraftTree
from dejavuu.drafters.suffix_index import SuffixIndex


class SAMDecoding(Drafter):
    """SAM-Decoding: keep two suffix indexes -- a static datastore (like REST) and the
    live generation (like SuffixDecoding) -- and each step draft from whichever yields
    the *longer* suffix match. Match length is the retrieval confidence: it both picks
    the source and caps the draft length (a long match earns a long draft, a short one
    a cautious one). The match-length source selection is the new bit vs REST/Suffix.

    ponytail: a hash n-gram index, not a real suffix automaton (see SuffixIndex). With
    no datastore it reduces to SuffixDecoding; the static store is where it earns its
    keep. No model-based fallback -- pure retrieval, so it stays lossless under greedy."""

    def __init__(
        self,
        datastore: list[list[int]] | None = None,
        order: int = 8,
        min_match: int = 2,
    ):
        self.static = SuffixIndex(order)
        for doc in datastore or []:
            self.static.extend(doc)
            self.static.append(self.static.SEP)
        self.dynamic = SuffixIndex(order)
        self.min_match = min_match
        self.seen = 0  # ctx tokens already rolled into the dynamic index

    def reset(self, prompt_ids: list[int]) -> None:
        self.dynamic.append(self.dynamic.SEP)
        self.seen = 0

    def _ingest(self, ctx: list[int]) -> None:
        self.dynamic.extend(ctx[self.seen :])
        self.seen = len(ctx)

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        self._ingest(ctx)
        cs, ns = self.static.continuation(ctx, budget, by_freq=True)
        cd, nd = self.dynamic.continuation(ctx, budget, by_freq=True)
        cont, n = (cs, ns) if ns >= nd else (cd, nd)  # higher-confidence source
        if n < self.min_match:
            return DraftTree.chain([ctx[-1]])
        return DraftTree.chain([ctx[-1], *cont[:n]])  # confidence-gated draft length

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        self._ingest(ctx)
        cs, ns = self.static.continuations(ctx, budget, width)
        cd, nd = self.dynamic.continuations(ctx, budget, width)
        conts, n = (cs, ns) if ns >= nd else (cd, nd)
        if n < self.min_match:
            return DraftTree.chain([ctx[-1]])
        return DraftTree.branches(ctx[-1], conts, budget)
