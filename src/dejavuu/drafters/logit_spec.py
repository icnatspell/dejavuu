"""LogitSpec -- logit-conditioned n-gram retrieval drafts.

The verifier's previous forward pass already computed a distribution for each input
token.  We retain its top candidates by token id.  At the next occurrence of that
token, the highest-ranked candidate starts the chain and an n-gram lookup retrieves
the tokens that historically followed it.  Tree mode keeps several top-logit
candidates as sibling branches.

This follows LogitSpec's useful part within DejaVu's anchor-root verifier protocol:
the current anchor remains the tree root, so all guesses are still checked by the
ordinary lossless verifier.  The first decode step has no cached logit and therefore
returns only the anchor.
"""

from __future__ import annotations

import numpy as np

from dejavuu.drafters.base import Drafter, DraftTree


class LogitSpec(Drafter):
    """Retrieve continuations conditioned on top candidates from verifier logits.

    ``observe`` consumes logits the engine has already paid to compute; proposal
    itself uses only token ids and the cached candidate list.  This keeps drafting
    off the model hot path while making lexical retrieval usable after a likely next
    token even when the current suffix has no exact match.
    """

    def __init__(self, k: int = 8, order: int = 3):
        self.k = k
        self.order = order
        self.successors: dict[int, list[int]] = {}

    def observe(self, input_tokens: list[int], logits) -> None:
        """Cache each verified token's top ``k`` predicted successors in rank order."""
        k = min(self.k, logits.shape[-1])
        if k <= 0:
            return
        topk = np.argpartition(-logits, k - 1, axis=-1)[:, :k]
        rows = np.arange(len(input_tokens))[:, None]
        topk = np.take_along_axis(topk, np.argsort(-logits[rows, topk], axis=-1), axis=-1)
        for token, candidates in zip(input_tokens, topk, strict=True):
            self.successors[token] = [int(candidate) for candidate in candidates]

    def _continuation(self, ctx: list[int], candidate: int, budget: int) -> list[int]:
        """Return ``candidate`` plus the longest earlier n-gram continuation.

        The fresh candidate at the end is deliberately excluded as a match source;
        only an earlier occurrence may supply its next-next-token continuation.
        """
        out = [candidate]
        if budget <= 1:
            return out[:budget]
        query = [*ctx, candidate]
        for n in range(min(self.order, len(query)), 0, -1):
            suffix = query[-n:]
            for start in range(len(ctx) - n, -1, -1):
                if ctx[start : start + n] != suffix:
                    continue
                end = start + n
                cont = ctx[end : end + budget - 1]
                if cont:
                    return [candidate, *cont]
        return out

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        candidates = self.successors.get(ctx[-1], [])
        if not candidates or budget <= 0:
            return DraftTree.chain([ctx[-1]])
        return DraftTree.chain([ctx[-1], *self._continuation(ctx, candidates[0], budget)])

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        candidates = self.successors.get(ctx[-1], [])[:width]
        if not candidates or budget <= 0:
            return DraftTree.chain([ctx[-1]])
        conts = [self._continuation(ctx, candidate, budget) for candidate in candidates]
        return DraftTree.branches(ctx[-1], conts, budget)
