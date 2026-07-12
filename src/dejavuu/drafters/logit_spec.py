"""LogitSpec -- logit-conditioned n-gram retrieval drafts.

The verifier's previous forward pass already computed a distribution for each input
token.  We retain top candidates by each submitted node's recent ancestor path (with
a token-id fallback for a cold cache). At the next matching context, the
highest-ranked candidate starts the chain and an n-gram lookup retrieves the tokens
that historically followed it. Tree mode keeps several top-logit candidates as
sibling branches.

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
    itself uses only token ids and the cached candidate list. Exact recent-context
    keys avoid reusing a repeated token's logits from an unrelated context, while a
    token-only fallback still lets the drafter start when a new path has no cache.
    """

    def __init__(self, k: int = 8, order: int = 3, context_order: int = 8):
        self.k = k
        self.order = order
        self.context_order = context_order
        self.successors: dict[int, list[int]] = {}
        self.context_successors: dict[tuple[int, ...], list[int]] = {}
        self._pending_contexts: list[tuple[int, ...]] = []

    def _key(self, ctx: list[int]) -> tuple[int, ...]:
        return tuple(ctx[-self.context_order :])

    def _remember_tree(self, ctx: list[int], tree: DraftTree) -> None:
        """Associate every submitted node with its real ancestor path.

        ``observe`` receives flat verifier rows, while tree verification scatters
        siblings. Keeping these paths here lets a later proposal reuse logits only
        when its recent token context matches the context that produced them.
        """
        contexts: list[list[int]] = [list(ctx)]
        for node in range(1, len(tree.token_ids)):
            contexts.append([*contexts[tree.parent[node]], tree.token_ids[node]])
        self._pending_contexts = [self._key(path) for path in contexts]

    def observe(self, input_tokens: list[int], logits) -> None:
        """Cache each verified token's top ``k`` predicted successors in rank order."""
        k = min(self.k, logits.shape[-1])
        if k <= 0:
            return
        topk = np.argpartition(-logits, k - 1, axis=-1)[:, :k]
        rows = np.arange(len(input_tokens))[:, None]
        topk = np.take_along_axis(topk, np.argsort(-logits[rows, topk], axis=-1), axis=-1)
        for i, (token, candidates) in enumerate(zip(input_tokens, topk, strict=True)):
            ranked = [int(candidate) for candidate in candidates]
            self.successors[token] = ranked
            if i < len(self._pending_contexts):
                self.context_successors[self._pending_contexts[i]] = ranked
        self._pending_contexts = []

    def _candidates(self, ctx: list[int]) -> list[int]:
        """Use an exact recent-context cache first; token-only is a cold fallback."""
        return self.context_successors.get(self._key(ctx), self.successors.get(ctx[-1], []))

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
        candidates = self._candidates(ctx)
        if not candidates or budget <= 0:
            tree = DraftTree.chain([ctx[-1]])
        else:
            tree = DraftTree.chain([ctx[-1], *self._continuation(ctx, candidates[0], budget)])
        self._remember_tree(ctx, tree)
        return tree

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        candidates = self._candidates(ctx)[:width]
        if not candidates or budget <= 0:
            tree = DraftTree.chain([ctx[-1]])
        else:
            conts = [self._continuation(ctx, candidate, budget) for candidate in candidates]
            tree = DraftTree.branches(ctx[-1], conts, budget)
        self._remember_tree(ctx, tree)
        return tree

    def reset(self, prompt_ids: list[int]) -> None:
        self.context_successors.clear()
        self._pending_contexts = []
