"""STAND -- stochastic adaptive n-gram drafting from cached verifier logits.

This first slice stores sparse top-k target candidates by token n-gram. The shared
seeded sampler remains the acceptance authority, preserving the target distribution.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dejavuu.drafters.base import Drafter, DraftTree


class STAND(Drafter):
    """Reuse free verifier logits as probability-ranked n-gram tree candidates."""

    def __init__(self, order: int = 4, k: int = 4, max_draft: int = 8):
        if min(order, k, max_draft) < 1:
            raise ValueError("STAND order, k, and max_draft must be positive")
        self.order, self.k, self.max_draft = order, k, max_draft
        self.cap = max_draft
        self._proposed = 0
        self.successors: dict[tuple[int, ...], list[tuple[int, float]]] = {}

    def observe(self, input_tokens: list[int], logits) -> None:
        logits = np.asarray(logits)
        k = min(self.k, logits.shape[-1])
        for row in range(len(input_tokens)):
            values = logits[row]
            part = np.argpartition(-values, k - 1)[:k]
            top = part[np.argsort(-values[part])]
            shifted = values - values.max()
            probs = np.exp(shifted) / np.exp(shifted).sum()
            candidates = [(int(tok), float(probs[tok])) for tok in top]
            for n in range(1, min(self.order, row + 1) + 1):
                self.successors[tuple(input_tokens[row - n + 1 : row + 1])] = candidates

    def _candidates(self, path: list[int]) -> list[tuple[int, float]]:
        for n in range(min(self.order, len(path)), 0, -1):
            if candidates := self.successors.get(tuple(path[-n:])):
                return candidates
        return []

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        tree = self.propose_tree(ctx, past_len, min(budget, self.cap), width=1)
        chain = [tree.token_ids[0]]
        node = 0
        while children := tree.children(node):
            node = children[0]
            chain.append(tree.token_ids[node])
        self._proposed = len(chain) - 1
        return DraftTree.chain(chain)

    def update(self, accepted: list[int]) -> None:
        if not self._proposed:
            return
        landed = len(accepted) - 1
        if landed >= self._proposed:
            self.cap = min(self.max_draft, self.cap + 1)
        else:
            self.cap = max(1, (self.cap + landed + 1) // 2)

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        if len(ctx) < self.order:
            return DraftTree.chain([ctx[-1]])
        tokens, parent = [ctx[-1]], [-1]
        frontier = deque([(0, list(ctx))])
        while frontier and len(tokens) - 1 < budget:
            node, path = frontier.popleft()
            for tok, _ in self._candidates(path)[:width]:
                if len(tokens) - 1 >= budget:
                    return DraftTree(tokens, parent)
                tokens.append(tok)
                parent.append(node)
                frontier.append((len(tokens) - 1, [*path, tok]))
        return DraftTree(tokens, parent)
