"""STAND -- stochastic adaptive n-gram drafting from cached verifier logits.

This first slice stores sparse top-k target candidates by token n-gram. The shared
seeded sampler remains the acceptance authority, preserving the target distribution.
"""

from __future__ import annotations

import heapq
import itertools

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
        self._sampler = None
        self._position = 0
        self._tree_sampling = False
        self.successors: dict[tuple[int, ...], list[tuple[int, float]]] = {}

    def observe(self, input_tokens: list[int], logits) -> None:
        logits = np.asarray(logits)
        k = min(self.k, logits.shape[-1])
        for row in range(len(input_tokens)):
            values = logits[row]
            # ponytail: partition around the k-largest directly -- skips negating the
            # whole 262k-wide vocab row (an alloc + write per draft token) just to flip
            # argpartition's low-side to the high-side.
            part = np.argpartition(values, -k)[-k:]
            top = part[np.argsort(-values[part])]
            # ponytail: softmax the k survivors locally -- no full-vocab exp at all.
            # The denominator only rescales candidates within a node (a constant shift
            # in log space), which leaves chain ranking and per-node Gumbel order
            # unchanged; argpartition is then the only vocab-wide pass observe pays.
            ex = np.exp(values[top] - values[top[0]])
            probs = ex / ex.sum()
            candidates = [(int(tok), float(prob)) for tok, prob in zip(top, probs)]
            for n in range(1, min(self.order, row + 1) + 1):
                self.successors[tuple(input_tokens[row - n + 1 : row + 1])] = candidates

    def _candidates(self, path: list[int]) -> list[tuple[int, float]]:
        for n in range(min(self.order, len(path)), 0, -1):
            if candidates := self.successors.get(tuple(path[-n:])):
                if self._sampler is not None and self._tree_sampling:
                    logits = np.log(np.asarray([p for _, p in candidates]))
                    order = self._sampler.gumbel_topk(
                        logits, self._position + len(path), len(candidates)
                    )
                    return [candidates[i] for i in order]
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

    def set_sampling(self, sampler, position: int, tree: bool = False) -> None:
        self._sampler, self._position, self._tree_sampling = sampler, position, tree

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        if len(ctx) < self.order:
            return DraftTree.chain([ctx[-1]])
        tokens, parent = [ctx[-1]], [-1]
        frontier: list[tuple[float, int, int, int, list[int]]] = []
        tie = itertools.count()
        for tok, prob in self._candidates(ctx)[:width]:
            heapq.heappush(frontier, (-prob, next(tie), 0, tok, [*ctx, tok]))
        while frontier and len(tokens) - 1 < budget:
            neg_prob, _, node, tok, path = heapq.heappop(frontier)
            tokens.append(tok)
            parent.append(node)
            child = len(tokens) - 1
            for nxt, prob in self._candidates(path)[:width]:
                heapq.heappush(frontier, (neg_prob * prob, next(tie), child, nxt, [*path, nxt]))
        return DraftTree(tokens, parent)
