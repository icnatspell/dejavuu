"""The drafter contract + the DraftTree it emits.

Every drafter works on raw token-id lists only -- never on model internals -- so the
exact same instances drive the LLM (`Model`) and the VLM (`VLM`) through the shared
`Verifier`/engine. A drafter is a chain or a branching tree (`propose_tree`).
"""

from __future__ import annotations

import heapq
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class DraftTree:
    """token_ids[0] = last accepted real token (root). A chain has parent[i]=i-1."""

    token_ids: list[int]
    parent: list[int]
    score: list[float] | None = None

    @staticmethod
    def chain(tokens: list[int]) -> DraftTree:
        return DraftTree(list(tokens), [-1, *range(len(tokens) - 1)])

    @staticmethod
    def branches(root: int, conts: list[list[int]], budget: int) -> DraftTree:
        """Root with one chain per continuation (each branch starts at the root);
        total guesses capped at `budget`. Used by retrieval drafters that retrieve
        several candidate continuations for tree verification."""
        tokens, parent = [root], [-1]
        for cont in conts:
            prev = 0  # branch from the root
            for t in cont:
                if len(tokens) - 1 >= budget:
                    return DraftTree(tokens, parent)
                tokens.append(t)
                parent.append(prev)
                prev = len(tokens) - 1
        return DraftTree(tokens, parent)

    @staticmethod
    def grow(
        root: int,
        children: Callable[[int], list[tuple[int, float]]],
        budget: int,
    ) -> DraftTree:
        """Sequoia-style dynamic topology: grow the tree by repeatedly adding the
        candidate node with the highest *path* acceptance probability (the product of
        edge probs from the root), until `budget` guesses are placed. `children(token)`
        returns that token's candidate successors as (token, prob), prob descending.

        Greedy-on-cumulative-prob is the optimal tree under a node budget when each
        edge's acceptance is independent (Sequoia): every added node contributes its
        path probability to the expected accepted length, so picking the max-marginal
        node each step maximizes the total. This replaces uniform `width` branching,
        which wastes budget on low-probability siblings (Token Recycling tree
        over-branched: accept% 77->19% before this)."""
        tokens, parent, score = [root], [-1], [1.0]
        heap: list[tuple[float, int, int, int, frozenset[int]]] = []
        tie = itertools.count()

        def push(node: int, path_p: float, seen: frozenset[int]) -> None:
            for tok, p in children(tokens[node]):
                if tok not in seen:  # no cycles along a path (mirrors chain mode)
                    heapq.heappush(heap, (-path_p * p, next(tie), node, tok, seen))

        push(0, 1.0, frozenset([root]))
        while heap and len(tokens) - 1 < budget:
            neg_p, _, par, tok, seen = heapq.heappop(heap)
            idx = len(tokens)
            tokens.append(tok)
            parent.append(par)
            score.append(-neg_p)
            push(idx, -neg_p, seen | {tok})
        return DraftTree(tokens, parent, score)

    def children(self, node: int) -> list[int]:
        return [i for i, p in enumerate(self.parent) if p == node]


class Drafter(ABC):
    @abstractmethod
    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree: ...

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        """Branching draft for tree verification. Default: the chain (width
        ignored). Drafters with ranked candidates (Token Recycling) override this."""
        return self.propose(ctx, past_len, budget)

    def update(self, accepted: list[int]) -> None:  # noqa: B027
        """Feed accepted tokens back; no-op for stateless drafters like PLD."""

    def observe(self, input_tokens: list[int], logits) -> None:  # noqa: B027
        """Verify logits the engine already computed -- free training signal for
        Token Recycling. No-op for retrieval drafters."""

    def observe_hidden(self, tokens: list[int], hidden, base_pos: int) -> None:  # noqa: B027
        """Per-token hidden states for the accepted path: tokens[i] sits at absolute
        position base_pos+i with row hidden[i]. Only called when the decoder emits
        hidden states. Representation-aware drafters (PLD+, AdaPLD) build a
        position->hidden memory here; no-op default."""

    def note_cost(self, verify_s: float, submitted: int) -> None:  # noqa: B027
        """Per-step verify time and # draft tokens submitted. Lets a cost-aware drafter
        learn verify_s ~= c0 + c1*submitted and size drafts to the load. No-op default."""

    def reset(self, prompt_ids: list[int]) -> None:  # noqa: B027
        """A new request is starting (called once per `generate`). Stateful
        drafters rotate per-request state / roll history into their store here."""
