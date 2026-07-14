"""NGramBackoff -- a memory-bounded, multi-order n-gram cache with high->low backoff.

Inspired by NG+ (issue #7): a hierarchical n-gram cache that keeps useful matches at
several orders under a fixed memory budget. This is a *concept* implementation -- the NG+
paper is paywalled (ACM 10.1145/3737902.3768352), so its exact eviction/promotion policy
was not verified against the source; the design here follows the issue description and the
standard stupid-backoff n-gram formulation. Reconcile with the paper if access is obtained.

How it differs from the existing n-gram drafters:

* `cacheback` is a *single* fixed-order leader->follower cache. This stores single-token
  continuations at *every* order in ``[min_order, max_order]`` and, when drafting, tries
  the longest (most specific) context first and *backs off* to shorter contexts on a miss.
  So a known long context is exploited when present, and a general short context still
  fires otherwise -- the hierarchical behaviour.
* `ngram_trie` rebuilds a trie from the *prompt* each request; this is a *persistent*
  online cache updated from emitted tokens (like `cacheback`), so it accumulates useful
  continuations across the whole run and across requests.

Memory is bounded by a single global LRU over contexts of all orders, so the footprint is
capped regardless of run length; the least-recently-used context is evicted first, which
keeps the entries that actually keep hitting. Lossless under greedy: the cache only
*proposes* tokens; the shared verifier alone decides what is emitted.
"""

from __future__ import annotations

from collections import OrderedDict

from dejavuu.drafters.base import Drafter, DraftTree


class NGramBackoff(Drafter):
    """Draft from a bounded LRU cache of multi-order n-gram continuations with backoff.

    ``min_order``..``max_order`` is the range of context lengths stored and matched.
    ``capacity`` bounds the total number of distinct contexts across all orders (the
    memory constraint); ``follower_capacity`` bounds the alternative next tokens kept per
    context. All limits apply online in the hot path.

    ``min_order`` defaults to 2 for a reason worth stating: backing off all the way to a
    unigram (``min_order=1``) drafts from a single-token context, which is a poor predictor
    -- on SPEED-Bench (Qwen3-0.6B) it dropped root top-1 agreement from ~0.46 (min_order=3)
    to ~0.25 and made it the weakest method in its family. Since every drafted step pays
    the verifier's fixed cost whether or not the guess lands, low-confidence unigram drafts
    are not worth submitting; keeping a floor of 2 trades a little recall for much higher
    precision. A confidence-gated backoff (only draft from an order whose continuation is
    frequent enough) is the natural next refinement.
    """

    def __init__(
        self,
        min_order: int = 2,
        max_order: int = 4,
        capacity: int = 32_768,
        follower_capacity: int = 4,
    ):
        if min(min_order, max_order, capacity, follower_capacity) < 1 or min_order > max_order:
            raise ValueError("NGramBackoff needs 1 <= min_order <= max_order and positive bounds")
        self.min_order = min_order
        self.max_order = max_order
        self.capacity = capacity
        self.follower_capacity = follower_capacity
        # One global LRU: context tuple (any order) -> recency-ordered set of next tokens.
        self.cache: OrderedDict[tuple[int, ...], OrderedDict[int, None]] = OrderedDict()
        self._tail: list[int] = []

    def reset(self, prompt_ids: list[int]) -> None:
        # The cache persists across requests; only the rolling update window is local.
        self._tail = []

    def update(self, accepted: list[int]) -> None:
        """Roll each newly emitted token into the cache as the continuation of every
        order-k context that ends just before it."""
        for token in accepted:
            self._tail.append(token)
            for order in range(self.min_order, self.max_order + 1):
                if len(self._tail) > order:  # need `order` tokens of context before `token`
                    self._put(tuple(self._tail[-order - 1 : -1]), token)
            if len(self._tail) > self.max_order:  # keep only what the largest order needs
                del self._tail[0]

    def _put(self, ctx: tuple[int, ...], token: int) -> None:
        followers = self.cache.get(ctx)
        if followers is None:
            followers = OrderedDict()
            self.cache[ctx] = followers
        else:
            self.cache.move_to_end(ctx)
        followers.pop(token, None)
        followers[token] = None
        while len(followers) > self.follower_capacity:
            followers.popitem(last=False)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def _lookup(self, window: list[int]) -> list[int]:
        """Longest-context-first backoff: recent next tokens for the most specific matched
        order, or ``[]`` if no order matches. Marks the hit context most-recently-used."""
        for order in range(self.max_order, self.min_order - 1, -1):
            if len(window) >= order:
                ctx = tuple(window[-order:])
                followers = self.cache.get(ctx)
                if followers is not None:
                    self.cache.move_to_end(ctx)
                    return list(reversed(followers))  # most recent first
        return []

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        if budget < 1:
            return DraftTree.chain([ctx[-1]])
        window = list(ctx)
        drafted: list[int] = []
        while len(drafted) < budget:
            cands = self._lookup(window)
            if not cands:
                break
            drafted.append(cands[0])  # most recent under the most specific matched order
            window.append(cands[0])
        return DraftTree.chain([ctx[-1], *drafted])

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        """Breadth-first expansion: branch each frontier node over its top-``width`` recent
        backoff candidates until the node budget is spent."""
        if budget < 1 or width < 1:
            return DraftTree.chain([ctx[-1]])
        tokens, parent = [ctx[-1]], [-1]
        frontier: list[tuple[int, list[int]]] = [(0, list(ctx))]
        while frontier and len(tokens) - 1 < budget:
            node, window = frontier.pop(0)
            for token in self._lookup(window)[:width]:
                if len(tokens) - 1 >= budget:
                    return DraftTree(tokens, parent)
                tokens.append(token)
                parent.append(node)
                frontier.append((len(tokens) - 1, [*window, token]))
        return DraftTree(tokens, parent)
