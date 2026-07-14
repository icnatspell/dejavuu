"""Tree verification primitives: flatten a DraftTree into one forward pass.

Model-agnostic -- pure numpy over token ids / KV arrays, so the same path serves the
LLM and the VLM. A chain is the degenerate tree (one child per node), so the chain
engine path is just this with a contiguous accepted path; we keep the cheap chain
code separate only to avoid the gather copy on the hot path.

The forward a tree needs that a chain does not: explicit `position_ids` (siblings
share a position) and a 4D additive attention mask (a node sees committed KV + its
ancestors, never its siblings). Today's stock exports expose neither -- see
`Verifier.supports_tree`; until a decoder does, `engine.generate(tree=True)` falls
back to chain. The equivalence these primitives must satisfy ("a tree forward gives
each node the same logits as running its root->node path as a plain causal chain")
is what test_tree.py pins down against a reference decoder, model-free.
"""

from __future__ import annotations

import numpy as np

from dejavuu.core.sampling import Sampler, pick

# gather_kv is re-exported so it keeps one home (dejavuu.core.verifier, alongside the
# other numpy KV op) while `tree.gather_kv` stays importable. The engine now calls the
# backend's `gather_kv` method, which defaults to this for numpy KV.
from dejavuu.core.verifier import gather_kv
from dejavuu.drafters import DraftTree

__all__ = ["accept", "depths", "gather_kv", "mask", "positions"]

NEG = -1e9  # additive-mask "blocked"; large enough to zero the softmax weight


def depths(parent: list[int]) -> list[int]:
    """Depth of each node (root=0). Requires parent[i] < i (topological order)."""
    d = [0] * len(parent)
    for i, p in enumerate(parent):
        if p >= 0:
            d[i] = d[p] + 1
    return d


def positions(parent: list[int], past_len: int) -> np.ndarray:
    """position_ids[i] = past_len + depth(i); siblings share a position. [1, M]."""
    return np.asarray([[past_len + x for x in depths(parent)]], dtype=np.int64)


def mask(parent: list[int], past_len: int) -> np.ndarray:
    """Additive attention bias [1, 1, M, past_len+M]: node i may attend to all
    committed KV and to its ancestors (incl. itself); everything else is blocked."""
    m = len(parent)
    bias = np.full((1, 1, m, past_len + m), NEG, dtype=np.float32)
    bias[..., :past_len] = 0.0  # all committed KV is visible to every node
    for i in range(m):
        j = i
        while j != -1:  # walk i -> root, unblocking each ancestor column
            bias[0, 0, i, past_len + j] = 0.0
            j = parent[j]
    return bias


def normalized_entropy(logits_row: np.ndarray) -> float:
    """Shannon entropy of softmax(logits), normalized to [0, 1] by log(vocab). 0 means
    one token dominates (confident); 1 means a uniform distribution (maximally uncertain)."""
    e = np.exp(logits_row - logits_row.max())
    p = e / e.sum()
    h = float(-(p * np.log(p + 1e-12)).sum())
    return h / np.log(len(logits_row))


def pick_child(
    tree: DraftTree,
    node: int,
    logits_row: np.ndarray,
    position: int,
    sampler: Sampler | None,
    top_k: int,
    entropy_gate: float = 0.0,
    min_prob_ratio: float = 0.0,
) -> tuple[int, int | None]:
    """One acceptance step, shared by the chain and tree descents. Returns
    (token_to_emit, child_to_descend_into); child None means stop (the token is the
    bonus/correction).

    Lossless (``top_k == 1`` or any ``sampler``): emit the model's pick -- greedy
    argmax or a position-seeded draw -- and descend only into the child equal to it.
    This is the exact greedy/coupling acceptance and the ONLY path the model-free
    conformance suite exercises; it must stay bit-exact.

    Loose (``top_k > 1``, greedy only): accept a drafted child whose token lies in the
    model's top-k, preferring the most probable such child. This trades token identity
    for speed and is opt-in -- callers measure the quality cost via the response scorers.
    Falls back to the argmax correction when no child qualifies.

    Plausibility gate (``min_prob_ratio > 0``): accept a non-argmax drafted child only
    when it is a genuine near-tie -- its probability is at least ``min_prob_ratio`` times
    the argmax's, i.e. ``logit[tok] >= max_logit + log(min_prob_ratio)`` (no softmax
    needed). This is the sharp version of the entropy gate: full-vocab entropy is nearly
    always low for a peaked LM, so it can't tell a plausible runner-up from an unlikely
    one, whereas the top-1-vs-runner-up margin directly measures the drift risk of a
    substitution. Set close to 1.0 for near-exact, lower to accept more.

    Entropy gate (``entropy_gate > 0``, FLy-style): loosen *only* where the target is
    uncertain. At a position whose normalized entropy is below the gate the model is
    confident, so acceptance is demoted to exact (top-1). Superseded by
    ``min_prob_ratio`` for drift control; kept for comparison sweeps.
    """
    k = top_k
    if (
        sampler is None
        and top_k > 1
        and entropy_gate > 0.0
        and normalized_entropy(logits_row) < entropy_gate
    ):
        k = 1  # confident position -> stay exact
    if sampler is None and k > 1:
        kk = min(k, len(logits_row))
        topk = {int(t) for t in np.argpartition(-logits_row, kk - 1)[:kk]}
        # Probability-ratio floor: a runner-up must be within `min_prob_ratio` of the
        # argmax to be a safe substitution. -inf disables the floor (argmax always clears
        # it, so an exact-match child is never filtered).
        floor = (
            -np.inf
            if min_prob_ratio <= 0.0
            else float(logits_row.max()) + float(np.log(min_prob_ratio))
        )
        best, best_logit = None, -np.inf
        for c in tree.children(node):
            tok = tree.token_ids[c]
            if tok in topk and logits_row[tok] >= floor and logits_row[tok] > best_logit:
                best, best_logit = c, logits_row[tok]
        if best is not None:
            return tree.token_ids[best], best
        return int(logits_row.argmax()), None
    pred = pick(logits_row, position, sampler)
    child = next((c for c in tree.children(node) if tree.token_ids[c] == pred), None)
    return pred, child


def accept(
    tree: DraftTree,
    logits: np.ndarray,
    sampler: Sampler | None = None,
    base_pos: int = 0,
    top_k: int = 1,
    entropy_gate: float = 0.0,
    min_prob_ratio: float = 0.0,
) -> tuple[list[int], int, list[int]]:
    """Descent over the tree via `pick_child`. `base_pos` is the root's absolute
    position; the token at depth d lives at base_pos+d+1, seeding its draw. Returns
    (emitted tokens incl. bonus, #guesses accepted, accepted node indices incl. root).
    The node-index path drives the KV gather; for a chain it is [0,1,..,n_acc].
    `top_k > 1` enables loose (lossy) acceptance; `min_prob_ratio`/`entropy_gate` gate
    it -- see `pick_child`."""
    emitted: list[int] = []
    path = [0]
    node = 0
    while True:
        tok, child = pick_child(
            tree,
            node,
            logits[node],
            base_pos + len(emitted) + 1,
            sampler,
            top_k,
            entropy_gate,
            min_prob_ratio,
        )
        emitted.append(tok)
        if child is None:
            return emitted, len(emitted) - 1, path
        path.append(child)
        node = child
