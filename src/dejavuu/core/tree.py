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


def accept(
    tree: DraftTree,
    logits: np.ndarray,
    sampler: Sampler | None = None,
    base_pos: int = 0,
) -> tuple[list[int], int, list[int]]:
    """Descent over the tree (same rule as the chain accept): follow the child whose
    token matches the model's prediction -- argmax, or a position-seeded draw under
    `sampler` -- until none does. `base_pos` is the root's absolute position; the
    token predicted at depth d lives at base_pos+d+1, seeding its draw. Returns
    (emitted tokens incl. bonus, #guesses accepted, accepted node indices incl. root).
    The node-index path drives the KV gather; for a chain it is [0,1,..,n_acc]."""
    emitted: list[int] = []
    path = [0]
    node = 0
    while True:
        pred = pick(logits[node], base_pos + len(emitted) + 1, sampler)
        child = next((c for c in tree.children(node) if tree.token_ids[c] == pred), None)
        if child is None:
            emitted.append(pred)  # bonus / correction token
            return emitted, len(emitted) - 1, path
        emitted.append(pred)
        path.append(child)
        node = child
