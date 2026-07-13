"""Generation loop: drafter -> verify -> greedy accept -> chain rollback.

Convention: the last emitted token is the uncommitted "anchor" -- it is re-fed as
input[0] each step, and committed only once verified. So past KV length == number
of committed (non-anchor) tokens, and accepting m of K draft guesses commits m+1
tokens (anchor + m), leaving the bonus token as the next anchor (plan 4.5).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import numpy as np
from loguru import logger

from dejavuu.core import tree as treelib
from dejavuu.core.sampling import Sampler, pick
from dejavuu.core.verifier import Verifier
from dejavuu.drafters import Drafter, DraftTree


@dataclass
class GenResult:
    tokens: list[int]
    steps: int = 0  # verifier forward passes
    accepted: int = 0  # draft guesses accepted (excludes bonus tokens)
    drafted: int = 0  # draft guesses proposed
    draft_s: float = 0.0  # cumulative time in drafter.propose
    verify_s: float = 0.0  # cumulative time in model.forward
    learn_s: float = 0.0  # cumulative time in the post-verify drafter callbacks
    prefill_s: float = 0.0  # one-time prompt prefill (kept out of per-step overhead)
    draft_setup_s: float = 0.0  # per-request drafter reset (online once, not decode)
    root_proposals: int = 0
    root_top1: int = 0
    root_top5: int = 0
    # Position-conditioned acceptance telemetry. Index 0 is the first draft
    # position below the anchor. A depth joins the denominator only when the
    # target path reached it and the drafter supplied a candidate there.
    conditional_attempts: list[int] = field(default_factory=list)
    conditional_accepted: list[int] = field(default_factory=list)


@cache  # warn once per model class, not every prompt
def _warn_no_tree(model_name: str) -> None:
    logger.warning(
        "tree=True but {} has no tree-capable decoder (2D mask / no position_ids) -- "
        "falling back to chain. Re-export with position_ids + 4D mask to enable.",
        model_name,
    )


def _accept_chain(
    tree: DraftTree, logits: np.ndarray, sampler: Sampler | None, base_pos: int
) -> tuple[list[int], int]:
    """Descend while the model's prediction (argmax, or a seeded draw under `sampler`)
    matches a child token. `base_pos` is the anchor's absolute position; the token
    predicted at depth d lives at base_pos+d+1, which seeds its draw.
    Returns (newly emitted tokens incl. bonus, # draft guesses accepted)."""
    emitted: list[int] = []
    node = 0
    while True:
        pred = pick(logits[node], base_pos + len(emitted) + 1, sampler)
        child = next((c for c in tree.children(node) if tree.token_ids[c] == pred), None)
        if child is None:
            emitted.append(pred)  # bonus / correction token
            return emitted, len(emitted) - 1
        emitted.append(pred)
        node = child


def _record_conditional_acceptance(res: GenResult, opportunities: int, accepted: int) -> None:
    """Accumulate conditional acceptance rate (CAR) counts by target-path depth.

    CAR[d] asks: after depths before d were accepted, how often did the drafter's
    candidate at d match the verifier? It intentionally excludes unreachable tree
    siblings and absent candidates, so chain and tree results share one definition.
    """
    if opportunities < accepted:
        raise ValueError("accepted draft positions cannot exceed CAR opportunities")
    while len(res.conditional_attempts) < opportunities:
        res.conditional_attempts.append(0)
        res.conditional_accepted.append(0)
    for depth in range(opportunities):
        res.conditional_attempts[depth] += 1
        if depth < accepted:
            res.conditional_accepted[depth] += 1


def generate(
    model: Verifier,
    prompt_ids: list[int],
    max_new: int,
    drafter: Drafter | None = None,
    budget: int = 8,
    eos: int | None = None,
    on_emit: Callable[[int, bool], None] | None = None,
    tree: bool = False,
    width: int = 2,
    sampler: Sampler | None = None,
) -> GenResult:
    """Decode against any Verifier (LLM or VLM). drafter=None is the plain
    autoregressive baseline. `tree=True` runs tree verification (branching
    drafts) -- but only if `model.supports_tree`; otherwise it falls back to the
    chain path (the stock 2D-mask exports can't express tree attention). `sampler=None`
    is greedy; a Sampler draws from the target distribution -- still lossless, the
    output law is the target's regardless of the drafts."""
    use_tree = tree and model.supports_tree
    if tree and not use_tree and drafter is not None:
        _warn_no_tree(type(model).__name__)
    seq = list(prompt_ids)
    t0 = time.perf_counter()
    past, committed = model.prefill(seq)  # prompt[-1] stays as the first anchor
    prefill_s = time.perf_counter() - t0
    draft_setup_s = 0.0
    if drafter is not None:
        t0 = time.perf_counter()
        drafter.reset(seq)  # rotate per-request state for stateful drafters
        draft_setup_s = time.perf_counter() - t0

    res = GenResult(tokens=[], prefill_s=prefill_s, draft_setup_s=draft_setup_s)
    while len(res.tokens) < max_new:
        if drafter is None:
            dtree = DraftTree.chain([seq[-1]])
        else:
            drafter.set_sampling(sampler, committed, use_tree)
            t0 = time.perf_counter()
            dtree = (
                drafter.propose_tree(seq, committed, budget, width)
                if use_tree
                else drafter.propose(seq, committed, budget)
            )
            res.draft_s += time.perf_counter() - t0
        guesses = len(dtree.token_ids) - 1

        t0 = time.perf_counter()
        if use_tree:
            pos = treelib.positions(dtree.parent, committed)
            bias = treelib.mask(dtree.parent, committed)
            logits, present, hidden = model.forward(dtree.token_ids, past, committed, pos, bias)
        else:
            logits, present, hidden = model.forward(dtree.token_ids, past, committed)
        step_verify_s = time.perf_counter() - t0
        res.verify_s += step_verify_s
        res.steps += 1
        res.drafted += guesses
        if guesses:
            root_children = {dtree.token_ids[c] for c in dtree.children(0)}
            ranked = np.argpartition(-logits[0], min(4, len(logits[0]) - 1))[:5]
            res.root_proposals += 1
            res.root_top1 += int(int(logits[0].argmax()) in root_children)
            res.root_top5 += int(any(int(token) in root_children for token in ranked))
        if drafter is not None:
            drafter.note_cost(step_verify_s, guesses)

        committed_old = committed
        if use_tree:
            emitted, n_acc, path = treelib.accept(dtree, logits, sampler, committed)
            # Every accepted child was an opportunity. The next position joins the
            # denominator only when the final accepted node has another candidate.
            _record_conditional_acceptance(res, n_acc + int(bool(dtree.children(path[-1]))), n_acc)
            committed += len(path)  # anchor + accepted path nodes
            past = model.gather_kv(present, committed_old, path)
        else:
            emitted, n_acc = _accept_chain(dtree, logits, sampler, committed)
            _record_conditional_acceptance(res, min(guesses, n_acc + 1), n_acc)
            path = list(range(n_acc + 1))  # root + accepted guesses (contiguous)
            committed += n_acc + 1  # anchor + accepted guesses; bonus is next anchor
            past = model.rollback_kv(present, committed)
        res.accepted += n_acc

        if drafter is not None:
            # learn = the drafter digesting this step (observe caches verifier logits,
            # update adapts draft length). Timed apart from accept/KV so logit-reuse
            # drafters (STAND, Token Recycling) don't hide their O(vocab) cost in overhead.
            t0 = time.perf_counter()
            drafter.observe(dtree.token_ids, logits)
            if hidden is not None:  # side channel for representation-aware drafters
                acc_tokens = [dtree.token_ids[i] for i in path]
                drafter.observe_hidden(acc_tokens, hidden[path], committed_old)
            drafter.update(emitted)
            res.learn_s += time.perf_counter() - t0

        for i, t in enumerate(emitted):
            seq.append(t)
            res.tokens.append(t)
            if on_emit is not None:
                on_emit(t, i < n_acc)  # accepted draft guess vs bonus/correction
            if (eos is not None and t == eos) or len(res.tokens) >= max_new:
                return res
    return res


def generate_seeded(
    model: Verifier,
    prompt_ids: list[int],
    max_new: int,
    drafter: Drafter,
    budget: int = 8,
    eos: int | None = None,
    on_emit: Callable[[int, bool], None] | None = None,
    tree: bool = False,
    width: int = 2,
) -> GenResult:
    """Prototype seeded-root verification for greedy decoding.

    Full-prompt prefill supplies the first root from target logits.  Each forward
    then verifies only its descendants; its correction token becomes the known root
    for the following step.  This deliberately sits beside ``generate`` while we
    measure whether the alternate state convention improves LogitSpec acceptance.
    """
    if max_new <= 0:
        return GenResult(tokens=[])
    use_tree = tree and model.supports_tree
    seq = list(prompt_ids)
    t0 = time.perf_counter()
    past, committed, root_logits = model.prefill_seeded(seq)
    prefill_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    drafter.reset(seq)
    draft_setup_s = time.perf_counter() - t0
    res = GenResult(tokens=[], prefill_s=prefill_s, draft_setup_s=draft_setup_s)
    root = int(root_logits.argmax())
    seq.append(root)  # known target output, still uncommitted in the KV
    # LogitSpec's observation hook can use the prefill distribution to construct
    # children below this known root. Other drafters leave this side channel unused.
    drafter.observe([root], root_logits[None, :])
    res.tokens.append(root)
    drafter.update([root])
    if on_emit is not None:
        on_emit(root, True)
    if (eos is not None and root == eos) or len(res.tokens) >= max_new:
        return res

    while len(res.tokens) < max_new:
        t0 = time.perf_counter()
        dtree = (
            drafter.propose_tree(seq, committed, budget, width)
            if use_tree
            else drafter.propose(seq, committed, budget)
        )
        res.draft_s += time.perf_counter() - t0
        if dtree.token_ids[0] != root:
            raise ValueError("seeded drafter must retain the known root")
        guesses = len(dtree.token_ids) - 1

        t0 = time.perf_counter()
        if use_tree:
            pos = treelib.positions(dtree.parent, committed)
            bias = treelib.mask(dtree.parent, committed)
            logits, present, hidden = model.forward(dtree.token_ids, past, committed, pos, bias)
        else:
            logits, present, hidden = model.forward(dtree.token_ids, past, committed)
        step_verify_s = time.perf_counter() - t0
        res.verify_s += step_verify_s
        res.steps += 1
        res.drafted += guesses
        drafter.note_cost(step_verify_s, guesses)

        committed_old = committed
        if use_tree:
            emitted, n_acc, path = treelib.accept(dtree, logits, None, committed)
            _record_conditional_acceptance(res, n_acc + int(bool(dtree.children(path[-1]))), n_acc)
            committed += len(path)
            past = model.gather_kv(present, committed_old, path)
        else:
            emitted, n_acc = _accept_chain(dtree, logits, None, committed)
            _record_conditional_acceptance(res, min(guesses, n_acc + 1), n_acc)
            path = list(range(n_acc + 1))
            committed += n_acc + 1
            past = model.rollback_kv(present, committed)
        res.accepted += n_acc
        drafter.observe(dtree.token_ids, logits)
        # The final accepted node's logits selected ``emitted[-1]``. Associate that
        # fresh distribution with the correction/root for the next seeded step.
        drafter.observe([emitted[-1]], logits[path[-1] : path[-1] + 1])
        if hidden is not None:
            drafter.observe_hidden([dtree.token_ids[i] for i in path], hidden[path], committed_old)
        drafter.update(emitted)
        seq.extend(emitted)
        for i, token in enumerate(emitted):
            res.tokens.append(token)
            if on_emit is not None:
                on_emit(token, i < n_acc)
            if (eos is not None and token == eos) or len(res.tokens) >= max_new:
                return res
        root = emitted[-1]  # target correction; it is the next uncommitted root
    return res
