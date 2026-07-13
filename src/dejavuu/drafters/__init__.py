"""The drafter zoo. Every drafter emits a `DraftTree` (chain, or a branching tree)
from raw token ids, so the same instances drive both the LLM and the VLM.

`DRAFTERS` is the method registry: it maps a name to a `DrafterSpec` (the factory plus
its capabilities) and is the single source of truth the CLI, the `DejaVu` API, and the
benchmark harnesses read from. `METHODS` is the full choice list, `"baseline"` (no
drafter) plus every registered name. Registering a drafter here wires it into all of
them and the conformance suite. This lives in the library, not the benchmark package,
so a plain `import dejavuu` never pulls the bench dependencies."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from dejavuu.drafters.adapld import AdaPLD
from dejavuu.drafters.anpd import ANPD
from dejavuu.drafters.asam import ASAM
from dejavuu.drafters.base import Drafter, DraftTree
from dejavuu.drafters.cacheback import Cacheback
from dejavuu.drafters.copyspec import CopySpec
from dejavuu.drafters.hybrid import Hybrid, PldRecycle, SuffixRecycle
from dejavuu.drafters.logit_spec import LogitSpec
from dejavuu.drafters.lookahead import Lookahead
from dejavuu.drafters.ngram_trie import NGramTrie
from dejavuu.drafters.pld_plus import PLDPlus
from dejavuu.drafters.prompt_lookup import PLD
from dejavuu.drafters.rest import REST
from dejavuu.drafters.sam_decoding import SAMDecoding
from dejavuu.drafters.stand import STAND
from dejavuu.drafters.suffix_decoding import SuffixDecoding
from dejavuu.drafters.suffix_index import SuffixIndex
from dejavuu.drafters.token_recycling import TokenRecycling

BASELINE = "baseline"  # the no-drafter method: plain autoregressive decoding


@dataclass(frozen=True)
class DrafterSpec:
    """One registry entry. `kwargs` are preset constructor arguments (so a method that
    is a preset of another class is explicit, not a bare `partial`); `needs_datastore`
    says whether `make_drafter` should feed it the static corpus."""

    factory: type[Drafter]
    kwargs: Mapping[str, object] = field(default_factory=dict)
    needs_datastore: bool = False
    doc: str = ""


DRAFTERS: dict[str, DrafterSpec] = {
    "pld": DrafterSpec(PLD, doc="prompt-lookup: longest suffix match in the context"),
    "copyspec": DrafterSpec(CopySpec, doc="earliest k-gram continuation copying"),
    "pld_plus": DrafterSpec(PLDPlus, doc="PLD + hidden-state reranking of matches"),
    "adapld": DrafterSpec(AdaPLD, doc="PLD+ + semantic fallback + branched tree"),
    "anpd": DrafterSpec(ANPD, doc="adaptive n-gram draft length"),
    "cacheback": DrafterSpec(Cacheback, doc="bounded LRU n-gram cache over emitted tokens"),
    "lookahead": DrafterSpec(Lookahead, doc="multi-candidate n-gram pool"),
    "logit_spec": DrafterSpec(
        LogitSpec, doc="verifier-logit candidates extended by n-gram retrieval"
    ),
    "ngram_trie": DrafterSpec(
        NGramTrie, doc="prompt n-gram continuation trie with deep tree branches"
    ),
    "token_recycling": DrafterSpec(TokenRecycling, doc="tree drafts from the verifier's logits"),
    "suffix_recycle": DrafterSpec(
        SuffixRecycle, doc="suffix index + verifier-logit fallback where retrieval is empty"
    ),
    "pld_recycle": DrafterSpec(PldRecycle, doc="PLD + verifier-logit fallback"),
    "rest": DrafterSpec(REST, needs_datastore=True, doc="retrieval from a static datastore"),
    "suffix_decoding": DrafterSpec(SuffixDecoding, doc="online suffix index over run history"),
    "sam_decoding": DrafterSpec(
        SAMDecoding, needs_datastore=True, doc="datastore + live generation, longer match wins"
    ),
    "stand": DrafterSpec(STAND, doc="logit n-gram candidates for sampled tree drafting"),
    "asam": DrafterSpec(ASAM, needs_datastore=True, doc="adaptive SAM: acceptance-calibrated cap"),
    "asam_verify": DrafterSpec(
        ASAM, kwargs={"verify_aware": True}, needs_datastore=True, doc="ASAM + verify-cost-aware"
    ),
    # asd / asd_verify are ASAM deliberately never given a datastore -> adaptive *suffix*
    # decoding (dynamic source only), so needs_datastore stays False.
    "asd": DrafterSpec(ASAM, doc="adaptive suffix decoding (ASAM, no datastore)"),
    "asd_verify": DrafterSpec(
        ASAM, kwargs={"verify_aware": True}, doc="adaptive suffix decoding + verify-cost-aware"
    ),
}

METHODS: list[str] = [BASELINE, *DRAFTERS]  # every valid --method value


def require_method(name: str) -> None:
    """Raise a clear error listing the valid names (not a bare KeyError) for an unknown
    method. Call it at any entry point before expensive work like loading a model."""
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; choose from: {', '.join(METHODS)}")


def make_drafter(name: str, datastore: list[list[int]] | None = None) -> Drafter | None:
    """Construct a drafter by name, feeding the static corpus to the ones that take one.
    One shared factory so the CLI and both benches agree. Returns None for `baseline`."""
    require_method(name)
    if name == BASELINE:
        return None
    spec = DRAFTERS[name]
    kwargs = dict(spec.kwargs)
    if datastore and spec.needs_datastore:
        kwargs["datastore"] = datastore
    return spec.factory(**kwargs)


__all__ = [
    "ANPD",
    "ASAM",
    "BASELINE",
    "DRAFTERS",
    "METHODS",
    "PLD",
    "REST",
    "STAND",
    "AdaPLD",
    "Cacheback",
    "CopySpec",
    "DraftTree",
    "Drafter",
    "DrafterSpec",
    "Hybrid",
    "LogitSpec",
    "Lookahead",
    "NGramTrie",
    "PLDPlus",
    "PldRecycle",
    "SAMDecoding",
    "SuffixDecoding",
    "SuffixIndex",
    "SuffixRecycle",
    "TokenRecycling",
    "make_drafter",
    "require_method",
]
