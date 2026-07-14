# Retrieval-based speculative decoding methods

Speculative decoding drafts several tokens cheaply, then verifies them in a single
model forward pass. The target model's accept rule keeps only the tokens it would have
produced on its own, so the output is *lossless*: it is bit-exact with plain greedy
decoding. The speedup comes from turning many one-token forward passes into fewer
multi-token ones.

*Retrieval-based* drafters get rid of the usual small draft model. Instead of running a
second network to guess the next tokens, they copy the guess straight from text the
model has already seen: the prompt, the model's own output so far, or a fixed corpus.
Because verification still decides what actually gets emitted, a wrong guess only costs
one wasted forward pass. It can never corrupt the output.

Every drafter here works on raw token ids (no model internals), so a single instance can
drive both the language model and the vision-language model. They differ along two axes:

1. **Where they retrieve from.** The prompt only, the live generation, and/or a
   persistent datastore.
2. **How they size and shape the draft.** Fixed length versus a length that adapts to
   how many tokens are getting accepted, and a single chain of guesses versus a
   branching tree of several candidates verified at once.

(The zoo also includes Token Recycling, which is not retrieval-based: it drafts from the
verifier's own logits. It is a useful baseline to compare against but is not covered
below.)

## Background terms

- **n-gram match.** Take the last few tokens of the context (an n-gram) and look for an
  earlier place where that same n-gram appeared. Whatever followed it there is a
  reasonable guess for what follows now. A longer matched n-gram is stronger evidence.
- **Suffix index.** A data structure that, given the current context, quickly returns the
  longest earlier run of tokens that matches the current suffix, plus what came next. It
  is the same idea as an n-gram match but without a fixed n, so it can find matches of
  any length.
- **Datastore.** A fixed body of text (a domain corpus, or the model's past outputs)
  indexed ahead of time, so drafts can be retrieved from knowledge that is not in the
  current prompt.
- **Draft budget.** A hard cap on how many tokens a drafter may propose per step. In this
  repo the default is 8, and every method's draft length is clamped to it.

## The methods

### PLD, Prompt Lookup Decoding
Find the longest trailing n-gram that appeared earlier in the context, and return the
tokens that followed it as a chain of guesses. Stateless and pure CPU.
- **Benefit:** zero setup and almost no overhead. Works well when the output repeats the
  prompt, as in summarization, retrieval-augmented generation, or code editing.
- **Drawback:** the draft length is fixed, so it over-drafts or under-drafts. It only ever
  finds one continuation, and it has nothing to copy on open-ended generation.

### CopySpec, earliest-occurrence continuation copying
Index fixed-length k-grams from the prompt and emitted history. When the current suffix
matches, copy the continuation after the *earliest* matching occurrence; new emitted
tokens extend the copy source for later steps.
- **Benefit:** targets redundant follow-up turns such as rewrites, corrections, and
  self-edits, where an earlier response is deliberately reused.
- **Difference from the paper:** this implementation is the token-only copying component.
  It omits CopySpec's optional model-drafter fallback so it remains a raw-token drop-in
  for both DejaVu backends. The verifier still supplies ordinary lossless correction.

### ANPD, Adaptive N-gram Parallel Decoding
The same match as PLD, but the draft *length* adapts to recent acceptance. When the last
draft was fully accepted it grows the length by one (the model might have taken more).
When some of it was rejected it shrinks the length toward how many tokens actually landed.
- **Benefit:** stops wasting verification on over-long drafts, which matters most on a
  small model where each forward pass is cheap and rejections are pure overhead.
- **Drawback:** still one candidate and still limited to prompt or history n-grams. The
  length estimate lags a sudden change in acceptance by a step or two.

### Lookahead, multi-candidate n-gram pool
The same longest n-gram match, but instead of one continuation it pools the top few
distinct continuations of that n-gram and verifies them as parallel branches of a tree.
In chain mode (no tree) it falls back to PLD.
- **Benefit:** when the same n-gram has been followed by different tokens at different
  points, verifying several candidates at once catches the right one. The pool is
  collected for free from the running context.
- **Drawback:** it only pays off with tree verification, and the extra branches cost verify
  width. If the n-gram only ever had one continuation, it adds nothing.

### LogitSpec, logit-conditioned n-gram retrieval
Reuse top candidates from verifier logits already computed on earlier steps. The
highest-ranked candidate starts the chain; its candidate-specific n-gram suffix then
retrieves a continuation. Under tree verification, several top-logit candidates are
siblings, each with its own retrieved continuation. The first decode step has no
cached verifier logit, so it deliberately makes no guess.
- **Benefit:** a likely next token creates a more specific lookup key, so retrieval can
  continue even when the current suffix alone has no exact hit. It adds no model forward
  or datastore setup cost.
- **Drawback:** cached candidates are keyed by a bounded recent token path, rather than a
  full hidden state. A cold path falls back to a token-level cache, which can still be
  stale for a repeated token. The verifier keeps this lossless, but acceptance still has
  to justify the extra tree width.

### N-Gram Trie, in-context continuation tree
Build a trie from every prompt n-gram and the tokens that followed it. A matching
suffix retrieves the complete continuation subtree, preserving branches below the
first next token instead of choosing one continuation immediately.
- **Benefit:** tree verification can hedge several full in-context futures while
  sharing their common prefix. It needs no datastore or model side channel.
- **Drawback:** building the trie is an online-once prompt cost, and its benefit is
  limited to prompts with repeated n-grams and genuinely branching continuations.

### REST, retrieval over a static datastore
Match the current suffix against a persistent datastore (a domain corpus supplied up
front, and/or completed generations added as the run proceeds) and return the
continuation. It ignores the in-progress output and draws only on the stored text.
- **Benefit:** it can supply tokens the current context does not contain yet. When the
  datastore matches the domain, this is a real advantage over prompt-only methods.
- **Drawback:** results depend entirely on datastore quality. A generic corpus adds
  distracting matches, and a domain or model mismatch makes it perform poorly. It is blind
  to repetition inside the current request.

### SuffixDecoding, online suffix index
Keep one growing index over everything seen in this run, both earlier generations and the
current prompt plus output. Take the longest suffix match, break ties by how often each
continuation occurred (frequency scoring), and let the draft length track the match length
(a longer match earns a longer, more confident draft).
- **Benefit:** no datastore to build or curate, and it gets stronger as it runs. Frequency
  scoring picks the likely continuation when several are possible. A robust default.
- **Drawback:** it starts each fresh request with little to go on, and it can only retrieve
  what has actually appeared during the run.

### Cacheback, bounded online n-gram cache
Cacheback stores a bounded least-recently-used (LRU) table of fixed-length **leader**
n-grams and their recent **follower** n-grams. Each emitted target token completes new
leader/follower pairs; on a cache hit, chain mode follows the most-recent follower and
tree mode recursively branches across several recent followers.
- **Drafting benefit:** lookup and eviction are bounded hot-path operations, so the
  drafter retains recent local patterns without growing an unbounded history index. It
  can adapt quickly when a request's topic or format changes.
- **Verification:** Cacheback never accepts its own tokens. The shared greedy verifier
  checks each copied path and emits a target-model correction at the first mismatch, so
  the text path remains lossless under greedy decoding.
- **Frozen-table option:** `Cacheback.from_frozen(path)` loads a versioned,
  tokenizer-specific table built offline with `python -m
  dejavuu.tools.build_cacheback_table`. Loading is an **online-once** cost, not a
  decode hot-path gain; benchmark cold, frozen, and warm modes separately.
- **Drawback:** it is cold at the beginning of a process and has only local online
  evidence. Small cache capacities can evict useful patterns; large capacities trade
  memory for fewer evictions.

### SAM-Decoding, static plus dynamic, longest match wins
Keep two suffix indexes at once: a static datastore (like REST) and the live generation
(like SuffixDecoding). Each step, draft from whichever one gives the longer suffix match.
The match length does double duty: it chooses the source and it caps the draft length.
- **Benefit:** the best of both. The datastore helps when it is relevant, and the live
  index carries the work when the datastore misses. It degrades gracefully to
  SuffixDecoding instead of failing outright the way REST can.
- **Drawback:** two indexes to maintain. A large generic datastore can produce a long but
  wrong match that wins the source selection, so retrieval precision drops as the store
  grows.

### ASAM and ASAM-verify, Adaptive SAM
SAM's two-source match with a smarter cap on draft length.
- `asam` uses an acceptance-calibrated cap: a running estimate of how many tokens the model
  tends to accept, applied under the match-length ceiling (the same adaptive idea as ANPD).
- `asam_verify` sizes each draft to the *measured* verification cost. It picks the length k
  that maximizes expected throughput, modeled as `E_accept(k) / (1 + g*k)`, where
  `E_accept(k)` is the expected number of accepted tokens under a simple geometric
  acceptance model and `g` is the ratio of marginal to fixed verify cost, learned online.
  When verification is cheap (`g` near 0) the best k pushes up to the ceiling; when it is
  expensive, k pulls in. One method therefore self-selects between aggressive and
  conservative instead of being fixed to one.
- **Benefit:** keeps SAM's retrieval quality while spending verification where it pays off.
  The verify-aware variant adapts to the hardware and model cost profile with no tuning.
- **Drawback:** more moving parts and more online state. The throughput model assumes
  geometric acceptance, which is only approximate, and it still inherits SAM's
  datastore-precision risk.

### ASD and ASD-verify, Adaptive Suffix Decoding
ASAM with the datastore switched off, so only the live-generation source is used. In other
words, SuffixDecoding's online index with ASAM's acceptance-calibrated (or verify-aware)
draft sizing on top.
- **Benefit:** the adaptive draft sizing of ASAM with none of the datastore curation or the
  datastore-precision risk. A strong no-setup default.
- **Drawback:** gives up the cross-domain knowledge a good datastore can provide, and like
  SuffixDecoding it starts cold at the beginning of each request.
- **Note on `asd_verify` and the backend cost curve:** the verify-aware sizer learns the
  verify cost online and shortens drafts when extra draft tokens are expensive to verify.
  This pays off only when that per-token cost is real. On a *launch-bound* backend (e.g.
  the int4_genai GPU chain path) verify is a step function — a large fixed jump from a
  1-token to a multi-token step, then nearly flat per extra token — so the marginal token
  is cheap and the sizer should draft long. Only drafted (multi-token) steps inform the
  cost fit for exactly this reason; folding the cheap 1-token steps in would misread the
  fixed jump as a steep per-token slope. On a backend whose verify genuinely grows with
  length (e.g. some CPU paths), the same sizer correctly pulls drafts in.

### PLD+, prompt lookup with hidden-state reranking
PLD finds the longest trailing n-gram and copies the *most recent* continuation. PLD+
gathers *every* earlier match of that n-gram, reranks them by cosine similarity of the
target model's hidden states (current context vs each match's context), and copies the
best-matching continuation. Chain-only.
- **Benefit:** when the same n-gram was followed by different tokens at different points,
  the hidden state picks the continuation whose context actually resembles the present one,
  instead of blindly taking the latest. Better draft quality with no extra verify width.
- **Drawback:** it needs a decoder that emits hidden states, which here is the SmolVLM
  tree+hidden export only; on a token-only decoder (the Gemma text path) the hidden memory
  stays empty and PLD+ degrades to plain PLD. The rerank layer is fixed (last, untuned).

### AdaPLD, adaptive PLD+ with semantic fallback and a draft tree
The current SOTA in this family. PLD+ retrieval plus two additions: a **semantic fallback**
(when no n-gram matches, retrieve the past position whose hidden state is most similar and
copy what followed it, fixing PLD's no-hit failure), and a **branched draft tree** (the
reranked main copy path, plus the top-K next tokens from the target's own verify logits
Token-Recycling style, each extended by one hidden-reranked successor token), verified in
one pass with tree attention.
- **Benefit:** the semantic fallback keeps drafting even when lexical matching finds
  nothing, and the logit branches hedge the first uncertain step, so it holds up on more
  open-ended output than the pure-lexical methods. Chain mode is ~PLD+.
- **Drawback:** same hidden-state requirement as PLD+ (SmolVLM only here); the tree branches
  need tree verification; the semantic fallback is a brute-force cosine over the memory (no
  ANN), and the retrieval query is one position stale (the anchor's own hidden isn't
  available until after the step).

## At a glance

| method | source | match | draft length | tree | needs datastore |
|---|---|---|---|---|---|
| PLD | prompt/history | longest n-gram | fixed | no | no |
| CopySpec | prompt/history | earliest fixed k-gram | fixed | no | no |
| PLD+ | prompt/history | longest n-gram, hidden-reranked | fixed | no | no (needs hidden states) |
| AdaPLD | prompt/history + semantic | n-gram or hidden similarity | fixed | yes (branches) | no (needs hidden states) |
| ANPD | prompt/history | longest n-gram | adaptive | no | no |
| Lookahead | prompt/history | longest n-gram | fixed | yes (pool) | no |
| LogitSpec | prior verifier logits + prompt/history | candidate-conditioned n-gram | fixed | yes | no |
| Cacheback | bounded online cache | recent leader/follower n-grams | follower-capped | yes | no |
| N-Gram Trie | prompt | continuation trie | fixed | yes (deep branches) | no |
| REST | datastore | longest suffix | match-capped | yes | yes |
| SuffixDecoding | live + history | freq-scored suffix | adaptive (match) | yes | no |
| SAM-Decoding | datastore + live | longer of the two | match-capped | yes | optional |
| ASAM | datastore + live | longer of the two | acceptance-calibrated | via SAM | optional |
| ASAM-verify | datastore + live | longer of the two | verify-cost-optimal | via SAM | optional |
| ASD | live + history | freq-scored suffix | acceptance-calibrated | no | no |
| ASD-verify | live + history | freq-scored suffix | verify-cost-optimal | no | no |

## Will retrieval even help? The entropy diagnostic

Retrieval drafting only pays off when the model's outputs repeat themselves, so a
draft copied from earlier text is likely to be accepted. You can measure this up front,
before running any benchmark, from about 100 sample outputs:

1. Build a suffix index over the outputs (`SuffixIndex`).
2. For each indexed context (each n-gram), take the distribution of the tokens seen to
   follow it and compute its entropy in bits.
3. Average those entropies, weighted by how often each context occurred.

Low weighted entropy means the next token is usually predictable from recent context,
which is exactly when a retrieved draft lands. High entropy means open-ended output that
a copy cannot guess, so the speedup collapses toward baseline. Rough reference points
from the SuffixDecoding paper: about 0.1 bits gave roughly 10x, 2.5 bits about 2x, and
above 3 bits only a modest gain.

`SuffixIndex.weighted_entropy()` computes this, and
`tools/specbench_entropy.py` runs it per Spec-Bench category on SmolVLM outputs, so you
can see which categories the retrieval drafters will speed up before committing to a full
method sweep.

## Choosing one

- **No datastore, want a strong default:** SuffixDecoding, or ASD / ASD-verify if you also
  want the adaptive draft sizing.
- **You have a relevant domain corpus:** SAM-Decoding, or ASAM. They use the store when it
  helps and fall back to the live index when it does not, so a mediocre store cannot hurt
  much.
- **The output mostly repeats the prompt and you want zero setup:** PLD, or ANPD if the
  drafts are over-shooting or under-shooting. On the SmolVLM hidden-emitting decoder, PLD+
  (better match selection) or AdaPLD (adds a semantic fallback and a draft tree) improve on
  plain PLD; on the token-only text decoder they degrade back to PLD.
- **Contexts that branch (one n-gram, several possible futures) with tree verification
  available:** Lookahead.
- **Avoid REST** unless the datastore is genuinely well-matched. Its lack of a live fallback
  makes it the most fragile method in this family (see the SAM-versus-REST behavior in the
  benchmarks).

All of these are lossless under greedy decoding. Retrieval only chooses which tokens to
propose and how many. The verifier alone decides what is actually emitted.

## Loose (lossy) verification

Everything above keeps the exact greedy accept rule: a drafted token is accepted only if
it *is* the token the target would have produced. Loose verification deliberately relaxes
that rule to accept more per step, trading token identity for speed. It is **opt-in and
off by default** -- the lossless path (the repo's headline guarantee) is untouched unless
you turn these knobs on. Two compose:

- `--accept-top-k K` (`accept_top_k`, default 1 = lossless): accept a drafted token when
  it is among the target's top-`K`, not just the argmax. Higher `K` accepts more but drifts
  further from the baseline.
- `--accept-min-prob-ratio R` (`accept_min_prob_ratio`, default 0 = off): a *plausibility
  gate*. Accept a non-argmax draft only when its probability is at least `R`x the argmax's
  -- i.e. a genuine near-tie the target itself was unsure about. Implemented as a logit
  margin (`logit[tok] >= max_logit + log R`), so it costs nothing. This is the lever that
  keeps meaning intact: it filters the confident-but-wrong substitutions that a bare top-k
  would wave through. (An older `--accept-entropy-gate` gates on full-vocab entropy instead;
  it is kept for comparison but superseded -- a peaked LM's entropy is near-zero almost
  everywhere, so it cannot tell a plausible runner-up from an unlikely one.)

Because loose output diverges from the greedy baseline on purpose, character-level metrics
(the default `text_similarity`) understate it badly: an early divergence that stays on-topic
scores near zero. Judge loose runs by **meaning** with `scripts/rescore.py`, which re-scores
existing bundles with static-embedding cosine (`uv pip install model2vec`).

**Measured (SPEED-Bench qualitative, Qwen3-0.6B fp32, tree, budget 4, top-k 3):**

| setting | acceptance | decode speedup | semantic vs baseline (pld / suffix / copyspec) |
| --- | --- | --- | --- |
| lossless (k=1) | 0.30 | 1.46x | 1.00 / 1.00 / 1.00 |
| top-k 3, no gate | 0.40 | 1.60x | 0.91 / 0.90 / 0.96 |
| top-k 3, ratio **0.3** | 0.38 | **1.66x** | **0.93 / 0.92 / 0.96** |
| top-k 3, ratio 0.7 | 0.36 | 1.53x | 0.93 / 0.93 / 0.96 |

Two things to take from this. First, the plausibility gate at `R = 0.3` is a *Pareto win*
over bare top-k: higher speedup, higher semantic fidelity, barely lower acceptance -- so it
is the recommended loose setting. Second, fidelity **plateaus** around 0.93 for `pld` /
`suffix_decoding`: tightening `R` past 0.3 buys almost no extra meaning while it costs
acceptance and speed. That residual gap is not implausible runner-ups (those are already
filtered) but locally-plausible near-ties that *compound* into divergent meaning over the
following tokens -- a per-position gate cannot see that. `copyspec` sits highest (~0.96)
because it copies contiguous spans and so diverges least to begin with.

**Recommended loose recipe:** `--accept-top-k 3 --accept-min-prob-ratio 0.3` on `copyspec`.
If you need `pld` / `suffix_decoding` above ~0.93 semantic, that calls for downstream
(windowed) verification rather than a sharper single-position gate -- see the "deferred
window" note in the tracking issue for FLy loose verification.
