# Method fidelity matrix

This matrix distinguishes a paper's method from the repository adaptation. It is a
scope record, not a performance ranking: benchmark conclusions must name the measured
implementation and optimization level.

| Method | Reference | Repository adaptation and deliberate differences | Verification status |
|---|---|---|---|
| PLD | [original code](https://github.com/apoorvumang/prompt-lookup-decoding) | Bounded longest prompt/history n-gram search; adds a `DraftTree` tree path that branches matching first tokens. | Ordinary greedy lossless. |
| ANPD | [paper](https://arxiv.org/abs/2404.08698) | Prompt/history lookup with acceptance-driven draft-length adaptation; does not reproduce a paper-specific serving stack. | Ordinary greedy lossless. |
| Lookahead | [paper/code](https://github.com/hao-ai-lab/LookaheadDecoding) | Token-only multi-continuation pool, not Jacobi parallel decoding; uses this engine's tree verifier. | Ordinary greedy lossless. |
| Token Recycling | [paper/code](https://github.com/Luowaterbi/TokenRecycling) | Recycles verifier logits into sparse token-successor maps and uses Sequoia-style probability tree growth; this is an added topology optimization. | Ordinary greedy and seeded-sampling lossless. |
| REST | [paper/code](https://github.com/FasterDecoding/REST) | Fixed-order hash suffix index over supplied token documents, rather than the paper's full retrieval stack; completed generations are folded into the store between requests. | Ordinary greedy lossless. |
| SuffixDecoding | [paper](https://arxiv.org/abs/2411.04975) | Fixed-order hash index, not a production suffix tree; adds frequency scoring and tree continuations under the shared contract. | Ordinary greedy lossless. |
| SAM-Decoding | [paper/code](https://github.com/hemingkx/SAM-Decoding) | Two fixed-order suffix indexes, not a literal suffix automaton; chooses static or live source by matched length. | Ordinary greedy lossless. |
| ASAM / ASD | repository adaptation of SAM/Suffix | Acceptance- and verify-cost-aware draft caps are repository policies, not claimed paper reproductions. ASD disables the static store deliberately. | Ordinary greedy lossless. |
| PLD+ | repository method | Adds hidden-state cosine reranking; requires a hidden-emitting decoder and falls back to PLD otherwise. | Hidden states only choose drafts; verifier remains lossless. |
| AdaPLD | [paper](https://arxiv.org/abs/2606.05742) | Implements lexical reranking, brute-force semantic fallback, and Token-Recycling-style branches. ANN retrieval and any paper-specific tuned representation choices are not reproduced. | Ordinary greedy lossless; decoder numerical behavior may be near-lossless on quantized SmolVLM. |
| LogitSpec | [paper/code](https://github.com/smart-lty/LogitSpec) | Uses cached verifier-logit candidates plus n-gram continuations under the anchor-root engine. It does **not** implement the paper's full prefill/current-logit seeded-root protocol; the separate prototype measured slower and remains experimental. | Ordinary greedy lossless. |
| N-Gram Trie | [paper/code](https://github.com/mrlife219/Ngram-Trie) | Documented on its stacked PR: prompt-only continuation trie with deep tree branches. It omits the paper's retrieval/serving stack and is benchmarked separately. | Ordinary greedy lossless. |

## Backend qualification

The generic verifier gate is exact under greedy decoding. The quantized SmolVLM tree
export is not length-invariant, so its benchmark reports token match instead of calling
every speculative output bit-exact. That is a backend numerical limitation, not a
permission to weaken verifier acceptance.
