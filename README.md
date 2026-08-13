# Product Search Benchmark: BM25 vs. Dense vs. Hybrid

Does semantic search replace keyword search for product search? Measure it
instead of assuming.

Three retrieval architectures over 42,994 real product listings, scored against
233,448 human relevance judgments, with paired significance tests on every claim.

**Status: Phases 0–4 complete.** Aggregate results are reported first, but the
stratified results below them are the actual finding — the average is misleading
on its own.

---

## Headline

Exact-match relevance, 379 queries, nDCG@10 as the primary metric:

| Retriever | nDCG@10 | Precision@10 | Recall@100 | Index time | Index size | p50 latency |
|---|---|---|---|---|---|---|
| random control | 0.0049 | 0.0024 | 0.0033 | — | — | <1ms |
| bm25-naive | 0.5932 | 0.3784 | 0.5836 | 3.5s | 36 MB | 2.5ms |
| bm25-tuned | 0.7149 | 0.4631 | 0.6570 | 3.5s | 36 MB | 2.4ms |
| dense-minilm-full | 0.6598 | 0.4285 | 0.5687 | 43s | 152 MB | 12.0ms |
| dense-qwen3-full | 0.6738 | 0.4427 | 0.5912 | 820s | 566 MB | 26.7ms |
| **hybrid-rrf** | **0.7402** | **0.4873** | **0.6724** | — | — | 30.6ms |

All differences below are paired bootstrap, 10,000 resamples, on the same 379
queries.

| Comparison | Δ nDCG@10 | 95% CI | p | per-query record |
|---|---|---|---|---|
| hybrid vs. bm25-tuned | +0.0252 (+3.5%) | [+0.009, +0.042] | 0.0024 | 157 W / 111 L / 111 T |
| bm25-tuned vs. dense-qwen3 | +0.0412 (+6.1%) | [+0.018, +0.065] | 0.0006 | 188 W / 125 L / 66 T |
| hybrid vs. dense-qwen3 | +0.0664 (+9.9%) | [+0.052, +0.081] | 0.0001 | 221 W / 75 L / 83 T |

## The finding: the average is misleading

Splitting the 379 queries into terciles by **lexical coverage** — what fraction
of the query's words literally appear in the text of its relevant products —
shows that hybrid's entire aggregate gain comes from one third of the queries.

nDCG@10 by bucket (Exact-only):

| Retriever | low overlap (126) | mid overlap (126) | high overlap (127) |
|---|---|---|---|
| bm25-tuned | 0.5112 | 0.7240 | **0.9081** |
| dense-qwen3-full | 0.5071 | 0.6880 | 0.8250 |
| **hybrid-rrf** | **0.5696** | **0.7455** | 0.9040 |

Hybrid vs. tuned BM25, per bucket:

| Bucket | Δ nDCG@10 | 95% CI | p | W/L/T |
|---|---|---|---|---|
| low overlap | +0.0585 (+11.4%) | [+0.026, +0.091] | 0.0007 | 67 / 39 / 20 |
| mid overlap | +0.0215 (+3.0%) | [−0.002, +0.045] | 0.0764 | 61 / 36 / 29 |
| high overlap | −0.0040 (−0.5%) | [−0.031, +0.023] | 0.7597 | 29 / 36 / 62 |

**Fusion is worth 11.4% where the shopper's words are largely absent from the
correct products. On the easiest third it is a coin flip that slightly loses, and
it loses on more queries there than it wins (36 vs. 29).** 62 of those 127 queries
are exact ties — both retrievers return the same thing and fusion has nothing to
contribute.

The practical reading: fusion is not a free upgrade applied everywhere. Its value
is concentrated in the queries lexical matching already handles badly.

### Dense retrieval never won a single bucket

| Bucket | dense vs. bm25-tuned |
|---|---|
| low overlap | −0.8% |
| mid overlap | −5.0% |
| high overlap | −9.2% |

This is the result that contradicts the usual story. Dense retrieval does not
beat BM25 even in the bucket constructed to favour it — the queries whose words
do not appear in the relevant products. Embeddings were not better at conceptual
queries here; they were *differently wrong*.

Which means their value in this system is entirely as a **diversity source for
fusion**, not as a retriever. That is what the overlap diagnostic measures: 7.5
relevant products per query that BM25 never returns. The fusion gain comes from
disagreement, not from dense ranking better.

### What did not hold up

Regressed queries have mean coverage 0.753 against 0.706 for improved queries —
the predicted direction, but a small gap, and regressions appear across all three
buckets (39 / 36 / 36). So the bucket-level effect is real and significant while
the query-level prediction is not. Coverage predicts where fusion helps *on
average*; it does not predict which individual queries it will damage.

One methodological limit worth stating: coverage is computed against the known
relevant products, which a production system does not have at query time. It is a
diagnostic for understanding failure modes, not a routing signal. Routing on it
would require a proxy computable from the query alone.

## Four findings

**1. Tuning BM25 is worth ~21%, and skipping it invalidates the comparison.**
Adding an english analyzer (stemming, stopwords) and boosting the product name
over the description moved BM25 from 0.5932 to 0.7149 nDCG@10 — a gain that held
at both relevance bars. Untuned BM25 scores *below* both embedding models; tuned
BM25 beats both. Any benchmark that reports "embeddings beat keyword search"
without stating its analyzer and field weights has not established which of
those two baselines it used.

**2. Dense retrieval lost, and cost 235x more to build.** Qwen3-Embedding-0.6B
with full product text scored 0.6738 against tuned BM25's 0.7149 (p=0.0006).
Building the index took 820s against 3.5s, stored 566 MB against 36 MB, and
answered queries in 26.7ms against 2.4ms.

**3. Hybrid RRF won — and made 29% of queries worse.** Fusion beat BM25 by
+3.5% nDCG (p=0.0024). But the per-query record is 157 wins, 111 losses, 111
ties: nearly a third of queries got a *worse* result. The mean improves
reliably; individual queries do not. Phase 4 locates the gain: it is almost
entirely in the low-lexical-overlap tercile.

The mechanism is visible in the overlap diagnostic. The two top-100 lists share
only 31% of their contents (Jaccard), and dense uniquely surfaces **7.5 relevant
products per query that BM25 never retrieves at all**. The retrievers are not
redundant — they fail on different queries. That, not the aggregate score, is
what justifies running both.

**4. 27x the model parameters bought 2%.** MiniLM-L6-v2 (22M params, 2021)
scored 0.6598 against Qwen3-0.6B's 0.6738 — while embedding the corpus in 33s
rather than 768s, and truncating 1,396 documents at its 256-token limit. On this
corpus, model size was close to irrelevant.

## Two errors found mid-project, both favouring the same conclusion

Recorded because they are the most transferable part of this work.

- **The dense side was given less information than BM25.** The first dense runs
  embedded only name + class + category while tuned BM25 indexed the description
  too. Adding descriptions moved Qwen3 from 0.5994 to 0.6587 nDCG@10 — nearly
  10%. The original justification (avoiding truncation) was wrong: Qwen3's
  context is 32,768 tokens and zero documents were truncated.
- **`ef_search=100` was silently costing recall.** Raising HNSW search breadth to
  500 gained +3.1% nDCG@10 and +4.4% recall@100 for **+0.2ms** of latency. At
  43k vectors the graph is small enough that the recall/latency tradeoff barely
  exists; the default was simply too low.

Both mistakes understated dense retrieval. Uncorrected, the reported gap would
have been 19% instead of 6%. A comparison between two architectures is mostly a
comparison between two configurations, and the configuration deserves the same
scrutiny as the result.

One further variable, measured: Qwen3's query instruction prefix is worth
**+20.3% nDCG@10** on its own (0.6587 with, 0.5477 without). The exact wording
was not — matching the model card's template character-for-character changed
nDCG by −0.006 while improving recall@100 by +0.005, i.e. noise. Instruction
*presence* matters enormously; instruction phrasing did not.

## Data

[WANDS](https://github.com/wayfair/WANDS) (Wayfair Annotation Dataset, MIT):
42,994 products, 480 queries, 233,448 judgments labeled Exact / Partial /
Irrelevant.

Loaded as 231,873 unique query-product pairs, plus 1,559 duplicate pairs where
annotators agreed and 16 where they conflicted. Conflicts resolve to the higher
grade — immaterial at 16/233,448, but resolved explicitly rather than by
last-write-wins.

## Metrics, and why these ones

WANDS averages **357.3 relevant products per query** at the Partial+Exact bar,
0.83% of the corpus. That single fact rules out the metric this project
originally planned around:

- **recall@10 is not viable as a headline.** Per-query recall@10 is bounded by
  10/|relevant|, so a perfect retriever scores in the low single digits on most
  queries. Reported, but not quotable.
- **nDCG@10 and Precision@10** measure final ranking quality — what a shopper
  sees on page one. nDCG uses graded gains (`2^grade − 1`, TREC convention), so
  an Exact match is worth 3 and a Partial 1.
- **Recall@100** measures candidate-generation quality — what a second-stage
  reranker would have to work with.

That split maps onto an architecture decision rather than being bookkeeping: a
first stage is judged on recall@100, a reranker on nDCG@10.

### Two relevance bars

`--relevant-at 2` (Exact only) is primary: on a storefront a partial match is a
bad result, and it leaves 0.54 of headroom above the baseline rather than 0.18.
`--relevant-at 1` (Partial+Exact) is reported alongside.

**nDCG does not respond to `--relevant-at`** — it reads the graded labels
directly. Differences in nDCG between the two bars are a query-subset effect
(101 queries have no Exact match and drop out), not a relevance-definition
effect.

## Known limitations

- **Exact-only excludes 101 of 480 queries** — precisely those with no exact
  match, which skew vague and conceptual. That is the population where dense
  retrieval should be strongest, so the primary configuration may systematically
  disadvantage embeddings. Reporting both bars is a partial mitigation; Phase 4
  is the real one.
- **Per-bucket tests have ~126 queries each**, so two of the three hybrid-vs-BM25
  bucket comparisons are not significant at p<0.05. The low-overlap result
  (p=0.0007) is; the mid and high buckets are directional only.
- **Tercile boundaries are data-derived, not principled.** Cuts fall at coverage
  0.678 and 0.917. A different split would move the bucket means, though the
  monotonic trend is robust to where the lines go.
- **Hybrid is a better final ranker, not a better first stage.** At the
  Partial+Exact bar, hybrid recall@100 (0.4013) is marginally *below* BM25's
  (0.4022) while nDCG improved 4.9% — the fused top-100 is drawn from a
  200-candidate union, so dense items displace BM25 items at positions 60–100.
  A reranker should be fed BM25's candidates, not the fused list.
- **Judgments are pooled.** Unjudged pairs score 0. This understates every
  retriever, though not necessarily equally.
- **Tuned BM25 boosts (3.0 / 2.0 / 1.5 / 1.0) were reasoned, not swept.** BM25
  `k1=1.2, b=0.75` are Elasticsearch defaults, written explicitly into the
  mapping rather than inherited.
- **`rrf_k=60` is the Cormack et al. default and was not tuned.** Deliberately:
  sweeping it against these 379 queries until hybrid won would fit the constant
  to the test set.
- **Qwen3 pooling configuration is unverified.** The model card recommends
  `padding_side="left"` alongside flash_attention_2, and Qwen3-Embedding uses
  last-token pooling. Results respond to the query prefix as expected, which
  suggests pooling works, but sentence-transformers' handling of this was not
  independently confirmed.
- **Latency is localhost against single-node services.** Valid for comparing
  retrievers to each other; not a production latency claim.
- **379 queries is small once split into buckets.** Phase 4 differences will need
  the same paired bootstrap, not eyeballing.
- **One corpus, one domain.** Furniture listings with short titles and long
  descriptions. Nothing here should be extrapolated to a different catalogue
  without rerunning it.

## Phases

| Phase | Adds | Status |
|---|---|---|
| 0 | WANDS loader, graded metrics, random control | Complete |
| 1 | Docker Compose, Elasticsearch BM25, naive vs. tuned | Complete |
| 2 | sentence-transformer embeddings, pgvector + HNSW | Complete |
| 3 | RRF fusion, overlap diagnostic, paired bootstrap | Complete |
| 4 | Query stratification by lexical coverage, per-bucket significance | Complete |
| 5 | Parameter sweeps, embedding-model reindex cost | Planned |

A synthesised exact-identifier bucket was considered and dropped: tuned BM25
already scores 0.908 in the high-overlap tercile, so a bucket of self-generated
SKU queries would restate a result the real queries already show.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/download_wands.sh
docker compose up -d          # wait for both containers to report healthy

# baselines
python -m bench.run --retriever random --relevant-at 2
python -m bench.run --retriever bm25 --variant tuned --relevant-at 2
python -m bench.run --retriever bm25 --variant naive --relevant-at 2 --skip-index

# dense (embeds 43k products; ~1 min for minilm, ~13 min for qwen3)
python -m bench.run --retriever dense --model qwen3 --text full --ef-search 500 --relevant-at 2

# everything from here reuses the live index
./scripts/run_phase3.sh

# stratification: pure analysis over saved per-query scores, no retrieval
python -m bench.stratify --relevant-at 2 \
  results/bm25-tuned_rel2.json \
  results/dense-qwen3-full_rel2_ef500.json \
  "results/hybrid-rrf(bm25-tuned+dense-qwen3-full)_rel2_ef500.json"
```

`--skip-index` reuses the live Elasticsearch index and pgvector tables; use it
only when the corpus is unchanged. Every run writes a JSON record to
[`results/`](results/), including per-query scores for the significance tests.
Result filenames encode every variable that changes the numbers (model, text
config, `ef_search`, relevance bar, prefix override) so runs cannot silently
overwrite each other.

Sanity checks worth keeping: `docs_indexed` must equal 42,994, and the random
control must land near 10/42,994 for recall@10. Random Precision@10 came out
0.0092 against a predicted 357.3/42,994 = 0.0083.

## Design note

`bench/run.py` and `bench/retrievers/base.py` are meant to stop changing. Each
phase adds one class implementing the `Retriever` protocol and registers it in
`build_retriever()`. The measurement loop stays fixed so results stay comparable
across phases.